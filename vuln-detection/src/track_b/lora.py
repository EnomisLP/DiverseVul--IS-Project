"""
Phase 1 of HEFT: inject LoRA adapters into W_q and W_v of NeoBERT (or any
compatible backbone).

Public API
----------
apply_lora(model, cfg, r)       — wraps backbone with a PEFT LoRA adapter
save_lora_adapter(model, path)  — saves adapter weights only (never full model)
load_lora_adapter(model, path)  — reloads adapter weights in-place
freeze_lora(model)              — freezes all LoRA parameters before Phase 2
get_lora_param_count(model)     — counts trainable LoRA parameters

Config read from configs/heft.yaml:
  lora.target_modules  [query, value]
  lora.r               [8, 16]   — callers iterate; each rank is one EXP_3 run
  lora.lora_alpha      16
  lora.lora_dropout    0.05
  lora.bias            none
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft import set_peft_model_state_dict

from src.track_b.model import NeoBERTForVulnClassification
from src.utils import get_logger, load_config

_log = get_logger(__name__)

_DEFAULT_HEFT_CONFIG = "configs/heft.yaml"


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_lora(
    model: NeoBERTForVulnClassification,
    cfg,
    *,
    r: int,
) -> NeoBERTForVulnClassification:
    """Inject LoRA adapters into the backbone's W_q and W_v projections.

    The classification head is intentionally excluded from adaptation; LoRA
    modifies only the attention projections listed in ``cfg.lora.target_modules``.

    Args:
        model: An initialised ``NeoBERTForVulnClassification``. Its backbone
               will be wrapped in-place by PEFT.
        cfg:   Config namespace loaded from ``configs/heft.yaml``.
        r:     LoRA rank. Must be one of the values in ``cfg.lora.r``.

    Returns:
        The same model object with its backbone replaced by a ``PeftModel``.

    Raises:
        ValueError: If ``r`` is not in ``cfg.lora.r``.
    """
    valid_ranks: List[int] = list(cfg.lora.r)
    if r not in valid_ranks:
        raise ValueError(
            f"Rank r={r} not in configured ranks {valid_ranks}. "
            "Update configs/heft.yaml or pass a valid rank."
        )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(cfg.lora.target_modules),
        r=r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        inference_mode=False,
    )

    # Wrap only the backbone; the classification head remains a plain nn.Linear.
    model.backbone = get_peft_model(model.backbone, lora_config)

    n_lora = get_lora_param_count(model)
    n_total = sum(p.numel() for p in model.parameters())
    _log.info(
        "LoRA applied — r=%d  alpha=%d  targets=%s  "
        "trainable=%s / %s params (%.2f%%)",
        r,
        cfg.lora.lora_alpha,
        list(cfg.lora.target_modules),
        f"{n_lora:,}",
        f"{n_total:,}",
        100.0 * n_lora / n_total if n_total else 0.0,
    )
    return model


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_lora_adapter(
    model: NeoBERTForVulnClassification,
    path: Union[str, Path],
) -> Path:
    """Save LoRA adapter weights and the classification head — never the full backbone.

    The backbone base weights are NOT saved (they are reloaded from HuggingFace
    at inference time). Only the LoRA delta matrices and the head are persisted.

    Args:
        model: Model whose backbone is a ``PeftModel``.
        path:  Directory to write adapter files into.

    Returns:
        Resolved ``Path`` of the output directory.

    Raises:
        TypeError: If the backbone is not a ``PeftModel`` (i.e. LoRA was not
                   applied before calling this function).
    """
    if not isinstance(model.backbone, PeftModel):
        raise TypeError(
            "model.backbone is not a PeftModel. "
            "Call apply_lora(model, cfg, r=...) first."
        )

    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Adapter delta weights (LoRA A/B matrices only).
    model.backbone.save_pretrained(str(out_dir))

    # Classification head (always float32, small).
    head_path = out_dir / "classifier_head.pt"
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "hidden_size":           model.classifier.in_features,
            "num_labels":            model.classifier.out_features,
        },
        head_path,
    )

    _log.info("LoRA adapter + head saved → %s", out_dir)
    return out_dir


def load_lora_adapter(
    model: NeoBERTForVulnClassification,
    path: Union[str, Path],
) -> NeoBERTForVulnClassification:
    """Reload LoRA adapter weights and the classification head from disk.

    The model's backbone must already have LoRA applied (via ``apply_lora``)
    before calling this function; only the learned weight deltas are updated.

    Args:
        model: Model whose backbone is a ``PeftModel``.
        path:  Directory previously written by ``save_lora_adapter``.

    Returns:
        The same model with updated adapter and head weights.

    Raises:
        FileNotFoundError: If the adapter directory or head file does not exist.
        TypeError: If the backbone is not a ``PeftModel``.
    """
    if not isinstance(model.backbone, PeftModel):
        raise TypeError(
            "model.backbone is not a PeftModel. "
            "Call apply_lora(model, cfg, r=...) before load_lora_adapter()."
        )

    in_dir = Path(path).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {in_dir}")

    head_path = in_dir / "classifier_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {head_path}")

    # Reload LoRA delta weights.
    adapter_weights = torch.load(
        in_dir / "adapter_model.bin",
        map_location="cpu",
        weights_only=True,
    ) if (in_dir / "adapter_model.bin").exists() else torch.load(
        in_dir / "adapter_model.safetensors",
        map_location="cpu",
        weights_only=True,
    )
    set_peft_model_state_dict(model.backbone, adapter_weights)

    # Reload classification head.
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])

    _log.info("LoRA adapter + head loaded ← %s", in_dir)
    return model


# ---------------------------------------------------------------------------
# Freeze (called before Phase 2 ReFT)
# ---------------------------------------------------------------------------

def freeze_lora(model: NeoBERTForVulnClassification) -> None:
    """Freeze all LoRA adapter parameters.

    Called at the start of EXP_4 / EXP_5 (Phase 2) to lock the LoRA deltas
    before ReFT interventions are applied on top.  The backbone base weights
    are already frozen; this additionally freezes the A/B adapter matrices.

    Args:
        model: Model whose backbone is a ``PeftModel`` with loaded LoRA weights.

    Raises:
        TypeError: If the backbone is not a ``PeftModel``.
    """
    if not isinstance(model.backbone, PeftModel):
        raise TypeError(
            "model.backbone is not a PeftModel. "
            "Call apply_lora() before freeze_lora()."
        )

    frozen_count = 0
    for name, param in model.backbone.named_parameters():
        if "lora_" in name:
            param.requires_grad = False
            frozen_count += 1

    _log.info(
        "LoRA frozen — %d adapter parameter tensors locked for Phase 2.",
        frozen_count,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_lora_param_count(model: NeoBERTForVulnClassification) -> int:
    """Count the number of trainable LoRA parameters in the backbone.

    Args:
        model: Model whose backbone may be a ``PeftModel``.

    Returns:
        Integer count of parameters whose names contain ``'lora_'`` and whose
        ``requires_grad`` flag is ``True``.
    """
    if not isinstance(model.backbone, PeftModel):
        return 0
    return sum(
        p.numel()
        for name, p in model.backbone.named_parameters()
        if "lora_" in name and p.requires_grad
    )


def build_lora_model(
    base_model: NeoBERTForVulnClassification,
    heft_config_path: Union[str, Path] = _DEFAULT_HEFT_CONFIG,
    *,
    r: int,
) -> NeoBERTForVulnClassification:
    """Convenience wrapper: load heft config and apply LoRA in one call.

    Args:
        base_model:       A freshly built ``NeoBERTForVulnClassification``.
        heft_config_path: Path to ``configs/heft.yaml``.
        r:                LoRA rank (must be in ``heft.yaml`` lora.r list).

    Returns:
        Model with LoRA applied to its backbone.
    """
    cfg = load_config(heft_config_path)
    return apply_lora(base_model, cfg, r=r)