"""
Phase 2 of HEFT: apply LoreftIntervention to hidden states at post-RMSNorm
positions of selected layers, on top of a frozen LoRA backbone.

Public API
----------
apply_reft(model, cfg, r)        — wraps frozen LoRA model with ReFT interventions
save_reft_adapter(model, path)   — saves ReFT intervention weights + head only
load_reft_adapter(model, path)   — reloads ReFT weights in-place
get_reft_param_count(model)      — counts trainable ReFT parameters

Config read from configs/heft.yaml:
  reft.layers            [12, 16, 20, 24]
  reft.r                 [4, 8]   — callers iterate; each rank is one EXP_4 run
  reft.prefix_length     2
  reft.intervention_type LoreftIntervention
  reft.position          post_rmsnorm
  reft.lr                5.0e-5
  reft.epochs            3
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.track_b.model import NeoBERTForVulnClassification
from src.utils import get_logger, load_config

_log = get_logger(__name__)

_DEFAULT_HEFT_CONFIG = "configs/heft.yaml"

# Sentinel attribute written onto the model so train.py can always check
# whether ReFT has been applied without importing pyreft directly.
_REFT_ATTR = "_reft_handler"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_pyreft():
    """Import pyreft lazily and surface a clear error if it is missing."""
    try:
        import pyreft
        return pyreft
    except ImportError as exc:
        raise ImportError(
            "pyreft is required for Track B Phase 2. "
            "Install with: pip install pyreft"
        ) from exc


def _build_representations(
    layers: List[int],
    prefix_length: int,
    hidden_size: int,
    r: int,
) -> List[Dict]:
    """Build the pyreft representations list.

    Each entry targets one (layer, token-position) pair. We target the first
    ``prefix_length`` token positions (i.e. [CLS] + up to prefix_length-1
    subsequent tokens) at each of the specified layers.

    Args:
        layers:        Layer indices to intervene on.
        prefix_length: Number of prefix token positions targeted per layer.
        hidden_size:   Backbone hidden dimension.
        r:             ReFT rank (dimension of the intervention subspace).

    Returns:
        List of representation dicts accepted by ``pyreft.ReftConfig``.
    """
    pyreft = _import_pyreft()
    representations = []
    for layer_idx in layers:
        for token_pos in range(prefix_length):
            representations.append(
                {
                    "layer":             layer_idx,
                    "component":         "block_output",   # post-RMSNorm position
                    "low_rank_dimension": r,
                    "intervention":      pyreft.LoreftIntervention(
                        embed_dim=hidden_size,
                        low_rank_dimension=r,
                    ),
                }
            )
    return representations


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_reft(
    model: NeoBERTForVulnClassification,
    cfg,
    *,
    r: int,
) -> NeoBERTForVulnClassification:
    """Wrap a frozen LoRA model with LoreftIntervention at selected layers.

    The LoRA backbone and classification head weights must already be loaded
    and frozen (via ``lora.freeze_lora``) before this is called.

    Intervention formula applied at each targeted hidden state h:
        h_intervened = h + R^T (W_reft · h + b - R · h)
    where R is a low-rank orthogonal matrix of shape (r, hidden_size).

    Args:
        model: ``NeoBERTForVulnClassification`` with a frozen LoRA backbone.
        cfg:   Config namespace loaded from ``configs/heft.yaml``.
        r:     ReFT rank. Must be one of the values in ``cfg.reft.r``.

    Returns:
        The same model object, with a pyreft ``ReftModel`` handler stored as
        ``model._reft_handler``. The backbone itself is not replaced — pyreft
        registers forward hooks internally.

    Raises:
        ValueError: If ``r`` is not in ``cfg.reft.r``.
        RuntimeError: If LoRA parameters are still trainable (freeze first).
    """
    pyreft = _import_pyreft()

    valid_ranks: List[int] = list(cfg.reft.r)
    if r not in valid_ranks:
        raise ValueError(
            f"ReFT rank r={r} not in configured ranks {valid_ranks}. "
            "Update configs/heft.yaml or pass a valid rank."
        )

    # Safety check: LoRA must be frozen before Phase 2.
    lora_trainable = [
        name
        for name, p in model.named_parameters()
        if "lora_" in name and p.requires_grad
    ]
    if lora_trainable:
        raise RuntimeError(
            f"Found {len(lora_trainable)} trainable LoRA parameter(s). "
            "Call lora.freeze_lora(model) before apply_reft()."
        )

    # Derive hidden size from backbone config.
    backbone_config = model.backbone.config
    hidden_size: int = backbone_config.hidden_size

    layers:        List[int] = list(cfg.reft.layers)
    prefix_length: int       = cfg.reft.prefix_length

    representations = _build_representations(
        layers=layers,
        prefix_length=prefix_length,
        hidden_size=hidden_size,
        r=r,
    )

    reft_config  = pyreft.ReftConfig(representations=representations)
    reft_handler = pyreft.get_reft_model(model.backbone, reft_config)

    # Store handler on model; forward() in train.py will call it directly.
    setattr(model, _REFT_ATTR, reft_handler)

    n_reft   = get_reft_param_count(model)
    n_total  = sum(p.numel() for p in model.parameters())
    _log.info(
        "ReFT applied — r=%d  layers=%s  prefix=%d  "
        "trainable=%s / %s params (%.2f%%)",
        r,
        layers,
        prefix_length,
        f"{n_reft:,}",
        f"{n_total:,}",
        100.0 * n_reft / n_total if n_total else 0.0,
    )
    return model


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_reft_adapter(
    model: NeoBERTForVulnClassification,
    path: Union[str, Path],
) -> Path:
    """Save ReFT intervention weights and the classification head.

    Only the intervention parameters (R, W_reft, b) and the head are written.
    The backbone base weights and LoRA deltas are not saved here — they are
    stored by ``lora.save_lora_adapter`` and reloaded from HuggingFace.

    Args:
        model: Model with ``_reft_handler`` set by ``apply_reft``.
        path:  Directory to write adapter files into.

    Returns:
        Resolved ``Path`` of the output directory.

    Raises:
        RuntimeError: If ``apply_reft`` has not been called on this model.
    """
    _assert_reft_applied(model)

    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reft_handler = getattr(model, _REFT_ATTR)

    # pyreft exposes save_intervention for writing intervention state dicts.
    reft_handler.save(save_directory=str(out_dir), save_to_hf_hub=False)

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

    _log.info("ReFT adapter + head saved → %s", out_dir)
    return out_dir


def load_reft_adapter(
    model: NeoBERTForVulnClassification,
    path: Union[str, Path],
) -> NeoBERTForVulnClassification:
    """Reload ReFT intervention weights and the classification head from disk.

    ``apply_reft`` must already have been called on this model so that the
    intervention hooks are registered; only the learned parameters are updated.

    Args:
        model: Model with ``_reft_handler`` set by ``apply_reft``.
        path:  Directory previously written by ``save_reft_adapter``.

    Returns:
        The same model with updated ReFT and head weights.

    Raises:
        FileNotFoundError: If the adapter directory or head file does not exist.
        RuntimeError:      If ``apply_reft`` has not been called on this model.
    """
    _assert_reft_applied(model)

    in_dir = Path(path).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"ReFT adapter directory not found: {in_dir}")

    head_path = in_dir / "classifier_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {head_path}")

    pyreft      = _import_pyreft()
    reft_handler = getattr(model, _REFT_ATTR)

    # pyreft loads intervention parameters from the saved directory.
    reft_handler.load(load_directory=str(in_dir))

    # Reload classification head.
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])

    _log.info("ReFT adapter + head loaded ← %s", in_dir)
    return model


# ---------------------------------------------------------------------------
# ReFT-aware forward pass helper
# ---------------------------------------------------------------------------

def reft_forward(
    model: NeoBERTForVulnClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Run a forward pass through the ReFT handler and classification head.

    Replaces ``model.forward`` in the EXP_4 / EXP_5 training loop.  The ReFT
    handler applies interventions internally via registered hooks, then the
    modified hidden states flow into the head as normal.

    Args:
        model:          Model with ``_reft_handler`` set by ``apply_reft``.
        input_ids:      (B, L) token ids.
        attention_mask: (B, L) binary mask.
        labels:         (B,) integer class labels. If supplied, ``loss`` is
                        included in the returned dict.

    Returns:
        Dict with keys ``logits`` (always) and ``loss`` (when labels given).

    Raises:
        RuntimeError: If ``apply_reft`` has not been called on this model.
    """
    _assert_reft_applied(model)

    reft_handler = getattr(model, _REFT_ATTR)

    # Unit positions: [CLS] token (index 0) for each example in the batch.
    # pyreft expects a list of (batch_index, token_position) tuples.
    batch_size = input_ids.size(0)
    unit_locations = {"sources->base": (
        None,
        [[[0]] * batch_size],   # token position 0 ([CLS]) for all examples
    )}

    # The ReFT handler wraps the backbone; it returns a tuple
    # (intervened_outputs, collected_activations).
    intervened_outputs, _ = reft_handler(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        unit_locations=unit_locations,
        labels=None,            # loss computed below with class weights
        subspaces=None,
    )

    # Extract [CLS] hidden state from the intervened backbone output.
    cls_hidden = intervened_outputs.last_hidden_state[:, 0, :].float()
    cls_hidden = model.dropout(cls_hidden)
    logits     = model.classifier(cls_hidden)

    result: Dict[str, torch.Tensor] = {"logits": logits}
    if labels is not None:
        loss_fn    = nn.CrossEntropyLoss()
        result["loss"] = loss_fn(logits, labels)

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_reft_param_count(model: NeoBERTForVulnClassification) -> int:
    """Count trainable ReFT intervention parameters.

    Args:
        model: Model, possibly with ``_reft_handler`` set.

    Returns:
        Integer count of parameters whose names contain ``'intervention'`` and
        whose ``requires_grad`` flag is ``True``, or 0 if ReFT is not applied.
    """
    if not hasattr(model, _REFT_ATTR):
        return 0
    reft_handler = getattr(model, _REFT_ATTR)
    return sum(
        p.numel()
        for name, p in reft_handler.named_parameters()
        if p.requires_grad
    )


def _assert_reft_applied(model: NeoBERTForVulnClassification) -> None:
    """Raise RuntimeError if apply_reft() has not been called."""
    if not hasattr(model, _REFT_ATTR):
        raise RuntimeError(
            "ReFT has not been applied to this model. "
            "Call apply_reft(model, cfg, r=...) first."
        )


def build_reft_model(
    lora_model: NeoBERTForVulnClassification,
    heft_config_path: Union[str, Path] = _DEFAULT_HEFT_CONFIG,
    *,
    r: int,
) -> NeoBERTForVulnClassification:
    """Convenience wrapper: load heft config and apply ReFT in one call.

    Args:
        lora_model:       A ``NeoBERTForVulnClassification`` with frozen LoRA.
        heft_config_path: Path to ``configs/heft.yaml``.
        r:                ReFT rank (must be in ``heft.yaml`` reft.r list).

    Returns:
        Model with ReFT interventions registered on its backbone.
    """
    cfg = load_config(heft_config_path)
    return apply_reft(lora_model, cfg, r=r)