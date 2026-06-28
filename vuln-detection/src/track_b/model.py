"""
NeoBERT-250M (or ModernBERT-Base) with a 2-class linear classification head.

Bug fixes applied on every load:
  - Issue #11 (NaN): attention outputs and head kept in float32
  - Issue #7 (unpadding leakage): use_unpadding=False passed to model config

Class:   NeoBERTForVulnClassification
Factory: build_model(cfg)  — backbone swappable via cfg.model.name
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from src.utils import get_logger, load_config

_log = get_logger(__name__)

_DEFAULT_CONFIG_PATH = "configs/track_b.yaml"
_NEOBERT_MODEL_NAME  = "chandar-lab/NeoBERT"


class NeoBERTForVulnClassification(nn.Module):
    """Binary vulnerability classifier built on top of a BERT-style backbone.

    Applies two mandatory NeoBERT bug fixes (issues #7 and #11) and exposes
    helpers for differential LR setup and backbone freezing.

    Args:
        backbone: A pre-loaded HuggingFace transformer backbone.
        hidden_size: Dimensionality of the backbone's hidden states.
        num_labels: Number of output classes (default 2).
        dropout_prob: Dropout probability applied before the linear head.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        hidden_size: int,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout  = nn.Dropout(dropout_prob)
        # Head stays in float32 regardless of backbone precision (bug fix #11).
        self.classifier = nn.Linear(hidden_size, num_labels).float()
        self.num_labels  = num_labels

        _log.info(
            "Built NeoBERTForVulnClassification — hidden=%d  labels=%d",
            hidden_size, num_labels,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and optionally compute cross-entropy loss.

        Args:
            input_ids:      (B, L) token ids.
            attention_mask: (B, L) binary mask (1 = real token, 0 = padding).
            labels:         (B,) integer class labels. If supplied, ``loss`` is
                            included in the returned dict.

        Returns:
            Dict with keys ``logits`` (always) and ``loss`` (when labels given).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] hidden state — index 0 of the last hidden layer.
        # Cast to float32 before the head (bug fix #11: NaN prevention).
        cls_hidden = outputs.last_hidden_state[:, 0, :].float()
        cls_hidden = self.dropout(cls_hidden)
        logits     = self.classifier(cls_hidden)          # (B, num_labels)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    # ------------------------------------------------------------------
    # Parameter helpers for differential learning rates
    # ------------------------------------------------------------------

    def get_backbone_params(self) -> List[nn.Parameter]:
        """Return all trainable backbone parameters.

        Returns:
            List of parameters belonging to the backbone module.
        """
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_head_params(self) -> List[nn.Parameter]:
        """Return all classification-head parameters.

        Returns:
            List of parameters belonging to the dropout + linear head.
        """
        return list(self.dropout.parameters()) + list(self.classifier.parameters())

    # ------------------------------------------------------------------
    # Freeze / unfreeze backbone (EXP_2 probe experiment)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters.

        Called for EXP_2 (frozen probe): only the classification head trains.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        _log.info("Backbone frozen — head parameters only.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters.

        Called before LoRA or full fine-tuning phases.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        _log.info("Backbone unfrozen.")

    def count_trainable_params(self) -> int:
        """Return the number of currently trainable parameters.

        Returns:
            Integer count of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def build_model(
    cfg,
    *,
    num_labels: int = 2,
    dropout_prob: float = 0.1,
    device: Optional[Union[str, torch.device]] = None,
) -> NeoBERTForVulnClassification:
    """Instantiate a NeoBERTForVulnClassification from a track_b config namespace.

    Applies both mandatory NeoBERT bug fixes:
      - Issue #7  (unpadding leakage): forces ``use_unpadding=False`` in the
        model config before loading weights.
      - Issue #11 (NaN in fp16):       the classification head is kept in
        float32; backbone attention outputs are set to float32 via
        ``attention_output_dtype`` in the config where supported.

    The backbone is fully swappable via ``cfg.model.name``, so EXP_5
    (ModernBERT-Base) requires no code changes beyond the config.

    Args:
        cfg:          Config namespace loaded from ``configs/track_b.yaml``.
        num_labels:   Number of output classes (default 2).
        dropout_prob: Dropout probability before the head (default 0.1).
        device:       Target device. Auto-detected (CUDA > CPU) when None.

    Returns:
        Initialised ``NeoBERTForVulnClassification`` on ``device``.

    Raises:
        RuntimeError: If the backbone cannot be loaded from HuggingFace.
    """
    model_name     = cfg.model.name
    use_unpadding  = getattr(cfg.model, "use_unpadding", False)   # bug fix #7
    attn_out_dtype = getattr(cfg.model, "attention_output_dtype", "float32")  # bug fix #11

    _log.info("Loading backbone: %s", model_name)

    # --- Load and patch model config (bug fixes applied here) ---
    hf_config = AutoConfig.from_pretrained(model_name)

    # Bug fix #7: disable unpadding to prevent sequence-boundary leakage.
    if hasattr(hf_config, "use_unpadding"):
        hf_config.use_unpadding = use_unpadding
    else:
        _log.warning(
            "Config for '%s' has no 'use_unpadding' attribute; "
            "skipping bug-fix #7 patch.",
            model_name,
        )

    # Bug fix #11: force float32 attention outputs to prevent NaN in fp16 runs.
    if hasattr(hf_config, "attention_output_dtype"):
        hf_config.attention_output_dtype = attn_out_dtype
    else:
        _log.warning(
            "Config for '%s' has no 'attention_output_dtype' attribute; "
            "skipping bug-fix #11 config patch (head still kept in float32).",
            model_name,
        )

    backbone = AutoModel.from_pretrained(model_name, config=hf_config)

    # Derive hidden size from backbone config.
    hidden_size: int = hf_config.hidden_size

    model = NeoBERTForVulnClassification(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout_prob=dropout_prob,
    )

    # Resolve device.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_trainable = model.count_trainable_params()
    _log.info(
        "Model ready on %s — trainable params: %s",
        device,
        f"{n_trainable:,}",
    )
    return model


def build_model_from_config_path(
    config_path: Union[str, Path] = _DEFAULT_CONFIG_PATH,
    **kwargs,
) -> NeoBERTForVulnClassification:
    """Convenience wrapper: load config then call ``build_model``.

    Args:
        config_path: Path to ``configs/track_b.yaml``.
        **kwargs:    Forwarded to ``build_model``.

    Returns:
        Initialised ``NeoBERTForVulnClassification``.
    """
    cfg = load_config(config_path)
    return build_model(cfg, **kwargs)