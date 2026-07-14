"""Shared NeoBERT model helpers for Case Study 2.

The functions here are imported by EXP-3/EXP-4/EXP-5 so that notebooks remain
thin orchestration layers and experiment logic is not duplicated.

Notes:
- The current NeoBERT Hugging Face repository uses custom code, so
  `trust_remote_code=True` is required for the encoder.
- The public NeoBERT model card lists a BERT tokenizer; we therefore load the
  tokenizer separately from `google-bert/bert-base-uncased` in experiment code.
- `xformers` is optional in the current NeoBERT repository, but some Colab
  environments still fail if the import is missing.  The compatibility shim
  below provides only the `xformers.ops.SwiGLU` symbol needed by NeoBERT.
"""

from __future__ import annotations

import sys
import types
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


DEFAULT_NEOBERT_MODEL = "chandar-lab/NeoBERT"
DEFAULT_NEOBERT_TOKENIZER = "google-bert/bert-base-uncased"


def install_neobert_compatibility_shims(verbose: bool = True) -> None:
    """Install a lightweight xformers.ops.SwiGLU shim when xformers is absent."""

    try:
        import xformers.ops  # noqa: F401
        return
    except Exception:
        pass

    if "xformers.ops" in sys.modules:
        return

    class SwiGLU(nn.Module):
        """Small PyTorch replacement for xformers.ops.SwiGLU.

        The signature is intentionally permissive because NeoBERT's remote code
        may pass optional xformers-specific keyword arguments.
        """

        def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, **kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features * 4
            out_features = out_features or in_features
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    xformers_module = types.ModuleType("xformers")
    ops_module = types.ModuleType("xformers.ops")
    ops_module.SwiGLU = SwiGLU
    xformers_module.ops = ops_module
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module

    if verbose:
        print("[CS2] Installed lightweight xformers.ops.SwiGLU compatibility shim.")


def _dtype_from_policy(dtype_policy: str):
    policy = str(dtype_policy).lower()
    if policy in {"auto", "float16", "fp16"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if policy in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if policy in {"float32", "fp32", "none"}:
        return torch.float32
    raise ValueError(f"Unknown dtype_policy: {dtype_policy}")


class NeoBertFrozenEncoder(nn.Module):
    """Frozen NeoBERT encoder returning CLS embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_NEOBERT_MODEL,
        *,
        dtype_policy: str = "float16",
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        install_neobert_compatibility_shims(verbose=True)

        torch_dtype = _dtype_from_policy(dtype_policy)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.eval()
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        return hidden_states[:, 0, :]


class NeoBertSequenceClassifier(nn.Module):
    """Simple NeoBERT sequence classifier retained for EXP-4/EXP-5 reuse."""

    def __init__(
        self,
        model_name: str = DEFAULT_NEOBERT_MODEL,
        freeze_backbone: bool = True,
        dtype_policy: str = "float16",
    ):
        super().__init__()
        self.encoder = NeoBertFrozenEncoder(
            model_name=model_name,
            dtype_policy=dtype_policy,
            freeze_backbone=freeze_backbone,
        )
        self.classification_head = nn.Linear(self.encoder.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classification_head(pooled_output.float())
        return logits.squeeze(-1)


def get_exp4_lora_model(model_name: str = DEFAULT_NEOBERT_MODEL, rank: int = 8, lora_alpha: int = 16):
    """Return a LoRA-ready classifier for EXP-4.

    NeoBERT uses a fused qkv projection in its attention blocks; therefore qkv
    is the default LoRA target rather than separate q_proj/v_proj modules.
    """

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install peft before using EXP-4 LoRA: pip install peft") from exc

    base_classifier = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=False)
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(base_classifier, peft_config)


def get_exp5_heft_model(lora_model_path, model_name: str = DEFAULT_NEOBERT_MODEL, reft_rank: int = 4):
    """Return a ReFT/HEFT model for EXP-5.

    Kept as a placeholder-compatible helper; final EXP-5 may need adjustment
    after EXP-4 LoRA adapter format is finalized.
    """

    try:
        import pyreft
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Install pyreft and peft before using EXP-5 HEFT/ReFT.") from exc

    base_classifier = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=False)
    lora_model = PeftModel.from_pretrained(base_classifier, lora_model_path)

    for param in lora_model.parameters():
        param.requires_grad = False

    hidden_size = getattr(getattr(lora_model, "config", None), "hidden_size", base_classifier.encoder.hidden_size)
    reft_config = pyreft.ReftConfig(
        intervention_type=pyreft.PositionControlledIntervention,
        embed_dim=hidden_size,
        low_rank_dimension=reft_rank,
        intervened_layers=[12, 16, 20, 24],
        positions="prefix",
        num_prefix_tokens=2,
    )
    return pyreft.get_reft_model(lora_model, reft_config)
