"""Shared NeoBERT/model utilities for Case Study 2.

This module is shared by EXP-3, EXP-4, and EXP-5.  It must not contain
experiment-specific training loops.  It only provides common model/tokenizer
loading and optional adapter construction helpers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# Public NeoBERT checkpoint used for Case Study 2.
DEFAULT_NEOBERT_MODEL = "chandar-lab/NeoBERT"

# NeoBERT's released configuration uses a BERT-family tokenizer.  Loading the
# tokenizer directly avoids importing custom model code during tokenization.
DEFAULT_NEOBERT_TOKENIZER = "google-bert/bert-base-uncased"


def configure_huggingface_cache(cache_dir: Optional[str] = "/content/hf_cache") -> None:
    """Configure Hugging Face cache/download behavior explicitly.

    This reduces Colab download stalls and keeps model downloads outside Google
    Drive unless the caller intentionally chooses a Drive cache.
    """

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
        os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")

    # Enable the faster transfer package when installed.  If not installed,
    # Hugging Face silently falls back to normal download.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Xet transfer can be problematic in some Colab runtimes.  Disabling it is
    # often more reliable for a university-project notebook.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def install_xformers_swiglu_shim() -> None:
    """Install a minimal xformers.ops.SwiGLU shim when xformers is absent.

    Some NeoBERT remote-code versions import ``xformers.ops.SwiGLU``.  Colab
    often does not have xformers installed.  For inference/feature extraction,
    a small PyTorch implementation is enough to satisfy the import without
    forcing a heavy xformers installation.
    """

    if importlib.util.find_spec("xformers") is not None:
        return

    xformers_module = types.ModuleType("xformers")
    ops_module = types.ModuleType("xformers.ops")

    class SwiGLU(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(a) * b

    ops_module.SwiGLU = SwiGLU
    xformers_module.ops = ops_module
    sys.modules.setdefault("xformers", xformers_module)
    sys.modules.setdefault("xformers.ops", ops_module)


def _dtype_from_policy(dtype_policy: str):
    policy = str(dtype_policy).lower()
    if policy in {"float16", "fp16", "half"}:
        return torch.float16
    if policy in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if policy in {"float32", "fp32", "full"}:
        return torch.float32
    if policy in {"auto", "none"}:
        return None
    raise ValueError(f"Unknown dtype_policy: {dtype_policy}")


def load_neobert_tokenizer(tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER):
    """Load the shared tokenizer used across CS2 experiments."""

    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def load_neobert_encoder(
    model_name: str = DEFAULT_NEOBERT_MODEL,
    *,
    dtype_policy: str = "float16",
    device: Optional[torch.device] = None,
    freeze: bool = True,
    hf_cache_dir: Optional[str] = "/content/hf_cache",
):
    """Load NeoBERT encoder for embedding extraction or classifier wrapping."""

    configure_huggingface_cache(hf_cache_dir)
    install_xformers_swiglu_shim()

    torch_dtype = _dtype_from_policy(dtype_policy)
    kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModel.from_pretrained(model_name, **kwargs)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    if device is not None:
        model.to(device)
    return model


class NeoBertSequenceClassifier(nn.Module):
    """Reusable NeoBERT sequence classifier wrapper.

    EXP-4 and EXP-5 can adapt this wrapper.  EXP-3 normally uses only the frozen
    encoder embeddings, not this trainable classifier.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NEOBERT_MODEL,
        *,
        num_labels: int = 1,
        freeze_backbone: bool = False,
        dtype_policy: str = "float16",
        hf_cache_dir: Optional[str] = "/content/hf_cache",
    ):
        super().__init__()
        self.backbone = load_neobert_encoder(
            model_name=model_name,
            dtype_policy=dtype_policy,
            device=None,
            freeze=freeze_backbone,
            hf_cache_dir=hf_cache_dir,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_embedding = hidden_states[:, 0, :]
        logits = self.classification_head(cls_embedding)
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits


def count_trainable_parameters(model: nn.Module) -> dict:
    """Return total/trainable parameter counts."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


def infer_lora_target_modules(model: nn.Module) -> List[str]:
    """Infer likely LoRA target modules for NeoBERT.

    NeoBERT implementations may use fused qkv projection rather than separate
    q_proj/v_proj names.  This helper lets EXP-4 inspect the actual model
    module names rather than hard-coding old target names.
    """

    names = [name for name, _ in model.named_modules()]
    candidates = []
    for key in ["qkv", "Wqkv", "q_proj", "v_proj", "query", "value"]:
        if any(name.endswith(key) or f".{key}" in name for name in names):
            candidates.append(key)

    # Prefer fused qkv when available because current NeoBERT code commonly
    # exposes attention as a fused projection.
    if "qkv" in candidates:
        return ["qkv"]
    if "Wqkv" in candidates:
        return ["Wqkv"]
    if "q_proj" in candidates and "v_proj" in candidates:
        return ["q_proj", "v_proj"]
    if "query" in candidates and "value" in candidates:
        return ["query", "value"]

    raise RuntimeError(
        "Could not infer LoRA target modules from model names. "
        "Print model.named_modules() and set target_modules manually in EXP-4."
    )


def create_lora_sequence_classifier(
    *,
    model_name: str = DEFAULT_NEOBERT_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    dtype_policy: str = "float16",
    hf_cache_dir: Optional[str] = "/content/hf_cache",
):
    """Create a LoRA-adapted NeoBERT classifier for EXP-4.

    PEFT is imported lazily so EXP-3 does not require PEFT to be installed.
    """

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "PEFT is required for EXP-4 LoRA but is not installed. "
            "Install with: pip install peft"
        ) from exc

    base = NeoBertSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
        dtype_policy=dtype_policy,
        hf_cache_dir=hf_cache_dir,
    )
    if target_modules is None:
        target_modules = infer_lora_target_modules(base)

    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(lora_dropout),
        bias="none",
    )
    return get_peft_model(base, config)


def get_exp4_lora_model(**kwargs):
    """Backward-compatible EXP-4 alias."""

    return create_lora_sequence_classifier(**kwargs)
