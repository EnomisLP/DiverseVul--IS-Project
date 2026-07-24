"""
Shared CodeBERTa-small-v1 model utilities for Case Study 2.

Shared by:
  - EXP-3 Frozen Linear Probe
  - EXP-4 LoRA

CodeBERTa-small-v1 is a standard RoBERTa-architecture encoder pretrained on
CodeSearchNet source code. Unlike NeoBERT, it needs no trust_remote_code,
no xformers/SwiGLU compatibility shims, and no custom runtime patches --
it loads via plain transformers.AutoModel / AutoTokenizer.

This file must not contain experiment-specific training loops.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_CODE_MODEL = "huggingface/CodeBERTa-small-v1"
DEFAULT_CODE_TOKENIZER = "huggingface/CodeBERTa-small-v1"  # ships its own code-trained BPE tokenizer


def configure_huggingface_cache(hf_cache_dir: Optional[str] = None) -> None:
    """
    Configure Hugging Face cache and safer transfer behavior for Colab.
    Should be called before model/tokenizer loading.
    """
    if hf_cache_dir:
        hf_cache_dir = str(hf_cache_dir)
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_cache_dir) / "hub"))

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def _dtype_from_policy(dtype_policy: str, device: str) -> Optional[torch.dtype]:
    dtype_policy = (dtype_policy or "auto").lower()
    device = str(device)

    if dtype_policy == "float16":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype_policy == "bfloat16":
        return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if dtype_policy == "float32":
        return torch.float32
    if dtype_policy == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device == "cuda":
            return torch.float32
        return torch.float32

    raise ValueError(f"Unknown dtype_policy: {dtype_policy}")


def load_code_tokenizer(
    tokenizer_name: str = DEFAULT_CODE_TOKENIZER,
    hf_cache_dir: Optional[str] = None,
):
    configure_huggingface_cache(hf_cache_dir)
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        cache_dir=hf_cache_dir,
    )


def load_code_encoder(
    model_name: str = DEFAULT_CODE_MODEL,
    dtype_policy: str = "auto",
    device: Optional[str] = None,
    freeze: bool = True,
    hf_cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load the CodeBERTa encoder. Standard AutoModel loading -- no runtime
    patches required (unlike NeoBERT's custom architecture).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_huggingface_cache(hf_cache_dir)

    dtype = _dtype_from_policy(dtype_policy, device)
    is_local_path = Path(str(model_name)).exists()

    kwargs: Dict[str, Any] = {
        "cache_dir": hf_cache_dir,
        "local_files_only": bool(is_local_path),
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings using the attention mask.
    Preferred default for frozen probes: raw CLS-token embeddings from a
    frozen encoder without a pooling-specific pretraining objective are a
    known weak sentence representation (Reimers & Gurevych, 2019).
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def cls_pool_last_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Use first token (<s> / CLS-equivalent) embedding."""
    return last_hidden_state[:, 0, :]


class CodeSequenceClassifier(nn.Module):
    """
    Shared sequence classifier wrapper for LoRA-style fine-tuning (EXP-4).
    EXP-3's linear probe normally only uses frozen encoder embeddings
    extracted separately, but this class is reused wherever an end-to-end
    trainable classification head is needed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CODE_MODEL,
        num_labels: int = 1,
        freeze_backbone: bool = False,
        pooling: str = "mean",
        dtype_policy: str = "auto",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = load_code_encoder(
            model_name=model_name,
            dtype_policy=dtype_policy,
            device=device,
            freeze=freeze_backbone,
            hf_cache_dir=hf_cache_dir,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_labels)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = cls_pool_last_hidden(hidden)
        else:
            pooled = mean_pool_last_hidden(hidden, attention_mask)
        logits = self.classification_head(pooled)
        return logits.squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percent": float(100.0 * trainable / max(total, 1)),
    }


def infer_lora_target_modules(model: nn.Module) -> List[str]:
    """
    Infer LoRA target module names for a RoBERTa-family encoder (CodeBERTa).
    RoBERTa self-attention uses separate `query` / `value` Linear layers
    (not a fused qkv projection like NeoBERT), so this is the expected match.
    """
    module_names = [name for name, _ in model.named_modules()]

    candidate_sets = [
        ["query", "value"],
        ["q_proj", "v_proj"],
        ["qkv"],
        ["in_proj"],
    ]

    for candidates in candidate_sets:
        if all(any(name.endswith(candidate) or f".{candidate}" in name for name in module_names) for candidate in candidates):
            return candidates

    return ["query", "value"]


def create_lora_sequence_classifier(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    dtype_policy: str = "auto",
    hf_cache_dir: Optional[str] = None,
):
    """
    Shared LoRA model creation helper for EXP-4. PEFT is imported lazily.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError("PEFT is required for EXP-4 LoRA. Install with `pip install peft`.") from exc

    base = CodeSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
        pooling="mean",
        dtype_policy=dtype_policy,
        hf_cache_dir=hf_cache_dir,
    )

    target_modules = infer_lora_target_modules(base)

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(base, config)


def get_exp4_lora_model(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    dtype_policy: str = "auto",
    hf_cache_dir: Optional[str] = None,
):
    return create_lora_sequence_classifier(
        model_name=model_name,
        rank=rank,
        lora_alpha=lora_alpha,
        dtype_policy=dtype_policy,
        hf_cache_dir=hf_cache_dir,
    )