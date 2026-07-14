"""
Shared NeoBERT model utilities for Case Study 2.

This file is intentionally shared by:
  - EXP-3 Linear Probe
  - EXP-4 LoRA
  - EXP-5 HEFT/ReFT

It must not contain experiment-specific training loops.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from importlib.machinery import ModuleSpec
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_NEOBERT_MODEL = "chandar-lab/NeoBERT"
DEFAULT_NEOBERT_TOKENIZER = "google-bert/bert-base-uncased"


class NeoBertCompatibleSwiGLU(nn.Module):
    """
    PyTorch compatibility replacement for xformers.ops.SwiGLU.

    NeoBERT's custom model code calls:
        SwiGLU(in_features, hidden_features, out_features, bias=False)

    Some Colab/xformers installations do not accept the `bias` keyword,
    and some Colab runtimes contain a half-loaded xformers module with
    __spec__ = None. This class accepts NeoBERT's expected signature.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        hidden_features = in_features if hidden_features is None else hidden_features
        out_features = in_features if out_features is None else out_features

        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)

        self.w12 = nn.Linear(
            self.in_features,
            2 * self.hidden_features,
            bias=bias,
        )
        self.w3 = nn.Linear(
            self.hidden_features,
            self.out_features,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def install_neobert_runtime_patches(clear_cached_remote_modules: bool = True) -> None:
    """
    Install runtime compatibility patches BEFORE loading NeoBERT.

    This function deliberately does NOT use importlib.util.find_spec("xformers"),
    because Colab can leave xformers in sys.modules with __spec__ = None,
    which makes find_spec raise:
        ValueError: xformers.__spec__ is None

    The function force-installs a small xformers.ops stub containing a compatible
    SwiGLU class. This is sufficient for NeoBERT's feed-forward block.
    """

    if clear_cached_remote_modules:
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("transformers_modules."):
                # Force remote model.py to be re-imported after our xformers stub exists.
                del sys.modules[module_name]

    # Remove partial / incompatible xformers modules.
    for module_name in list(sys.modules.keys()):
        if module_name == "xformers" or module_name.startswith("xformers."):
            del sys.modules[module_name]

    # Create a clean package-like xformers stub.
    xformers_mod = types.ModuleType("xformers")
    xformers_mod.__package__ = "xformers"
    xformers_mod.__path__ = []
    xformers_mod.__spec__ = ModuleSpec(
        name="xformers",
        loader=None,
        is_package=True,
    )

    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.__package__ = "xformers"
    ops_mod.__spec__ = ModuleSpec(
        name="xformers.ops",
        loader=None,
        is_package=False,
    )
    ops_mod.SwiGLU = NeoBertCompatibleSwiGLU

    xformers_mod.ops = ops_mod

    sys.modules["xformers"] = xformers_mod
    sys.modules["xformers.ops"] = ops_mod


def configure_huggingface_cache(hf_cache_dir: Optional[str] = None) -> None:
    """
    Configure Hugging Face cache and safer transfer behavior for Colab.

    This should be called before model/tokenizer loading.
    """
    if hf_cache_dir:
        hf_cache_dir = str(hf_cache_dir)
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_cache_dir) / "hub"))

    # Avoid problematic Xet / transfer layers in some Colab runtimes.
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
        return torch.float16 if device == "cuda" else torch.float32

    raise ValueError(f"Unknown dtype_policy: {dtype_policy}")


def load_neobert_tokenizer(
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER,
    hf_cache_dir: Optional[str] = None,
):
    """
    Load the tokenizer. For NeoBERT we use the BERT tokenizer explicitly.
    """
    configure_huggingface_cache(hf_cache_dir)
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        cache_dir=hf_cache_dir,
    )


def load_neobert_encoder(
    model_name: str = DEFAULT_NEOBERT_MODEL,
    dtype_policy: str = "auto",
    device: Optional[str] = None,
    freeze: bool = True,
    hf_cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load the NeoBERT encoder robustly.

    model_name may be either:
      - remote repo id, e.g. 'chandar-lab/NeoBERT'
      - local snapshot path, e.g. '/content/neobert_local_snapshot'

    The xformers/SwiGLU patch is installed immediately before from_pretrained.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_huggingface_cache(hf_cache_dir)
    install_neobert_runtime_patches(clear_cached_remote_modules=True)

    dtype = _dtype_from_policy(dtype_policy, device)
    is_local_path = Path(str(model_name)).exists()

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": hf_cache_dir,
        "local_files_only": bool(is_local_path),
    }

    if dtype is not None:
        # transformers still accepts torch_dtype. Newer versions may warn;
        # the warning is harmless.
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
    Mean-pool token embeddings using attention mask.
    Useful fallback if CLS is not ideal.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def cls_pool_last_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Use first token embedding."""
    return last_hidden_state[:, 0, :]


class NeoBertSequenceClassifier(nn.Module):
    """
    Shared sequence classifier wrapper for LoRA / HEFT-style experiments.

    EXP-3 linear probe normally uses only frozen encoder embeddings, but EXP-4
    can reuse this class for end-to-end sequence classification.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NEOBERT_MODEL,
        num_labels: int = 1,
        freeze_backbone: bool = False,
        pooling: str = "cls",
        dtype_policy: str = "auto",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = load_neobert_encoder(
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
        if self.pooling == "mean":
            pooled = mean_pool_last_hidden(hidden, attention_mask)
        else:
            pooled = cls_pool_last_hidden(hidden)
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
    Infer likely LoRA target module names for NeoBERT.

    NeoBERT commonly uses fused qkv projections. We prefer 'qkv' if present,
    then fall back to common separate projection names.
    """
    module_names = [name for name, _ in model.named_modules()]

    candidate_sets = [
        ["qkv"],
        ["q_proj", "v_proj"],
        ["query", "value"],
        ["in_proj"],
    ]

    for candidates in candidate_sets:
        if all(any(name.endswith(candidate) or f".{candidate}" in name for name in module_names) for candidate in candidates):
            return candidates

    # Last-resort fallback: let caller inspect model if this fails.
    return ["qkv"]


def create_lora_sequence_classifier(
    model_name: str = DEFAULT_NEOBERT_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    dtype_policy: str = "auto",
    hf_cache_dir: Optional[str] = None,
):
    """
    Shared LoRA model creation helper for EXP-4.

    PEFT is imported lazily so EXP-3 does not require it.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError("PEFT is required for EXP-4 LoRA. Install with `pip install peft`.") from exc

    base = NeoBertSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
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


# Backwards-compatible name used by earlier EXP-4 scripts.
def get_exp4_lora_model(model_name: str = DEFAULT_NEOBERT_MODEL, rank: int = 8, lora_alpha: int = 16):
    return create_lora_sequence_classifier(
        model_name=model_name,
        rank=rank,
        lora_alpha=lora_alpha,
    )
