from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_CODE_MODEL = "huggingface/CodeBERTa-small-v1"
DEFAULT_CODE_TOKENIZER = "huggingface/CodeBERTa-small-v1"


def configure_huggingface_cache(hf_cache_dir: Optional[str] = None) -> None:
    if hf_cache_dir:
        hf_cache_dir = str(hf_cache_dir)
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_cache_dir) / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_huggingface_cache(hf_cache_dir)
    dtype = _dtype_from_policy(dtype_policy, device)

    kwargs: Dict[str, Any] = {"cache_dir": hf_cache_dir}
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
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def cls_pool_last_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


class CodeSequenceClassifier(nn.Module):
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
        
        # Safely squeeze dim -1 only if 2D tensor and single output label
        if logits.ndim > 1 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percent": float(100.0 * trainable / max(total, 1)),
    }


def infer_lora_target_modules(model: nn.Module) -> List[str]:
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
    return ["query", "value"]


def create_lora_sequence_classifier(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    pooling: str = "mean",
    dtype_policy: str = "auto",
    hf_cache_dir: Optional[str] = None,
):
    from peft import LoraConfig, get_peft_model

    base = CodeSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
        pooling=pooling,
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
        modules_to_save=["classification_head"],
    )
    return get_peft_model(base, config)


def get_lora_model(model_name: str = DEFAULT_CODE_MODEL, rank: int = 8, lora_alpha: int = 16, pooling: str = "mean"):
    return create_lora_sequence_classifier(
        model_name=model_name,
        rank=rank,
        lora_alpha=lora_alpha,
        pooling=pooling,
    )


# =====================================================================
# EXP-5: HEFT (Hierarchical Efficient Fine-Tuning: LoRA + ReFT)
# =====================================================================

def freeze_lora_parameters(model: nn.Module) -> None:
    """Freezes LoRA adapter parameters so only ReFT learns during Phase 2."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False


def create_heft_sequence_classifier(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    reft_rank: int = 4,
    layer_target: int = 4,
    lora_alpha: int = 16,
    pooling: str = "mean",
    dtype_policy: str = "auto",
    hf_cache_dir: Optional[str] = None,
    freeze_lora: bool = True,
):
    from peft import LoraConfig, get_peft_model
    import pyreft

    # 1. Base Classifier
    base = CodeSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
        pooling=pooling,
        dtype_policy=dtype_policy,
        hf_cache_dir=hf_cache_dir,
    )

    # 2. Coarse Weight Adaptation (LoRA)
    target_modules = infer_lora_target_modules(base)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        modules_to_save=["classification_head"],
    )
    lora_model = get_peft_model(base, lora_config)

    # 3. Freeze LoRA weights to establish coarse-to-fine hierarchy (Hill 2025)
    if freeze_lora:
        freeze_lora_parameters(lora_model)

    # 4. Fine Representation Steering (ReFT)
    hidden_size = int(base.backbone.config.hidden_size)
    reft_config = pyreft.ReftConfig(
        representations={
            "layer": layer_target,
            "component": "block_output",
            "low_rank_dimension": reft_rank,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=hidden_size,
                low_rank_dimension=reft_rank,
            ),
        }
    )

    heft_model = pyreft.get_reft_model(lora_model, reft_config)
    return heft_model


def get_heft_model(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    reft_rank: int = 4,
    layer_target: int = 4,
    heft_alpha: int = 16,
    pooling: str = "mean",
    freeze_lora: bool = True,
):
    return create_heft_sequence_classifier(
        model_name=model_name,
        rank=rank,
        reft_rank=reft_rank,
        layer_target=layer_target,
        lora_alpha=heft_alpha,
        pooling=pooling,
        freeze_lora=freeze_lora,
    )