from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_CODE_MODEL = "huggingface/CodeBERTa-small-v1"
DEFAULT_CODE_TOKENIZER = "huggingface/CodeBERTa-small-v1"

# NeoBERT-250M backbone (Chandar Research Lab); ships as trust_remote_code on the Hub.
DEFAULT_NEOBERT_MODEL = "chandar-lab/NeoBERT"
DEFAULT_NEOBERT_TOKENIZER = "chandar-lab/NeoBERT"

# Substring match so NeoBERT forks/finetunes are still recognized.
_NEOBERT_NAME_HINTS = ("neobert",)


def _is_neobert_model(model_name: str) -> bool:
    """Check whether a model name refers to a NeoBERT-family checkpoint."""
    name = (model_name or "").lower()
    return any(hint in name for hint in _NEOBERT_NAME_HINTS)


def configure_huggingface_cache(hf_cache_dir: Optional[str] = None) -> None:
    """Set the Hugging Face cache/env variables used across this project's downloads."""
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
    """Resolve a torch dtype from a policy name and device."""
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


def _apply_neobert_config_overrides(config: Any) -> Any:
    """Disable NeoBERT sequence unpadding, since we pad batches instead of packing them."""
    candidate_flags = ("use_unpadding", "unpad_inputs", "unpad", "pack_sequences")
    matched = False
    for flag in candidate_flags:
        if hasattr(config, flag):
            setattr(config, flag, False)
            matched = True
    if not matched:
        warnings.warn(
            "[models] No known unpadding flag found on the NeoBERT config "
            f"(checked: {candidate_flags}); verify attention-mask correctness manually."
        )
    return config


def load_code_tokenizer(
    tokenizer_name: str = DEFAULT_CODE_TOKENIZER,
    hf_cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
):
    """Load the tokenizer for a code model, auto-detecting NeoBERT's trust_remote_code need."""
    configure_huggingface_cache(hf_cache_dir)
    if trust_remote_code is None:
        trust_remote_code = _is_neobert_model(tokenizer_name)
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        cache_dir=hf_cache_dir,
        trust_remote_code=trust_remote_code,
    )


def load_code_encoder(
    model_name: str = DEFAULT_CODE_MODEL,
    dtype_policy: str = "auto",
    device: Optional[str] = None,
    freeze: bool = True,
    hf_cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> nn.Module:
    """Load a code backbone encoder, applying NeoBERT-specific safeguards when needed."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_huggingface_cache(hf_cache_dir)
    dtype = _dtype_from_policy(dtype_policy, device)

    is_neobert = _is_neobert_model(model_name)
    if trust_remote_code is None:
        trust_remote_code = is_neobert

    kwargs: Dict[str, Any] = {"cache_dir": hf_cache_dir, "trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    if is_neobert:
        # NeoBERT's fused flash/memory-efficient SDPA backends crash with a
        # device-side CUDA assert on real (non-toy) batches on this
        # environment; force the math (unfused) backend instead.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # Patch the config before the backbone is instantiated.
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code
        )
        config = _apply_neobert_config_overrides(config)
        kwargs["config"] = config

    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over non-padded positions."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def cls_pool_last_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Take the [CLS]-position embedding."""
    return last_hidden_state[:, 0, :]


class CodeSequenceClassifier(nn.Module):
    """Backbone encoder + linear classification head over pooled embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_CODE_MODEL,
        num_labels: int = 1,
        freeze_backbone: bool = False,
        pooling: str = "mean",
        dtype_policy: str = "auto",
        hf_cache_dir: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        enforce_fp32_head: Optional[bool] = None,
    ) -> None:
        """Build the backbone and classification head."""
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = load_code_encoder(
            model_name=model_name,
            dtype_policy=dtype_policy,
            device=device,
            freeze=freeze_backbone,
            hf_cache_dir=hf_cache_dir,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_labels)
        self.pooling = pooling

        # Pooling and the classification head run in float32 regardless of
        # ambient autocast dtype, to avoid baking a bf16/fp16 NaN/Inf from
        # NeoBERT's attention stack into the trainable head.
        if enforce_fp32_head is None:
            enforce_fp32_head = _is_neobert_model(model_name)
        self.enforce_fp32_head = enforce_fp32_head

    @property
    def config(self):
        """Expose the underlying backbone config to peft."""
        return self.backbone.config

    @property
    def device(self) -> torch.device:
        """Expose the device where parameters reside."""
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode, pool, and classify a batch, returning logits."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state

        if self.enforce_fp32_head:
            hidden = hidden.float()
            attention_mask_for_pool = attention_mask.float()
        else:
            attention_mask_for_pool = attention_mask

        if self.pooling == "cls":
            pooled = cls_pool_last_hidden(hidden)
        else:
            pooled = mean_pool_last_hidden(hidden, attention_mask_for_pool)

        if self.enforce_fp32_head:
            # Disable autocast so the head matmul isn't downcast back to bf16/fp16.
            with torch.autocast(device_type=pooled.device.type, enabled=False):
                logits = self.classification_head(pooled.float())
        else:
            logits = self.classification_head(pooled)

        if logits.ndim > 1 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Report trainable vs. total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percent": float(100.0 * trainable / max(total, 1)),
    }


def infer_lora_target_modules(model: nn.Module) -> List[str]:
    """Guess which attention projection module names LoRA should target."""
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
    trust_remote_code: Optional[bool] = None,
):
    """Wrap a CodeSequenceClassifier with a LoRA adapter via peft."""
    from peft import LoraConfig, get_peft_model

    base = CodeSequenceClassifier(
        model_name=model_name,
        freeze_backbone=False,
        pooling=pooling,
        dtype_policy=dtype_policy,
        hf_cache_dir=hf_cache_dir,
        trust_remote_code=trust_remote_code,
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


def get_lora_model(
    model_name: str = DEFAULT_CODE_MODEL,
    rank: int = 8,
    lora_alpha: int = 16,
    pooling: str = "mean",
    trust_remote_code: Optional[bool] = None,
):
    """Build a LoRA-adapted sequence classifier for the given backbone."""
    return create_lora_sequence_classifier(
        model_name=model_name,
        rank=rank,
        lora_alpha=lora_alpha,
        pooling=pooling,
        trust_remote_code=trust_remote_code,
    )

