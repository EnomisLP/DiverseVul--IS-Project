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

    @property
    def config(self):
        """Expose the underlying backbone config to pyreft/peft."""
        return self.backbone.config

    @property
    def device(self) -> torch.device:
        """Expose the device where parameters reside for pyreft."""
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = cls_pool_last_hidden(hidden)
        else:
            pooled = mean_pool_last_hidden(hidden, attention_mask)
        logits = self.classification_head(pooled)
        
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
# EXP-5: HEFT (Hierarchical Efficient Fine-Tuning: LoRA -> freeze -> ReFT)
# =====================================================================
#
# HEFT is a two-phase procedure:
#   Phase 1 (LoRA): train a standard LoRA-adapted sequence classifier
#                    (use get_lora_model() below, already defined above).
#   Phase 2 (ReFT):  freeze *everything* learned in Phase 1 (LoRA adapters,
#                    classification head, backbone) and train a LoReFT
#                    intervention on top of the frozen, LoRA-adapted backbone
#                    (use attach_reft_to_lora_model() below).
#
# These two builders are meant to be driven by the two-phase training loop in
# exp5_heft.py -- they only construct models, they don't train anything.

def freeze_lora_parameters(model: nn.Module) -> None:
    """Freezes LoRA adapter parameters learned in Phase 1."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False


def reft_component_path(layer_target: int) -> str:
    """
    Dotted/bracket component path pyreft needs to locate the target layer's
    output *inside a peft-wrapped model*.

    peft.get_peft_model() re-nests the original module tree under
    "base_model.model.*" rather than preserving the original top-level
    attribute names. pyreft/pyvene resolve `component` strings via
    nn.Module.get_submodule(), which walks the *real* module registry (not
    Python attribute-forwarding), so a path like
    "backbone.encoder.layer[i].output" -- valid on the bare, unwrapped model --
    does not exist once the model has been wrapped with LoRA, and pyreft will
    fail to find it. It must be prefixed with "base_model.model." to match the
    wrapped model's actual module tree. (This mirrors the pattern pyreft's own
    docs use for peft-wrapped causal LMs: "base_model.model.model.layers[i]...".)
    """
    return f"base_model.model.backbone.encoder.layer[{layer_target}].output"


def attach_reft_to_lora_model(
    lora_model: nn.Module,
    reft_rank: int = 4,
    layer_target: int = 4,
    freeze_previous_phase: bool = True,
):
    """
    HEFT Phase 2. Takes an already Phase-1-trained LoRA model (as returned by
    get_lora_model / create_lora_sequence_classifier) and attaches a LoReFT
    intervention on top of it.

    Freezing behaviour:
      - `freeze_previous_phase=True` explicitly freezes the LoRA adapter
        parameters first (belt-and-braces).
      - pyreft.get_reft_model() *also* freezes every remaining parameter of
        the wrapped model by design -- that's the whole point of ReFT: adapt
        frozen representations via a small intervention instead of updating
        weights. So after this call, the classification head and backbone end
        up frozen too; only the newly added LoReFT intervention parameters
        are trainable. That matches "train LoRA, freeze it, apply ReFT".
    """
    import pyreft

    if freeze_previous_phase:
        freeze_lora_parameters(lora_model)

    hidden_size = int(lora_model.base_model.model.backbone.config.hidden_size)
    component_path = reft_component_path(layer_target)

    # NOTE: pyreft.ReftConfig expects `representations` as plain dict(s), NOT a
    # pyreft.RepresentationConfig object -- no such class exists in pyreft's
    # public API. Every real example in pyreft's own README/docs builds it this way.
    reft_config = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer_target,
                "component": component_path,
                "low_rank_dimension": reft_rank,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=hidden_size,
                    low_rank_dimension=reft_rank,
                ),
            }
        ]
    )

    # set_device=False prevents PyReft from probing custom module properties during init
    heft_model = pyreft.get_reft_model(lora_model, reft_config, set_device=False)

    # Force PyVene to single-stream / direct intervention mode (bypasses source-to-base requirement)
    heft_model.mode = "single"

    return heft_model