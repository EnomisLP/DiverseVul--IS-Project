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

# EXP-7: NeoBERT-250M backbone (Chandar Research Lab). Plug-and-play replacement
# for the CodeBERTa backbone above -- same hidden size (768), but a 4,096-token
# RoPE/YaRE context window, SwiGLU activations, and Pre-RMSNorm. Ships as
# `trust_remote_code` model code on the Hub rather than a native `transformers`
# architecture, so it needs a couple of extra loading safeguards (see
# `_neobert_loading_overrides` below).
DEFAULT_NEOBERT_MODEL = "chandar-lab/NeoBERT"
DEFAULT_NEOBERT_TOKENIZER = "chandar-lab/NeoBERT"

# Model-name substrings that identify a NeoBERT-family checkpoint and therefore
# require `trust_remote_code=True` plus the safeguards below. Kept as a simple
# substring match (rather than a fixed set) so forks/finetunes of NeoBERT
# (e.g. "chandar-lab/NeoBERT", "someuser/NeoBERT-finetuned-...") are still
# picked up automatically.
_NEOBERT_NAME_HINTS = ("neobert",)


def _is_neobert_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(hint in name for hint in _NEOBERT_NAME_HINTS)


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


def _apply_neobert_config_overrides(config: Any) -> Any:
    """
    Mandatory Track-B safeguard (PDD sec. 5.2, GitHub Issue #7 on
    chandar-lab/NeoBERT -- "How is unpadding handled when unpacking?"):
    sequence-packing / unpadding in the reference NeoBERT code can leak
    attention across the pad boundary when the collator doesn't also emit
    packed cu_seqlens, which is exactly our setup (we pad batches instead of
    packing them). Forcing `use_unpadding=False` makes NeoBERT fall back to
    strict padded multi-head attention with the HF attention mask, which is
    the safe/correct path for this pipeline.

    The exact attribute name has moved around across NeoBERT code revisions,
    so we try the known aliases and only set whichever is actually present on
    this revision's config, rather than hard-failing.
    """
    candidate_flags = ("use_unpadding", "unpad_inputs", "unpad", "pack_sequences")
    matched = False
    for flag in candidate_flags:
        if hasattr(config, flag):
            setattr(config, flag, False)
            matched = True
    if not matched:
        warnings.warn(
            "[models] Could not find a known unpadding flag on the NeoBERT config "
            "(checked: %s). This revision of chandar-lab/NeoBERT may handle "
            "padding differently -- double check attention-mask correctness "
            "manually (see PDD sec. 5.2 / GitHub Issue #7)." % (candidate_flags,)
        )
    return config


def load_code_tokenizer(
    tokenizer_name: str = DEFAULT_CODE_TOKENIZER,
    hf_cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
):
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
        # Load + patch the config explicitly (rather than relying on
        # AutoModel.from_pretrained's implicit config loading) so the
        # unpadding override in _apply_neobert_config_overrides is guaranteed
        # to be in effect before the backbone is instantiated.
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
        trust_remote_code: Optional[bool] = None,
        enforce_fp32_head: Optional[bool] = None,
    ) -> None:
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

        # Mandatory Track-B safeguard (PDD sec. 5.2, GitHub Issue #11 on
        # chandar-lab/NeoBERT -- NaN training bug): pooling + the
        # classification head are forced to run in explicit float32,
        # regardless of the ambient autocast dtype, so a bf16/fp16 NaN/Inf
        # produced upstream in NeoBERT's attention stack doesn't get baked
        # into the (trainable) head via a half-precision matmul. This is a
        # wrapper-level mitigation -- it doesn't patch NeoBERT's own remote
        # code, it just keeps *our* downstream math numerically safe.
        if enforce_fp32_head is None:
            enforce_fp32_head = _is_neobert_model(model_name)
        self.enforce_fp32_head = enforce_fp32_head

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
            # Disable autocast for the head matmul so it isn't silently
            # downcast back to bf16/fp16 by the enclosing `torch.amp.autocast`
            # context in the training loop.
            with torch.autocast(device_type=pooled.device.type, enabled=False):
                logits = self.classification_head(pooled.float())
        else:
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
    trust_remote_code: Optional[bool] = None,
):
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
    return create_lora_sequence_classifier(
        model_name=model_name,
        rank=rank,
        lora_alpha=lora_alpha,
        pooling=pooling,
        trust_remote_code=trust_remote_code,
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

    NOTE: this hardcoded path assumes a BERT/RoBERTa-style module tree
    (`encoder.layer[i].output`), which is what CodeBERTa (EXP-5) has. It does
    NOT apply to NeoBERT (EXP-8) -- see resolve_reft_component_path below,
    which is architecture-aware and used for both.
    """
    return f"base_model.model.backbone.encoder.layer[{layer_target}].output"


def resolve_reft_component_path(lora_model: nn.Module, layer_target: int, model_name: str) -> str:
    """
    Architecture-aware version of reft_component_path, used by
    attach_reft_to_lora_model for both EXP-5 (CodeBERTa) and EXP-8 (NeoBERT).

    CodeBERTa is a standard BERT/RoBERTa-style module tree
    (`encoder.layer[i].output` per layer -- a dedicated submodule whose output
    IS the post-residual hidden state), so the hardcoded reft_component_path
    above is safe and is used as-is for backward compatibility with EXP-5.

    NeoBERT's module tree is different and NOT part of the public
    `transformers` library (it's loaded via `trust_remote_code`, and the
    Hugging Face Hub copy of its modeling code is not pinned/inspectable
    ahead of time the way a released `transformers` architecture is): its
    encoder stack is `transformer_encoder` (an nn.ModuleList of
    `EncoderBlock`s), and each block has no separate "output" submodule --
    the residual add happens inline in EncoderBlock.forward(), so the block
    itself (whose forward() return value IS the post-residual hidden state)
    is the right interception point, not some child of it.

    Rather than hardcode a guessed dotted path for NeoBERT (wrong by even one
    level of nesting and pyreft fails after Phase-1 LoRA has already finished
    training -- an expensive way to find out), this WALKS the model's actual
    module registry via named_modules() and finds whichever module path ends
    in "transformer_encoder.<layer_target>", so it's correct regardless of
    exactly how many wrapper levels (peft's "base_model.model.", our own
    "backbone.", NeoBERT's own internal "model." prefix, etc.) sit above it.
    Raises immediately with a clear message (before any Phase-2 training
    starts) if no matching module is found, rather than letting pyreft fail
    with a more cryptic KeyError deeper inside get_reft_model().
    """
    if not _is_neobert_model(model_name):
        return reft_component_path(layer_target)

    suffix = f"transformer_encoder.{layer_target}"
    matches = [name for name, _ in lora_model.named_modules() if name.endswith(suffix)]

    if not matches:
        available = sorted(
            {name for name, _ in lora_model.named_modules() if "transformer_encoder" in name}
        )
        raise RuntimeError(
            f"[reft] Could not find a module path ending in '{suffix}' in the "
            f"LoRA-wrapped NeoBERT model's module tree (needed for EXP-8's ReFT phase). "
            f"NeoBERT's Hub modeling code may have changed since this was written. "
            f"Module paths containing 'transformer_encoder' that WERE found: "
            f"{available[:20]}{'...' if len(available) > 20 else ''}. "
            f"Update resolve_reft_component_path in models.py to match the current structure."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"[reft] Ambiguous: found {len(matches)} module paths ending in '{suffix}' "
            f"({matches}) -- expected exactly one. Refusing to guess which one is correct."
        )
    return matches[0]


def attach_reft_to_lora_model(
    lora_model: nn.Module,
    reft_rank: int = 4,
    layer_target: int = 4,
    freeze_previous_phase: bool = True,
    model_name: str = DEFAULT_CODE_MODEL,
):
    """
    HEFT Phase 2. Takes an already Phase-1-trained LoRA model (as returned by
    get_lora_model / create_lora_sequence_classifier) and attaches a LoReFT
    intervention on top of it.

    `model_name` is used only to pick the right component-path resolution
    strategy (see resolve_reft_component_path) -- it does not affect which
    weights are loaded, since `lora_model` is already an instantiated model.

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
    component_path = resolve_reft_component_path(lora_model, layer_target, model_name)

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

    return heft_model