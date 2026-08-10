from __future__ import annotations

import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import (
    get_lora_model,
    attach_reft_to_lora_model,
    count_trainable_parameters,
    DEFAULT_CODE_MODEL,
)
from case_study_2.training_utils import EarlyStoppingConfig, run_training_with_early_stopping


def forward_lora(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Phase 1 forward pass: plain LoRA-adapted classifier, no interventions."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    if logits.ndim > 1 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return logits


def forward_reft(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Phase 2 forward pass: pyreft-wrapped model with a LoReFT intervention.

    LoReFT is "sourceless" -- it edits the base model's own hidden state directly,
    it doesn't need a counterfactual source example. pyvene has a dedicated code
    path for exactly this case: passing unit_locations as {"base": <positions>}
    (a flat, non-batched list of token positions) tells pyvene's own broadcasting
    logic (IntervenableModel._broadcast_unit_locations, under the default
    mode="parallel" that pyreft sets) to expand it across the batch and across
    every intervention itself. We don't need to hand-build a
    (num_interventions, batch, seq_len) tensor, and there is no "single" mode in
    pyvene -- only "parallel" and "serial" are supported.
    """
    seq_len = input_ids.shape[1]
    positions = list(range(seq_len))

    outputs = model(
        base={"input_ids": input_ids, "attention_mask": attention_mask},
        unit_locations={"base": positions},
    )

    logits = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
    if logits.ndim > 1 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return logits


def _run_training_epochs(*_args, **_kwargs):
    raise NotImplementedError(
        "_run_training_epochs was replaced by "
        "case_study_2.training_utils.run_training_with_early_stopping (see FIX notes "
        "in that module and in train_heft_model below). This stub is kept only so a "
        "stale import doesn't fail silently with an AttributeError instead of a clear message."
    )


def train_heft_model(
    train_df,
    val_df,
    tokenizer,
    rank,
    lora_epochs=2,
    reft_epochs=2,
    batch_size=16,
    grad_accum_steps=2,
    eval_batch_size=32,
    num_workers=2,
    device="cuda",
    hf_cache_dir=None,
    code_column="normalized_code",
    max_length=512,
    verbose=True,
    log_every_steps=50,
    log_prefix="",
    reft_rank=4,
    layer_target=4,
    heft_alpha=16,
    lora_lr=2e-4,
    reft_lr=2e-4,
    # --- FIX (see case_study_2/training_utils.py header for full context) --
    # `lora_epochs`/`reft_epochs` are now CEILINGS: each phase early-stops on
    # its own validation PR-AUC (computed with THAT phase's forward pass --
    # plain LoRA for phase 1, the pyreft-wrapped model for phase 2) and
    # restores its own best checkpoint before the next phase starts, instead
    # of always running a fixed epoch count and handing whatever the last
    # epoch produced to the next phase. Separate patience/min_epochs for LoRA
    # vs ReFT because empirically ReFT converges much faster (val loss was
    # already flat by epoch 2 in the original run) than LoRA.
    early_stopping=True,
    min_delta=1e-4,
    lora_patience=2,
    lora_min_epochs=2,
    reft_patience=2,
    reft_min_epochs=1,
):
    """
    HEFT training: Phase 1 trains LoRA to convergence, Phase 2 freezes
    everything from Phase 1 and trains a LoReFT intervention on top.

    Returns (val_scores, heft_model) where heft_model is the final Phase-2
    (pyreft-wrapped) model, matching what run_exp5_* in exp5_nested_pipeline.py
    expects (it calls .save_pretrained() on the returned model).
    """
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=True, num_workers=num_workers, code_column=code_column,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=eval_batch_size, max_length=max_length,
        shuffle=False, num_workers=num_workers, code_column=code_column,
    )

    is_cuda = (device == "cuda") or (hasattr(device, "type") and device.type == "cuda")
    total_steps_per_epoch = -(-len(train_df) // batch_size)
    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------------------------------------------------
    # Phase 1: LoRA
    # -----------------------------------------------------------------
    lora_model = get_lora_model(
        model_name=DEFAULT_CODE_MODEL,
        rank=rank,
        lora_alpha=heft_alpha,
    ).to(device)

    if verbose:
        stats = count_trainable_parameters(lora_model)
        print(
            f"{log_prefix}[lora] rank={rank} | train_rows={len(train_df)} | val_rows={len(val_df)} | "
            f"batch_size={batch_size} | grad_accum={grad_accum_steps} | steps/epoch={total_steps_per_epoch} | "
            f"trainable={stats['trainable_parameters']:,} ({stats['trainable_percent']:.3f}%) | "
            f"total={stats['total_parameters']:,}"
        )
        if is_cuda:
            print(f"{log_prefix}[lora] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    lora_trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    lora_optimizer = AdamW(lora_trainable_params, lr=lora_lr)

    lora_es_config = EarlyStoppingConfig(
        max_epochs=lora_epochs, patience=lora_patience, min_delta=min_delta,
        min_epochs=lora_min_epochs, enabled=early_stopping,
    )
    _, lora_best_epoch, lora_history = run_training_with_early_stopping(
        lora_model, forward_lora, train_loader, val_loader, lora_optimizer, criterion, device,
        es_config=lora_es_config, is_cuda=is_cuda, grad_accum_steps=grad_accum_steps,
        verbose=verbose, log_every_steps=log_every_steps, log_prefix=log_prefix,
        phase_name="lora", total_steps_per_epoch=total_steps_per_epoch,
    )

    del lora_optimizer
    if is_cuda:
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Phase 2: freeze Phase 1, attach + train LoReFT
    # -----------------------------------------------------------------
    heft_model = attach_reft_to_lora_model(
        lora_model,
        reft_rank=reft_rank,
        layer_target=layer_target,
        freeze_previous_phase=True,
    ).to(device)

    if verbose:
        stats = count_trainable_parameters(heft_model)
        print(
            f"{log_prefix}[reft] reft_rank={reft_rank} layer={layer_target} | "
            f"trainable={stats['trainable_parameters']:,} ({stats['trainable_percent']:.3f}%) | "
            f"total={stats['total_parameters']:,} (should be tiny: only the LoReFT intervention)"
        )

    reft_trainable_params = [p for p in heft_model.parameters() if p.requires_grad]
    reft_optimizer = AdamW(reft_trainable_params, lr=reft_lr)

    reft_es_config = EarlyStoppingConfig(
        max_epochs=reft_epochs, patience=reft_patience, min_delta=min_delta,
        min_epochs=reft_min_epochs, enabled=early_stopping,
    )
    all_scores, reft_best_epoch, reft_history = run_training_with_early_stopping(
        heft_model, forward_reft, train_loader, val_loader, reft_optimizer, criterion, device,
        es_config=reft_es_config, is_cuda=is_cuda, grad_accum_steps=grad_accum_steps,
        verbose=verbose, log_every_steps=log_every_steps, log_prefix=log_prefix,
        phase_name="reft", total_steps_per_epoch=total_steps_per_epoch,
    )

    if verbose:
        print(
            f"{log_prefix}[heft] training done | lora best_epoch={lora_best_epoch}/{lora_epochs} "
            f"| reft best_epoch={reft_best_epoch}/{reft_epochs}"
        )

    del train_loader, val_loader, criterion, reft_optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), heft_model


def score_model(model, df, tokenizer, device, code_column="normalized_code",
                 max_length=512, batch_size=32, num_workers=2):
    """
    Public scoring helper for the final (Phase-2, pyreft-wrapped) HEFT model,
    no-gradient, deliberately separate from the early-stopping validation
    loop used during training (see leakage-guard rationale in
    run_exp5_canonical_retrain).
    """
    loader = create_dataloader(
        df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=False, num_workers=num_workers, code_column=code_column,
    )
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_reft(model, input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())
    return np.array(all_scores)


def train_heft_model_safe(*args, max_retries=2, **kwargs):
    """
    Note: on a CUDA OOM this retries the *entire* two-phase call (Phase 1 +
    Phase 2) from scratch with a smaller batch size -- same retry semantics
    as before, just be aware Phase 1 gets redone too if the OOM happens in
    Phase 2.
    """
    batch_size = kwargs.pop("batch_size", 16)
    grad_accum_steps = kwargs.pop("grad_accum_steps", 2)

    attempt = 0
    while True:
        try:
            return train_heft_model(
                *args, batch_size=batch_size, grad_accum_steps=grad_accum_steps, **kwargs
            )
        except torch.cuda.OutOfMemoryError:
            attempt += 1
            gc.collect()
            torch.cuda.empty_cache()
            if attempt > max_retries or batch_size <= 2:
                raise
            new_batch_size = max(2, batch_size // 2)
            new_grad_accum_steps = grad_accum_steps * max(1, batch_size // new_batch_size)
            print(
                f"[heft] CUDA OOM at batch_size={batch_size}; retrying "
                f"(attempt {attempt}/{max_retries}) with batch_size={new_batch_size}, "
                f"grad_accum_steps={new_grad_accum_steps} (effective batch size unchanged)."
            )
            batch_size, grad_accum_steps = new_batch_size, new_grad_accum_steps
