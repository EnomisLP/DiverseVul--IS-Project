from __future__ import annotations

import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_lora_model, count_trainable_parameters, DEFAULT_NEOBERT_MODEL
from case_study_2.training_utils import EarlyStoppingConfig, run_training_with_early_stopping


def forward_lora(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    if logits.ndim > 1 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return logits


def train_lora_model(
    train_df,
    val_df,
    tokenizer,
    rank,
    epochs=3,
    batch_size=16,
    grad_accum_steps=2,
    eval_batch_size=32,
    num_workers=2,
    device="cuda",
    hf_cache_dir=None,
    code_column="normalized_code",
    max_length=512,
    verbose=True,
    # 500, not 50 -- inherited by the nested rank search / canonical retrain,
    # which run many epochs across many folds/candidates; 50 produced
    # hundreds of near-duplicate "step N/M" lines per run with no signal
    # beyond the per-epoch summary. Short/interactive runs (smoke test) can
    # still pass a smaller value explicitly.
    log_every_steps=500,
    log_prefix="",
    # --- FIX (see case_study_2/training_utils.py header for full context) --
    # EXP-7 (NeoBERT, 28 layers) was inheriting the same fixed epoch budget
    # tuned for EXP-4 (CodeBERTa, 6 layers). The canonical retrain's training
    # loss was still falling steadily at the last epoch (1.14 -> 1.02 -> 0.94
    # -> 0.83 over 4 epochs), i.e. training was stopped before convergence.
    # `epochs` is now a CEILING: training early-stops on validation PR-AUC
    # and the best checkpoint (not necessarily the last) is returned. Uses
    # the same shared early-stopping loop as EXP-4 and EXP-5's LoRA phase.
    early_stopping=True,
    patience=2,
    min_delta=1e-4,
    min_epochs=2,
):
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=True, num_workers=num_workers, code_column=code_column,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=eval_batch_size, max_length=max_length,
        shuffle=False, num_workers=num_workers, code_column=code_column,
    )

    model = get_lora_model(
        model_name=DEFAULT_NEOBERT_MODEL, rank=rank, lora_alpha=16,
        trust_remote_code=True,
    ).to(device)

    is_cuda = (device == "cuda") or (hasattr(device, "type") and device.type == "cuda")
    total_steps_per_epoch = -(-len(train_df) // batch_size)

    if verbose:
        stats = count_trainable_parameters(model)
        print(
            f"{log_prefix}[lora] rank={rank} | train_rows={len(train_df)} | val_rows={len(val_df)} | "
            f"batch_size={batch_size} | grad_accum={grad_accum_steps} | steps/epoch={total_steps_per_epoch} | "
            f"trainable={stats['trainable_parameters']:,} ({stats['trainable_percent']:.3f}%) | "
            f"total={stats['total_parameters']:,}"
        )
        if is_cuda:
            print(f"{log_prefix}[lora] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=2e-4)

    es_config = EarlyStoppingConfig(
        max_epochs=epochs, patience=patience, min_delta=min_delta,
        min_epochs=min_epochs, enabled=early_stopping,
    )
    all_scores, best_epoch, history = run_training_with_early_stopping(
        model, forward_lora, train_loader, val_loader, optimizer, criterion, device,
        es_config=es_config, is_cuda=is_cuda, grad_accum_steps=grad_accum_steps,
        verbose=verbose, log_every_steps=log_every_steps, log_prefix=log_prefix,
        phase_name="lora", total_steps_per_epoch=total_steps_per_epoch,
    )

    del train_loader, val_loader, criterion, optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), model, history


def score_model(model, df, tokenizer, device, code_column="normalized_code",
                 max_length=512, batch_size=32, num_workers=2):
    """
    Public scoring helper: run a trained model over an arbitrary dataframe,
    no gradient, no early-stopping bookkeeping. Used by
    run_exp7_canonical_retrain for the FINAL pass over the frozen outer
    holdout, kept deliberately separate from the early-stopping validation
    loop (which must never see the holdout -- see exp7_nested_rank.py
    run_exp7_canonical_retrain for the leakage-guard rationale).
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
                logits = forward_lora(model, input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())
    return np.array(all_scores)


def train_lora_model_safe(*args, max_retries=2, **kwargs):
    batch_size = kwargs.pop("batch_size", 16)
    grad_accum_steps = kwargs.pop("grad_accum_steps", 2)

    attempt = 0
    while True:
        try:
            return train_lora_model(
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
                f"[lora] CUDA OOM at batch_size={batch_size}; retrying "
                f"(attempt {attempt}/{max_retries}) with batch_size={new_batch_size}, "
                f"grad_accum_steps={new_grad_accum_steps} (effective batch size unchanged)."
            )
            batch_size, grad_accum_steps = new_batch_size, new_grad_accum_steps
