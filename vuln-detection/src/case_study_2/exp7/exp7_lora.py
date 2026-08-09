from __future__ import annotations

import copy
import gc
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import average_precision_score
import numpy as np

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_lora_model, count_trainable_parameters, DEFAULT_NEOBERT_MODEL


def score_model(model, df, tokenizer, device, code_column="normalized_code",
                 max_length=512, batch_size=32, num_workers=2):
    """
    Public scoring helper: run a trained model over an arbitrary dataframe
    and return sigmoid scores as a numpy array, in the dataframe's row order.

    Used by run_exp7_canonical_retrain for the FINAL, no-gradient pass over
    the frozen outer holdout, kept separate from the early-stopping
    validation loop (which must never see the holdout -- see handoff notes).
    """
    loader = create_dataloader(
        df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=False, num_workers=num_workers, code_column=code_column,
    )
    _, scores = _score_val_loader(model, loader, device)
    return scores


def _score_val_loader(model, val_loader, device):
    """Run inference over val_loader and return (y_true, y_score) numpy arrays."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_scores)


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
    log_every_steps=50,
    log_prefix="",
    # --- FIX (undertraining diagnosis, see handoff notes) -----------------
    # EXP-7 (NeoBERT, 28 layers) was inheriting the same fixed epoch budget
    # tuned for EXP-4 (CodeBERTa, 6 layers). The training loss was still
    # falling steadily at the last epoch (1.14 -> 1.02 -> 0.94 -> 0.83 over
    # 4 epochs), i.e. training was stopped before convergence, not because
    # it had plateaued. Rather than guess a new fixed epoch count, this adds
    # validation-PR-AUC-based early stopping: keep training up to
    # `max_epochs`, but stop once val PR-AUC hasn't improved for `patience`
    # consecutive epochs, and return the BEST checkpoint seen (not the last).
    # `epochs` is kept as the effective ceiling for backward compatibility
    # with existing call sites (nested search / canonical retrain) that
    # already pass `epochs=config.search_epochs` / `epochs=config.epochs`.
    early_stopping=True,
    patience=2,
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

    # EXP-7: identical LoRA recipe to EXP-4, only the backbone changes
    # (CodeBERTa -> NeoBERT-250M). get_lora_model() auto-detects the NeoBERT
    # name and applies trust_remote_code=True plus the unpadding/fp32-head
    # safeguards defined in case_study_2/models.py.
    model = get_lora_model(model_name=DEFAULT_NEOBERT_MODEL, rank=rank, lora_alpha=16).to(device)

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

    t0 = time.time()

    best_val_prauc = -1.0
    best_epoch = -1
    best_state_dict = None
    best_scores = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        epoch_t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / grad_accum_steps

            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if verbose and log_every_steps and (step + 1) % log_every_steps == 0:
                elapsed_min = (time.time() - epoch_t0) / 60
                steps_left = total_steps_per_epoch - (step + 1)
                rate = (step + 1) / max(elapsed_min, 1e-6)
                eta_min = steps_left / max(rate, 1e-6)
                print(
                    f"{log_prefix}[lora] epoch {epoch+1}/{epochs} step {step+1}/{total_steps_per_epoch} | "
                    f"avg_loss_so_far={epoch_loss/max(n_steps,1):.4f} | "
                    f"elapsed={elapsed_min:.1f} min | ETA epoch ~{eta_min:.1f} min"
                )

        optimizer.step()
        optimizer.zero_grad()

        # --- FIX: evaluate on val_df every epoch, track the best checkpoint ---
        val_labels, val_scores_epoch = _score_val_loader(model, val_loader, device)
        val_prauc = float(average_precision_score(val_labels, val_scores_epoch))
        improved = val_prauc > best_val_prauc

        if verbose:
            elapsed_min = (time.time() - t0) / 60
            epoch_min = (time.time() - epoch_t0) / 60
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
            flag = " <- best so far" if improved else ""
            print(
                f"{log_prefix}[lora] epoch {epoch+1}/{epochs} done | avg_loss={epoch_loss/max(n_steps,1):.4f} "
                f"| val_pr_auc={val_prauc:.4f}{flag} "
                f"| epoch_time={epoch_min:.1f} min | total_elapsed={elapsed_min:.1f} min | peak_VRAM={peak_vram:.2f} GB"
            )

        if improved:
            best_val_prauc = val_prauc
            best_epoch = epoch + 1
            best_scores = val_scores_epoch
            epochs_without_improvement = 0
            if early_stopping:
                # Keep weights on CPU to avoid holding two full copies on GPU.
                best_state_dict = copy.deepcopy(model.state_dict())
                best_state_dict = {k: v.cpu() for k, v in best_state_dict.items()}
        else:
            epochs_without_improvement += 1

        if early_stopping and (epoch + 1) >= min_epochs and epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"{log_prefix}[lora] early stopping at epoch {epoch+1}/{epochs} "
                    f"(no val PR-AUC improvement for {patience} epochs; best was epoch {best_epoch}, "
                    f"val_pr_auc={best_val_prauc:.4f})"
                )
            break

    if verbose:
        print(f"{log_prefix}[lora] training done ({epoch+1} epoch(s) run).")

    if early_stopping and best_state_dict is not None:
        # Restore the best-epoch weights before returning, so the caller
        # (nested search / canonical retrain / holdout eval) always gets the
        # checkpoint with the highest validation PR-AUC, not just the last one.
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        all_scores = best_scores
        if verbose:
            print(f"{log_prefix}[lora] restored best checkpoint from epoch {best_epoch} (val_pr_auc={best_val_prauc:.4f}).")
    else:
        # early_stopping=False, or somehow no epoch ever improved (shouldn't
        # happen since epoch 1 always sets best_*): fall back to last epoch's
        # scores, matching the old (pre-fix) behaviour.
        all_scores = best_scores if best_scores is not None else _score_val_loader(model, val_loader, device)[1]

    del train_loader, val_loader, criterion, optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), model


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