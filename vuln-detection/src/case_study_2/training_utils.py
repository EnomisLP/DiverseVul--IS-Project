from __future__ import annotations

"""
Shared PyTorch training loop with validation-PR-AUC early stopping.

CONTEXT (see handoff notes / PDD addendum): EXP-4 (LoRA/CodeBERTa), EXP-5
(HEFT LoRA+ReFT/CodeBERTa) and EXP-7 (LoRA/NeoBERT) each had their own
hand-written `for epoch in range(fixed_epochs)` training loop with no
validation checkpointing -- the model was simply trained for a fixed,
hand-picked number of epochs and the LAST epoch's weights were used,
regardless of whether training loss (let alone validation performance) had
actually plateaued.

This is inconsistent with the rest of the project: Track A's MLP
(case_study_1/exp2/exp2_mlp.py::_train_mlp_with_early_stopping) already does
proper early stopping with a patience/min_delta rule and best-checkpoint
restoration, and Track B's own linear probe (EXP-3) is a sklearn
LogisticRegression(solver="lbfgs") that converges on a real gradient
tolerance. EXP-4/5/7 were the odd ones out.

Diagnostic evidence that this was a real (not hypothetical) problem:
  - EXP-7 (NeoBERT LoRA) canonical retrain: loss 1.1424 -> 1.0220 -> 0.9392
    -> 0.8328 over 4 epochs -- still falling steadily, no sign of plateau.
  - EXP-4 (CodeBERTa LoRA) canonical retrain: loss 1.1369 -> 1.0712 -> 1.0228
    -> 0.9808 over 4 epochs -- decelerating but still declining.
  - EXP-5 (CodeBERTa HEFT) Phase-1 LoRA sub-stage only ran 2 epochs (fewer
    than EXP-4's own 4), while Phase-2 ReFT was already flat by epoch 2
    (1.0449 -> 1.0435) -- ReFT converges fast, but its LoRA foundation was
    undertrained even relative to EXP-4's own (also undertrained) LoRA.

This module factors out ONE early-stopping loop (mirroring exp2_mlp.py's
convention: track val PR-AUC as the primary selection metric, val loss as a
tie-break, `min_delta` to avoid stopping on noise, `patience` epochs of no
improvement before stopping, always restore the best-epoch weights) so
EXP-4/5/7 all use the same, tested convergence criterion instead of three
slightly different copy-pasted loops.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Mirrors case_study_1/exp2/exp2_mlp.py's Exp2Config early-stopping fields."""
    max_epochs: int = 10
    patience: int = 2
    min_delta: float = 1e-4
    min_epochs: int = 2
    enabled: bool = True


def run_training_with_early_stopping(
    model: torch.nn.Module,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    es_config: EarlyStoppingConfig,
    is_cuda: bool = True,
    grad_accum_steps: int = 1,
    verbose: bool = True,
    log_every_steps: Optional[int] = 50,
    log_prefix: str = "",
    phase_name: str = "train",
    total_steps_per_epoch: Optional[int] = None,
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    """
    Generic epoch loop: trains `model` via `forward_fn(model, input_ids,
    attention_mask) -> logits`, evaluates PR-AUC + loss on val_loader after
    every epoch, and (if es_config.enabled) stops once `patience` epochs pass
    with no PR-AUC improvement beyond `min_delta`, restoring the best-epoch
    weights before returning.

    Returns (best_val_scores, best_epoch, history) where `history` is a list
    of per-epoch dicts (epoch, train_loss, val_loss, val_pr_auc,
    is_best_epoch) -- same shape as exp2_mlp.py's history_rows, useful for
    plotting/auditing convergence after the fact.

    If es_config.enabled=False, this simply runs all `max_epochs` epochs and
    returns the LAST epoch (old behaviour), for cases where a caller
    explicitly wants a fixed-epoch run (e.g. reproducing a prior result).
    """
    t0 = time.time()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_pr_auc = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_scores: Optional[np.ndarray] = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, es_config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        epoch_t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_fn(model, input_ids, attention_mask)
                loss = criterion(logits, labels) / grad_accum_steps

            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if verbose and log_every_steps and total_steps_per_epoch and (step + 1) % log_every_steps == 0:
                elapsed_min = (time.time() - epoch_t0) / 60
                steps_left = total_steps_per_epoch - (step + 1)
                rate = (step + 1) / max(elapsed_min, 1e-6)
                eta_min = steps_left / max(rate, 1e-6)
                print(
                    f"{log_prefix}[{phase_name}] epoch {epoch}/{es_config.max_epochs} "
                    f"step {step+1}/{total_steps_per_epoch} | "
                    f"avg_loss_so_far={epoch_loss/max(n_steps,1):.4f} | "
                    f"elapsed={elapsed_min:.1f} min | ETA epoch ~{eta_min:.1f} min"
                )

        if n_steps > 0 and n_steps % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # --- validation pass -------------------------------------------------
        model.eval()
        val_scores_parts, val_labels_parts = [], []
        val_loss_sum, val_rows = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True).float()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = forward_fn(model, input_ids, attention_mask)
                    loss = criterion(logits, labels)
                scores = torch.sigmoid(logits)
                val_scores_parts.append(scores.float().cpu().numpy())
                val_labels_parts.append(labels.float().cpu().numpy())
                val_loss_sum += float(loss.item()) * int(labels.shape[0])
                val_rows += int(labels.shape[0])

        val_scores = np.concatenate(val_scores_parts) if val_scores_parts else np.array([])
        val_labels = np.concatenate(val_labels_parts) if val_labels_parts else np.array([])
        val_pr_auc = float(average_precision_score(val_labels, val_scores)) if val_rows else float("nan")
        val_loss = float(val_loss_sum / max(val_rows, 1))

        # Same tie-break rule as exp2_mlp.py: PR-AUC is primary, val loss
        # breaks near-ties (within min_delta) so we don't chase noise.
        improved = (
            val_pr_auc > best_val_pr_auc + es_config.min_delta
            or (
                abs(val_pr_auc - best_val_pr_auc) <= es_config.min_delta
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_pr_auc = val_pr_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores = val_scores
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": int(epoch),
            "train_loss": epoch_loss / max(n_steps, 1),
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "is_best_epoch": bool(improved),
        })

        if verbose:
            elapsed_min = (time.time() - t0) / 60
            epoch_min = (time.time() - epoch_t0) / 60
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
            flag = " <- best so far" if improved else ""
            print(
                f"{log_prefix}[{phase_name}] epoch {epoch}/{es_config.max_epochs} done | "
                f"train_loss={epoch_loss/max(n_steps,1):.4f} | val_loss={val_loss:.4f} | "
                f"val_pr_auc={val_pr_auc:.4f}{flag} | epoch_time={epoch_min:.1f} min | "
                f"total_elapsed={elapsed_min:.1f} min | peak_VRAM={peak_vram:.2f} GB"
            )

        if (
            es_config.enabled
            and epoch >= es_config.min_epochs
            and no_improve >= es_config.patience
        ):
            if verbose:
                print(
                    f"{log_prefix}[{phase_name}] early stopping at epoch {epoch}/{es_config.max_epochs} "
                    f"(no val PR-AUC improvement > {es_config.min_delta} for {es_config.patience} epochs); "
                    f"restoring epoch {best_epoch} (val_pr_auc={best_val_pr_auc:.4f})."
                )
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if verbose and best_epoch != epoch:
        print(
            f"{log_prefix}[{phase_name}] reached max_epochs={es_config.max_epochs}; "
            f"restoring best epoch {best_epoch} (val_pr_auc={best_val_pr_auc:.4f})."
        )

    if best_scores is None:
        # Degenerate case (max_epochs somehow 0) -- shouldn't happen in
        # practice, but avoid returning None rather than silently failing.
        best_scores = np.array([])

    return best_scores, best_epoch, history
