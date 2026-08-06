from __future__ import annotations

import gc
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import pyreft
from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_heft_model, count_trainable_parameters, DEFAULT_CODE_MODEL


def forward_heft(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Safely executes the forward pass for either standard or PyReft/PyVene wrapped models."""
    if hasattr(model, "interventions"):
        batch_size, seq_len = input_ids.shape
        
        # Helper generates valid pyvene locations for all tokens in batch
        unit_locations = pyreft.get_intervention_locations(
            last_position=seq_len,
            first_position=0,
            num_targets=1,
            num_transformations=len(model.interventions),
            batch_size=batch_size,
        )
        
        # Move generated tensor locations to current device
        if isinstance(unit_locations, torch.Tensor):
            unit_locations = unit_locations.to(input_ids.device)
            
        outputs = model(
            base={"input_ids": input_ids, "attention_mask": attention_mask},
            unit_locations={"sources->base": unit_locations},
        )
        
        if isinstance(outputs, (tuple, list)):
            logits = outputs[1]
        else:
            logits = getattr(outputs, "logits", outputs)
    else:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    if logits.ndim > 1 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)

    return logits

def train_heft_model(
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
    reft_rank=4,
    layer_target=4,
    heft_alpha=16,
):
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=True, num_workers=num_workers, code_column=code_column,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=eval_batch_size, max_length=max_length,
        shuffle=False, num_workers=num_workers, code_column=code_column,
    )

    model = get_heft_model(
        model_name=DEFAULT_CODE_MODEL,
        rank=rank,
        reft_rank=reft_rank,
        layer_target=layer_target,
        heft_alpha=heft_alpha,
    ).to(device)

    is_cuda = (device == "cuda") or (hasattr(device, "type") and device.type == "cuda")

    total_steps_per_epoch = -(-len(train_df) // batch_size)

    if verbose:
        stats = count_trainable_parameters(model)
        print(
            f"{log_prefix}[heft] rank={rank} (reft_r={reft_rank}, layer={layer_target}) | "
            f"train_rows={len(train_df)} | val_rows={len(val_df)} | "
            f"batch_size={batch_size} | grad_accum={grad_accum_steps} | steps/epoch={total_steps_per_epoch} | "
            f"trainable={stats['trainable_parameters']:,} ({stats['trainable_percent']:.3f}%) | "
            f"total={stats['total_parameters']:,}"
        )
        if is_cuda:
            print(f"{log_prefix}[heft] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Filter trainable parameters for AdamW optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=2e-4)

    t0 = time.time()

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
                logits = forward_heft(model, input_ids, attention_mask)
                loss = criterion(logits, labels.float()) / grad_accum_steps

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
                    f"{log_prefix}[heft] epoch {epoch+1}/{epochs} step {step+1}/{total_steps_per_epoch} | "
                    f"avg_loss_so_far={epoch_loss/max(n_steps,1):.4f} | "
                    f"elapsed={elapsed_min:.1f} min | ETA epoch ~{eta_min:.1f} min"
                )

        if n_steps > 0 and n_steps % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if verbose:
            elapsed_min = (time.time() - t0) / 60
            epoch_min = (time.time() - epoch_t0) / 60
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
            print(
                f"{log_prefix}[heft] epoch {epoch+1}/{epochs} done | avg_loss={epoch_loss/max(n_steps,1):.4f} "
                f"| epoch_time={epoch_min:.1f} min | total_elapsed={elapsed_min:.1f} min | peak_VRAM={peak_vram:.2f} GB"
            )

    if verbose:
        print(f"{log_prefix}[heft] training done, scoring validation set ({len(val_df)} rows)...")

    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_heft(model, input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())

    del train_loader, val_loader, criterion, optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), model


def train_heft_model_safe(*args, max_retries=2, **kwargs):
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