import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_exp4_lora_model, count_trainable_parameters, DEFAULT_NEOBERT_MODEL


def train_lora_model(
    train_df, val_df, tokenizer, rank, epochs=3, batch_size=16,
    grad_accum_steps=2, device="cuda", hf_cache_dir=None, verbose=True,
):
    """
    Trains a LoRA-adapted NeoBERT sequence classifier.
    batch_size=16 with grad_accum_steps=2 -> effective batch 32, matching the
    spec's intended batch size while halving peak VRAM per step.
    """
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=32, shuffle=False, num_workers=2,
    )

    model = get_exp4_lora_model(
        model_name=DEFAULT_NEOBERT_MODEL, rank=rank, lora_alpha=16,
    ).to(device)

    if verbose:
        stats = count_trainable_parameters(model)
        print(
            f"    [lora] rank={rank} | trainable={stats['trainable_parameters']:,} "
            f"({stats['trainable_percent']:.3f}%) | total={stats['total_parameters']:,}"
        )
        if device == "cuda":
            print(f"    [lora] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=2e-4)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
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

        # flush any remaining accumulated gradients
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            elapsed_min = (time.time() - t0) / 60
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            print(
                f"    [lora] epoch {epoch+1}/{epochs} | avg_loss={epoch_loss/max(n_steps,1):.4f} "
                f"| elapsed={elapsed_min:.1f} min | peak_VRAM={peak_vram:.2f} GB"
            )

    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())

    del train_loader, val_loader, criterion, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), model