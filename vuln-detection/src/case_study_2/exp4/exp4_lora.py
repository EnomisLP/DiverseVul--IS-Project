import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_exp4_lora_model, count_trainable_parameters, DEFAULT_CODE_MODEL


def train_lora_model(
    train_df, val_df, tokenizer, rank, epochs=3, batch_size=32,
    grad_accum_steps=1, device="cuda", hf_cache_dir=None, verbose=True,
    code_column="normalized_code", max_length=512,
):
    """
    Trains a LoRA-adapted CodeBERTa-small-v1 sequence classifier.

    CodeBERTa-small-v1 is a 6-layer, 84M-parameter RoBERTa-family encoder --
    roughly a third the size of NeoBERT-250M -- so it tolerates a larger
    default batch size (32, grad_accum=1) on a single T4/L4 GPU. Reduce
    batch_size / raise grad_accum_steps if you see OOM errors.
    """
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=True, num_workers=2, code_column=code_column,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=64, max_length=max_length,
        shuffle=False, num_workers=2, code_column=code_column,
    )

    model = get_exp4_lora_model(
        model_name=DEFAULT_CODE_MODEL, rank=rank, lora_alpha=16, hf_cache_dir=hf_cache_dir,
    ).to(device)

    is_cuda = (device == "cuda") or (hasattr(device, "type") and device.type == "cuda")

    if verbose:
        stats = count_trainable_parameters(model)
        print(
            f"    [lora] rank={rank} | trainable={stats['trainable_parameters']:,} "
            f"({stats['trainable_percent']:.3f}%) | total={stats['total_parameters']:,}"
        )
        if is_cuda:
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

            autocast_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / grad_accum_steps

            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            elapsed_min = (time.time() - t0) / 60
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
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
            autocast_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.float().cpu().numpy())

    del train_loader, val_loader, criterion, optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return np.array(all_scores), model