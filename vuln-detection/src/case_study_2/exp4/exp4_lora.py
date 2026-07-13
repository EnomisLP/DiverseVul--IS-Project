import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
import numpy as np
from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_exp4_lora_model

def train_lora_model(train_df, val_df, tokenizer, rank, epochs=3, batch_size=32, device="cuda"):
    train_loader = create_dataloader(train_df, tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, batch_size=64, shuffle=False)
    
    model = get_exp4_lora_model(model_name="Arize-ai/NeoBERT-250M", rank=rank, lora_alpha=16).to(device)
    
    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                scores = torch.sigmoid(logits)
            all_scores.extend(scores.cpu().numpy())
            
    del train_loader, val_loader, criterion, optimizer
    torch.cuda.empty_cache()
    
    return np.array(all_scores), model