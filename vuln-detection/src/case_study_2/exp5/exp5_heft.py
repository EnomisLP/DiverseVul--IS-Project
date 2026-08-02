import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
import numpy as np
from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import get_exp5_heft_model

def train_heft_model(train_df, val_df, tokenizer, lora_model_path, reft_rank, epochs=3, batch_size=32, device="cuda"):
    train_loader = create_dataloader(train_df, tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, batch_size=64, shuffle=False)
    
    model = get_exp5_heft_model(
        lora_model_path=lora_model_path, 
        model_name="Arize-ai/NeoBERT-250M", 
        reft_rank=reft_rank
    ).to(device)
    
    pos_weight = get_class_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.get_intervention_parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            unit_locations = model.get_unit_locations(input_ids=input_ids, attention_mask=attention_mask)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, logits = model(
                    base_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                    unit_locations=unit_locations
                )
                loss = criterion(logits.squeeze(-1), labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            unit_locations = model.get_unit_locations(input_ids=input_ids, attention_mask=attention_mask)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, logits = model(
                    base_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                    unit_locations=unit_locations
                )
                scores = torch.sigmoid(logits.squeeze(-1))
            all_scores.extend(scores.cpu().numpy())
            
    del train_loader, val_loader, criterion, optimizer
    torch.cuda.empty_cache()
    
    return np.array(all_scores), model