import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import NeoBertSequenceClassifier
from case_study_1 import split_manifest 

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
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
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def predict_oof(model, dataloader, device):
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                scores = torch.sigmoid(logits)
                
            all_scores.extend(scores.cpu().numpy())
            
    return np.array(all_scores)

def run_exp3_pipeline(abstracted_parquet_path, inner_manifest_path, outer_manifest_path, output_dir_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP3] Starting Linear Probe pipeline on Device: {device}")
    
    OUTPUT_DIR = Path(output_dir_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    full_df = pd.read_parquet(abstracted_parquet_path)
    outer_manifest_df = pd.read_parquet(outer_manifest_path)
    
    inner_manifest_df = split_manifest.load_manifest(
        inner_manifest_path,
        config=split_manifest.SplitConfig(n_splits=5, random_state=42, shuffle=True),
    )
    
    dev_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "development", "source_row_id"].tolist())
    development_frame = full_df[full_df["source_row_id"].isin(dev_ids)].copy().reset_index(drop=True)
    
    model_name = "Arize-ai/NeoBERT-250M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    oof_predictions_list = []
    scaler = torch.amp.GradScaler()
    
    for fold_id in range(5):
        print(f"\n--- [EXP3] Starting Outer Fold {fold_id + 1}/5 ---")
        
        train_manifest = inner_manifest_df[inner_manifest_df["fold"] != fold_id]
        val_manifest = inner_manifest_df[inner_manifest_df["fold"] == fold_id]
        
        train_df = development_frame[development_frame["source_row_id"].isin(train_manifest["source_row_id"])]
        val_df = development_frame[development_frame["source_row_id"].isin(val_manifest["source_row_id"])]
        
        train_loader = create_dataloader(train_df, tokenizer, batch_size=32, shuffle=True)
        val_loader = create_dataloader(val_df, tokenizer, batch_size=64, shuffle=False)
        
        model = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=True).to(device)
        
        pos_weight = get_class_weights(train_df).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.classification_head.parameters(), lr=1e-3, weight_decay=0.01)
        
        for epoch in range(3):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            print(f"  Epoch {epoch+1}/3 - Average Loss: {loss:.4f}")
            
        print(f"  Generating Out-of-Fold predictions for fold {fold_id}...")
        val_scores = predict_oof(model, val_loader, device)
        
        fold_oof_df = pd.DataFrame({
            "source_row_id": val_df["source_row_id"].values,
            "project": val_df["project"].values,
            "label": val_df["label"].values.astype(int),
            "y_score": val_scores,
            "fold": fold_id
        })
        oof_predictions_list.append(fold_oof_df)
        
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        
    oof_df = pd.concat(oof_predictions_list).reset_index(drop=True)
    oof_df.to_csv(OUTPUT_DIR / "exp3_oof_predictions.csv", index=False)
    print(f"\n[EXP3] Saved comprehensive OOF predictions to: {OUTPUT_DIR / 'exp3_oof_predictions.csv'}")
    
    print("\n--- [EXP3] Starting Canonical Retraining on complete Development Set (80%) ---")
    full_train_loader = create_dataloader(development_frame, tokenizer, batch_size=32, shuffle=True)
    
    final_model = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=True).to(device)
    final_pos_weight = get_class_weights(development_frame).to(device)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
    final_optimizer = AdamW(final_model.classification_head.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(3):
        loss = train_one_epoch(final_model, full_train_loader, final_optimizer, final_criterion, device, scaler)
        print(f"  [Retrain] Epoch {epoch+1}/3 - Average Loss: {loss:.4f}")
        
    torch.save(final_model.classification_head.state_dict(), OUTPUT_DIR / "final_exp3_linear_probe_head.pt")
    print(f"  Final model head saved to: {OUTPUT_DIR / 'final_exp3_linear_probe_head.pt'}")
    
    print("\n--- [EXP3] Starting Final Scoring on Outer Holdout partition (20%) ---")
    holdout_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "outer_holdout", "source_row_id"].tolist())
    holdout_frame = full_df[full_df["source_row_id"].isin(holdout_ids)].copy().reset_index(drop=True)
    
    holdout_loader = create_dataloader(holdout_frame, tokenizer, batch_size=64, shuffle=False)
    holdout_scores = predict_oof(final_model, holdout_loader, device)
    
    holdout_predictions_df = pd.DataFrame({
        "source_row_id": holdout_frame["source_row_id"].values,
        "project": holdout_frame["project"].values,
        "label": holdout_frame["label"].values.astype(int),
        "y_score": holdout_scores,
        "fold": 0
    })
    holdout_predictions_df.to_csv(OUTPUT_DIR / "exp3_holdout_predictions.csv", index=False)
    print(f"[EXP3] Pipeline successfully completed. Holdout predictions saved.")
    
    del final_model, full_train_loader, holdout_loader
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_exp3_pipeline(
        abstracted_parquet_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/processed/rdiversevul_cs1_normalized_plus_abstracted_v1.parquet",
        inner_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/inner_cv/cs1_project_grouped_5fold_manifest.parquet",
        outer_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/outer_holdout/cs1_outer_project_holdout_manifest.parquet",
        output_dir_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/outputs/exp3_linear_probe_output"
    )