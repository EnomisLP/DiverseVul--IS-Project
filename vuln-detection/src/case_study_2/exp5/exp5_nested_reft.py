import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from case_study_2.data_loader import create_dataloader
from case_study_2.exp5.exp5_heft import train_heft_model
from case_study_1 import split_manifest

def run_exp5_nested_pipeline(abstracted_parquet_path, inner_manifest_path, outer_manifest_path, lora_model_path, output_dir_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP5] Starting Nested HEFT (LoRA+ReFT) Tuning Pipeline on Device: {device}")
    
    OUTPUT_DIR = Path(output_dir_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    full_df = pd.read_parquet(abstracted_parquet_path)
    outer_manifest_df = pd.read_parquet(outer_manifest_path)
    
    outer_cv_manifest = split_manifest.load_manifest(
        inner_manifest_path,
        config=split_manifest.SplitConfig(n_splits=5, random_state=42, shuffle=True),
    )
    
    dev_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "development", "source_row_id"].tolist())
    development_frame = full_df[full_df["source_row_id"].isin(dev_ids)].copy().reset_index(drop=True)
    
    model_name = "Arize-ai/NeoBERT-250M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    REFT_RANK_GRID = [4, 8]
    oof_predictions_list = []
    selected_ranks_audit = []
    
    for outer_fold in range(5):
        print(f"\n=================== OUTER FOLD {outer_fold + 1}/5 ===================")
        
        outer_train_manifest = outer_cv_manifest[outer_cv_manifest["fold"] != outer_fold]
        outer_val_manifest = outer_cv_manifest[outer_cv_manifest["fold"] == outer_fold]
        
        outer_train_df = development_frame[development_frame["source_row_id"].isin(outer_train_manifest["source_row_id"])]
        outer_val_df = development_frame[development_frame["source_row_id"].isin(outer_val_manifest["source_row_id"])]
        
        print(f"  [Inner Loop] Starting inner ReFT tuning for Outer Fold {outer_fold}...")
        inner_cv_manifest = split_manifest.load_manifest(
            outer_train_df[["source_row_id", "label", "project"]],
            config=split_manifest.SplitConfig(n_splits=3, random_state=20260707, shuffle=True),
        )
        
        rank_performance = {}
        
        for rank_candidate in REFT_RANK_GRID:
            print(f"    Testing candidate ReFT Rank = {rank_candidate}")
            inner_fold_praucs = []
            
            for inner_fold in range(3):
                inner_train_ids = inner_cv_manifest[inner_cv_manifest["fold"] != inner_fold]["source_row_id"]
                inner_val_ids = inner_cv_manifest[inner_cv_manifest["fold"] == inner_fold]["source_row_id"]
                
                inner_train_df = outer_train_df[outer_train_df["source_row_id"].isin(inner_train_ids)]
                inner_val_df = outer_train_df[outer_train_df["source_row_id"].isin(inner_val_ids)]
                
                val_scores, tmp_model = train_heft_model(
                    inner_train_df, inner_val_df, tokenizer, lora_model_path=lora_model_path,
                    reft_rank=rank_candidate, epochs=3, device=device
                )
                
                prauc = average_precision_score(inner_val_df["label"].values, val_scores)
                inner_fold_praucs.append(prauc)
                
                del tmp_model
                torch.cuda.empty_cache()
                
            mean_inner_prauc = np.mean(inner_fold_praucs)
            print(f"    -> ReFT Rank {rank_candidate} | Inner Mean PR-AUC: {mean_inner_prauc:.4f}")
            rank_performance[rank_candidate] = mean_inner_prauc
            
        optimal_rank = max(rank_performance, key=rank_performance.get)
        print(f"  [Inner Loop] Selected Optimal ReFT Rank = {optimal_rank} for Outer Fold {outer_fold}")
        selected_ranks_audit.append({"outer_fold": outer_fold, "selected_rank": optimal_rank})
        
        print(f"  [Outer Refit] Executing final refit on Outer Train with ReFT Rank = {optimal_rank}...")
        outer_val_scores, final_outer_model = train_heft_model(
            outer_train_df, outer_val_df, tokenizer, lora_model_path=lora_model_path,
            reft_rank=optimal_rank, epochs=3, device=device
        )
        
        fold_oof_df = pd.DataFrame({
            "source_row_id": outer_val_df["source_row_id"].values,
            "project": outer_val_df["project"].values,
            "label": outer_val_df["label"].values.astype(int),
            "y_score": outer_val_scores,
            "fold": outer_fold
        })
        oof_predictions_list.append(fold_oof_df)
        
        del final_outer_model
        torch.cuda.empty_cache()
        
    oof_df = pd.concat(oof_predictions_list).reset_index(drop=True)
    oof_df.to_csv(OUTPUT_DIR / "exp5_oof_predictions.csv", index=False)
    
    audit_ranks_df = pd.DataFrame(selected_ranks_audit)
    audit_ranks_df.to_csv(OUTPUT_DIR / "exp5_selected_reft_rank_per_outer_fold.csv", index=False)
    
    best_global_rank = int(audit_ranks_df["selected_rank"].mode()[0])
    print(f"\n--- [EXP5] Starting Global Canonical Retraining with ReFT Rank Winner = {best_global_rank} ---")
    
    holdout_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "outer_holdout", "source_row_id"].tolist())
    holdout_frame = full_df[full_df["source_row_id"].isin(holdout_ids)].copy().reset_index(drop=True)
    
    holdout_scores, global_model = train_heft_model(
        development_frame, holdout_frame, tokenizer, lora_model_path=lora_model_path,
        reft_rank=best_global_rank, epochs=3, device=device
    )
    
    global_model.save_pretrained(OUTPUT_DIR / "final_canonical_heft_model")
    print(f"  Final HEFT model saved to: {OUTPUT_DIR / 'final_canonical_heft_model'}")
    
    holdout_predictions_df = pd.DataFrame({
        "source_row_id": holdout_frame["source_row_id"].values,
        "project": holdout_frame["project"].values,
        "label": holdout_frame["label"].values.astype(int),
        "y_score": holdout_scores,
        "fold": 0
    })
    holdout_predictions_df.to_csv(OUTPUT_DIR / "exp5_holdout_predictions.csv", index=False)
    print(f"[EXP5] HEFT (LoRA+ReFT) pipeline finished successfully.")
    
    del global_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_exp5_nested_pipeline(
        abstracted_parquet_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/processed/rdiversevul_cs1_normalized_plus_abstracted_v1.parquet",
        inner_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/inner_cv/cs1_project_grouped_5fold_manifest.parquet",
        outer_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/outer_holdout/cs1_outer_project_holdout_manifest.parquet",
        lora_model_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/outputs/exp4_lora_output/final_canonical_lora_model",
        output_dir_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/outputs/exp5_heft_output"
    )