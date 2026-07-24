import os
import sys
import gc
import time
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import configure_huggingface_cache, load_code_tokenizer, DEFAULT_CODE_TOKENIZER
from case_study_2.exp4.exp4_lora import train_lora_model
from case_study_1 import split_manifest
from sklearn.metrics import average_precision_score


def run_exp4_nested_pipeline(
    abstracted_parquet_path, inner_manifest_path, outer_manifest_path,
    output_dir_path, hf_cache_dir=None, rank_grid=(8, 16), epochs=3,
    code_column="normalized_code",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP4] Starting Nested LoRA Tuning Pipeline (CodeBERTa-small-v1) on Device: {device}")

    OUTPUT_DIR = Path(output_dir_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "exp4_nested_checkpoint.joblib"

    configure_huggingface_cache(hf_cache_dir)

    full_df = pd.read_parquet(abstracted_parquet_path)
    outer_manifest_df = pd.read_parquet(outer_manifest_path)

    outer_cv_manifest = split_manifest.load_manifest(
        inner_manifest_path,
        config=split_manifest.SplitConfig(n_splits=5, random_state=42, shuffle=True),
    )

    dev_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "development", "source_row_id"].tolist())
    development_frame = full_df[full_df["source_row_id"].isin(dev_ids)].copy().reset_index(drop=True)

    tokenizer = load_code_tokenizer(DEFAULT_CODE_TOKENIZER, hf_cache_dir=hf_cache_dir)

    oof_predictions_list = []
    selected_ranks_audit = []
    completed_outer_folds = set()

    if checkpoint_path.exists():
        ckpt = joblib.load(checkpoint_path)
        oof_predictions_list = ckpt["oof_predictions_list"]
        selected_ranks_audit = ckpt["selected_ranks_audit"]
        completed_outer_folds = ckpt["completed_outer_folds"]
        print(f"[EXP4] Resuming: {len(completed_outer_folds)}/5 outer folds already completed")

    for outer_fold in range(5):
        if outer_fold in completed_outer_folds:
            print(f"[EXP4] Outer fold {outer_fold}: already completed, skipping.")
            continue

        fold_t0 = time.time()
        print(f"\n=================== OUTER FOLD {outer_fold + 1}/5 ===================")

        outer_train_manifest = outer_cv_manifest[outer_cv_manifest["fold"] != outer_fold]
        outer_val_manifest = outer_cv_manifest[outer_cv_manifest["fold"] == outer_fold]

        outer_train_df = development_frame[development_frame["source_row_id"].isin(outer_train_manifest["source_row_id"])]
        outer_val_df = development_frame[development_frame["source_row_id"].isin(outer_val_manifest["source_row_id"])]

        print(f"  [Inner Loop] Starting inner tuning for Outer Fold {outer_fold}...")
        inner_cv_manifest = split_manifest.load_manifest(
            outer_train_df[["source_row_id", "label", "project"]],
            config=split_manifest.SplitConfig(n_splits=3, random_state=20260707, shuffle=True),
        )

        rank_performance = {}

        for rank_candidate in rank_grid:
            print(f"    Testing candidate Rank = {rank_candidate}")
            inner_fold_praucs = []

            for inner_fold in range(3):
                inner_t0 = time.time()
                inner_train_ids = inner_cv_manifest[inner_cv_manifest["fold"] != inner_fold]["source_row_id"]
                inner_val_ids = inner_cv_manifest[inner_cv_manifest["fold"] == inner_fold]["source_row_id"]

                inner_train_df = outer_train_df[outer_train_df["source_row_id"].isin(inner_train_ids)]
                inner_val_df = outer_train_df[outer_train_df["source_row_id"].isin(inner_val_ids)]

                val_scores, tmp_model = train_lora_model(
                    inner_train_df, inner_val_df, tokenizer, rank=rank_candidate,
                    epochs=epochs, device=device, hf_cache_dir=hf_cache_dir,
                    code_column=code_column,
                )

                prauc = average_precision_score(inner_val_df["label"].values, val_scores)
                inner_fold_praucs.append(prauc)

                print(
                    f"      inner_fold {inner_fold}: PR-AUC={prauc:.4f} "
                    f"| {(time.time()-inner_t0)/60:.1f} min"
                )

                del tmp_model, inner_train_df, inner_val_df
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            mean_inner_prauc = np.mean(inner_fold_praucs)
            print(f"    -> Rank {rank_candidate} | Inner Mean PR-AUC: {mean_inner_prauc:.4f}")
            rank_performance[rank_candidate] = mean_inner_prauc

        optimal_rank = max(rank_performance, key=rank_performance.get)
        print(f"  [Inner Loop] Selected Optimal Rank = {optimal_rank} for Outer Fold {outer_fold}")
        selected_ranks_audit.append({"outer_fold": outer_fold, "selected_rank": optimal_rank})

        print(f"  [Outer Refit] Executing final refit on Outer Train with Rank = {optimal_rank}...")
        outer_val_scores, final_outer_model = train_lora_model(
            outer_train_df, outer_val_df, tokenizer, rank=optimal_rank,
            epochs=epochs, device=device, hf_cache_dir=hf_cache_dir,
            code_column=code_column,
        )

        fold_oof_df = pd.DataFrame({
            "source_row_id": outer_val_df["source_row_id"].values,
            "project": outer_val_df["project"].values,
            "label": outer_val_df["label"].values.astype(int),
            "y_score": outer_val_scores,
            "fold": outer_fold,
        })
        oof_predictions_list.append(fold_oof_df)

        if outer_fold == 4:
            final_outer_model.save_pretrained(OUTPUT_DIR / "final_exp4_lora_adapter")
            print(f"  [Backup] Saved final fold LoRA adapter to: {OUTPUT_DIR / 'final_exp4_lora_adapter'}")

        del final_outer_model, outer_train_df, outer_val_df
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        completed_outer_folds.add(outer_fold)

        joblib.dump({
            "oof_predictions_list": oof_predictions_list,
            "selected_ranks_audit": selected_ranks_audit,
            "completed_outer_folds": completed_outer_folds,
        }, checkpoint_path)

        fold_min = (time.time() - fold_t0) / 60
        print(f"[EXP4] Outer fold {outer_fold} done in {fold_min:.1f} min | checkpoint saved")

    oof_df = pd.concat(oof_predictions_list).reset_index(drop=True)
    oof_df.to_csv(OUTPUT_DIR / "exp4_oof_predictions.csv", index=False)

    audit_ranks_df = pd.DataFrame(selected_ranks_audit)
    audit_ranks_df.to_csv(OUTPUT_DIR / "exp4_selected_rank_per_outer_fold.csv", index=False)

    best_global_rank = int(audit_ranks_df["selected_rank"].mode()[0])
    print(f"\n--- [EXP4] Starting Global Canonical Retraining with Winner Rank = {best_global_rank} ---")

    holdout_ids = set(outer_manifest_df.loc[outer_manifest_df["partition"] == "outer_holdout", "source_row_id"].tolist())
    holdout_frame = full_df[full_df["source_row_id"].isin(holdout_ids)].copy().reset_index(drop=True)

    holdout_scores, global_model = train_lora_model(
        development_frame, holdout_frame, tokenizer, rank=best_global_rank,
        epochs=epochs, device=device, hf_cache_dir=hf_cache_dir,
        code_column=code_column,
    )

    global_model.save_pretrained(OUTPUT_DIR / "final_canonical_lora_model")
    print(f"  Production model saved to: {OUTPUT_DIR / 'final_canonical_lora_model'}")

    holdout_predictions_df = pd.DataFrame({
        "source_row_id": holdout_frame["source_row_id"].values,
        "project": holdout_frame["project"].values,
        "label": holdout_frame["label"].values.astype(int),
        "y_score": holdout_scores,
        "fold": 0,
    })
    holdout_predictions_df.to_csv(OUTPUT_DIR / "exp4_holdout_predictions.csv", index=False)
    print(f"[EXP4] LoRA pipeline finished. Predictions saved to output.")

    del global_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return {
        "oof_predictions": oof_df,
        "selected_ranks_audit": audit_ranks_df,
        "holdout_predictions": holdout_predictions_df,
        "best_global_rank": best_global_rank,
        "output_dir": OUTPUT_DIR,
    }


if __name__ == "__main__":
    run_exp4_nested_pipeline(
        abstracted_parquet_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/processed/rdiversevul_cs1_normalized_plus_abstracted_v1.parquet",
        inner_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/inner_cv/cs1_project_grouped_5fold_manifest.parquet",
        outer_manifest_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/manifests/cs1_project_holdout20_innercv_v1/outer_holdout/cs1_outer_project_holdout_manifest.parquet",
        output_dir_path="/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/outputs/exp4_lora_codeberta_output",
        hf_cache_dir="/content/drive/MyDrive/IntelligentSystemProject/hf_cache",
    )