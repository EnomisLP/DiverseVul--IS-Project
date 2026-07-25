from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import configure_huggingface_cache, load_code_tokenizer, DEFAULT_CODE_TOKENIZER
from case_study_2.exp4.exp4_lora import train_lora_model
from case_study_1 import split_manifest
from case_study_1 import evaluation
from case_study_1.evaluation import EvaluationConfig
from case_study_1.confidence_intervals import bootstrap_metric_ci, format_ci_report


EXP4_VERSION = "cs2-exp4-codeberta-lora-v1"


@dataclass(frozen=True)
class Exp4Config:
    experiment_name: str = "cs2_exp4_codeberta_lora"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    hf_cache_dir: Optional[str] = None
    max_length: int = 512
    train_batch_size: int = 16
    grad_accum_steps: int = 2
    epochs: int = 3

    rank_grid: Tuple[int, ...] = (8, 16)
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    decision_threshold: float = 0.50

    n_splits: int = 5
    random_state: int = 42
    verbose: bool = True


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> Dict[str, Path]:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "selected": root / f"{prefix}_selected_rank.json",
        "training": root / f"{prefix}_outer_training.json",
    }


def _write_outer_checkpoint(output_dir: Path, outer_fold_id: int, predictions: pd.DataFrame, selected: dict, training: dict) -> None:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    predictions.to_parquet(paths["predictions"], index=False)
    with paths["selected"].open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, default=str)
    with paths["training"].open("w", encoding="utf-8") as f:
        json.dump(training, f, indent=2, default=str)


def _load_outer_checkpoint(output_dir: Path, outer_fold_id: int) -> Optional[dict]:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    if not all(p.exists() for p in paths.values()):
        return None
    with paths["selected"].open("r", encoding="utf-8") as f:
        selected = json.load(f)
    with paths["training"].open("r", encoding="utf-8") as f:
        training = json.load(f)
    return {
        "predictions": pd.read_parquet(paths["predictions"]),
        "selected": selected,
        "training": training,
    }


def _update_run_state(state_path: Path, completed_folds, status: str) -> None:
    state = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_outer_folds": sorted(int(f) for f in completed_folds),
    }
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_exp4_nested_rank(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    config: Exp4Config,
    output_dir: Path,
    resume: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("EXP-4 LoRA fine-tuning requires a CUDA device.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "exp4_nested_run_state.json"

    configure_huggingface_cache(config.hf_cache_dir)
    tokenizer = load_code_tokenizer(DEFAULT_CODE_TOKENIZER, hf_cache_dir=config.hf_cache_dir)

    fold_ids = sorted(development_manifest[config.fold_column].unique().tolist())

    oof_parts = []
    selected_rows = []
    outer_training_rows = []
    completed_folds = []

    t0 = time.time()

    for outer_fold_id in fold_ids:
        checkpoint = _load_outer_checkpoint(output_dir, outer_fold_id) if resume else None
        if checkpoint is not None:
            oof_parts.append(checkpoint["predictions"])
            selected_rows.append(checkpoint["selected"])
            outer_training_rows.append(checkpoint["training"])
            completed_folds.append(outer_fold_id)
            if config.verbose:
                print(f"[nested] Outer fold {outer_fold_id}: loaded from checkpoint.")
            continue

        fold_t0 = time.time()
        if config.verbose:
            print(f"[nested] Outer fold {outer_fold_id}: inner rank grid search...")

        outer_train_ids = development_manifest.loc[
            development_manifest[config.fold_column] != outer_fold_id, config.source_id_column
        ]
        outer_val_ids = development_manifest.loc[
            development_manifest[config.fold_column] == outer_fold_id, config.source_id_column
        ]
        outer_train_df = development_frame[development_frame[config.source_id_column].isin(outer_train_ids)].reset_index(drop=True)
        outer_val_df = development_frame[development_frame[config.source_id_column].isin(outer_val_ids)].reset_index(drop=True)

        inner_cv_manifest = split_manifest.load_manifest(
            outer_train_df[[config.source_id_column, config.label_column, config.project_column]],
            config=split_manifest.SplitConfig(n_splits=config.inner_n_splits, random_state=config.inner_random_state, shuffle=True),
        )

        rank_performance = {}
        for rank_candidate in config.rank_grid:
            inner_praucs = []
            for inner_fold in range(config.inner_n_splits):
                inner_train_ids = inner_cv_manifest[inner_cv_manifest["fold"] != inner_fold][config.source_id_column]
                inner_val_ids = inner_cv_manifest[inner_cv_manifest["fold"] == inner_fold][config.source_id_column]
                inner_train_df = outer_train_df[outer_train_df[config.source_id_column].isin(inner_train_ids)]
                inner_val_df = outer_train_df[outer_train_df[config.source_id_column].isin(inner_val_ids)]

                val_scores, tmp_model = train_lora_model(
                    inner_train_df, inner_val_df, tokenizer, rank=rank_candidate,
                    epochs=config.epochs, batch_size=config.train_batch_size,
                    grad_accum_steps=config.grad_accum_steps, device=device,
                    hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
                    max_length=config.max_length,
                )
                prauc = average_precision_score(inner_val_df[config.label_column].values, val_scores)
                inner_praucs.append(prauc)

                del tmp_model, inner_train_df, inner_val_df
                gc.collect()
                torch.cuda.empty_cache()

            rank_performance[rank_candidate] = float(np.mean(inner_praucs))
            if config.verbose:
                print(f"    rank={rank_candidate} | inner mean PR-AUC={rank_performance[rank_candidate]:.4f}")

        optimal_rank = max(rank_performance, key=rank_performance.get)

        outer_val_scores, final_outer_model = train_lora_model(
            outer_train_df, outer_val_df, tokenizer, rank=optimal_rank,
            epochs=config.epochs, batch_size=config.train_batch_size,
            grad_accum_steps=config.grad_accum_steps, device=device,
            hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
            max_length=config.max_length,
        )

        fold_oof = pd.DataFrame({
            config.source_id_column: outer_val_df[config.source_id_column].values,
            config.project_column: outer_val_df[config.project_column].values,
            "label": outer_val_df[config.label_column].astype(int).values,
            "y_score": outer_val_scores,
            "fold": outer_fold_id,
        })

        selected_row = {"outer_fold_id": outer_fold_id, "selected_rank": optimal_rank}
        training_row = {
            "outer_fold_id": outer_fold_id,
            "selected_rank": optimal_rank,
            "n_train": int(len(outer_train_df)),
            "n_val": int(len(outer_val_df)),
            "elapsed_minutes": (time.time() - fold_t0) / 60,
        }

        if outer_fold_id == fold_ids[-1]:
            final_outer_model.save_pretrained(output_dir / "final_exp4_lora_adapter")

        _write_outer_checkpoint(output_dir, outer_fold_id, fold_oof, selected_row, training_row)

        oof_parts.append(fold_oof)
        selected_rows.append(selected_row)
        outer_training_rows.append(training_row)
        completed_folds.append(outer_fold_id)

        _update_run_state(state_path, completed_folds, status="running")

        del outer_train_df, outer_val_df, final_outer_model
        gc.collect()
        torch.cuda.empty_cache()

        if config.verbose:
            print(f"[nested] Outer fold {outer_fold_id} done in {training_row['elapsed_minutes']:.1f} min | checkpoint saved")

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)

    eval_config = EvaluationConfig(threshold=config.decision_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    selected_df = pd.DataFrame(selected_rows)
    outer_training_df = pd.DataFrame(outer_training_rows)

    artifacts = {
        "oof_predictions": output_dir / "exp4_nested_oof_predictions.parquet",
        "selected_rank_per_fold": output_dir / "exp4_selected_rank_per_outer_fold.csv",
        "outer_training_audit": output_dir / "exp4_outer_training_audit.csv",
        "run_metadata": output_dir / "exp4_nested_run_metadata.json",
    }
    oof_predictions.to_parquet(artifacts["oof_predictions"], index=False)
    selected_df.to_csv(artifacts["selected_rank_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)

    metadata = {
        "exp4_version": EXP4_VERSION,
        "config": {**asdict(config), "rank_grid": list(config.rank_grid)},
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    _update_run_state(state_path, completed_folds, status="completed")

    return {
        "oof_predictions": oof_predictions,
        "evaluation": eval_results,
        "selected_rank": selected_df,
        "outer_fold_training": outer_training_df,
        "artifacts": artifacts,
        "tokenizer": tokenizer,
    }


def run_exp4_canonical_retrain(
    development_frame: pd.DataFrame,
    tokenizer,
    selected_rank: int,
    holdout_frame: pd.DataFrame,
    config: Exp4Config,
    output_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_scores, global_model = train_lora_model(
        development_frame, holdout_frame, tokenizer, rank=selected_rank,
        epochs=config.epochs, batch_size=config.train_batch_size,
        grad_accum_steps=config.grad_accum_steps, device=device,
        hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
        max_length=config.max_length,
    )
    global_model.save_pretrained(output_dir / "final_canonical_lora_model")

    holdout_predictions = pd.DataFrame({
        config.source_id_column: holdout_frame[config.source_id_column].values,
        config.project_column: holdout_frame[config.project_column].values,
        "label": holdout_frame[config.label_column].astype(int).values,
        "y_score": holdout_scores,
        "fold": 0,
    })

    del global_model
    gc.collect()
    torch.cuda.empty_cache()

    return {"holdout_predictions": holdout_predictions}


def run_exp4_holdout_evaluation(
    holdout_predictions: pd.DataFrame,
    config: Exp4Config,
    output_dir: Path,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_predictions.to_csv(output_dir / "exp4_holdout_predictions.csv", index=False)

    eval_config = EvaluationConfig(threshold=config.decision_threshold, expected_n_folds=1)
    holdout_metrics = evaluation.evaluate_oof_predictions(holdout_predictions, config=eval_config)
    print(evaluation.format_metric_report(holdout_metrics["pooled_metrics"]))

    ci_result = bootstrap_metric_ci(
        holdout_predictions,
        metric="average_precision_pr_auc",
        group_column="project",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=config.random_state,
    )
    with open(output_dir / "exp4_holdout_pr_auc_bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci_result.as_dict(), f, indent=2)
    print(format_ci_report(ci_result))

    y_true = holdout_predictions["label"].values
    y_score = holdout_predictions["y_score"].values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = holdout_metrics["pooled_metrics"]["average_precision_pr_auc"]
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="b", label=f"EXP-4 LoRA (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Frozen Outer Holdout")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "exp4_outer_holdout_pr_curve.png")
    plt.close()

    y_pred = (y_score >= config.decision_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Frozen Outer Holdout")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Non-Vuln (0)", "Vuln (1)"])
    plt.yticks([0, 1], ["Non-Vuln (0)", "Vuln (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "exp4_outer_holdout_confusion_matrix.png")
    plt.close()

    return {
        "holdout_metrics": holdout_metrics,
        "bootstrap_ci": ci_result.as_dict(),
        "y_pred": y_pred,
    }