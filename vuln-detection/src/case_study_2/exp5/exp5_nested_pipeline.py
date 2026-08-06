from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

from case_study_2.data_loader import create_dataloader, get_class_weights
from case_study_2.models import configure_huggingface_cache, load_code_tokenizer, DEFAULT_CODE_TOKENIZER
from case_study_2.exp5.exp5_heft import train_heft_model_safe
from case_study_1 import split_manifest
from case_study_1 import evaluation
from case_study_1.evaluation import EvaluationConfig
from case_study_1.confidence_intervals import bootstrap_metric_ci, format_ci_report


EXP5_VERSION = "cs2-exp5-codeberta-heft-v1"


@dataclass(frozen=True)
class Exp5Config:
    experiment_name: str = "cs2_exp5_codeberta_heft"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    hf_cache_dir: Optional[str] = None
    max_length: int = 512
    train_batch_size: int = 16
    grad_accum_steps: int = 2
    lora_epochs: int = 2
    reft_epochs: int = 2

    rank_grid: Tuple[int, ...] = (8, 16)
    reft_rank: int = 4
    layer_target: int = 4
    
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    decision_threshold: float = 0.50

    search_epochs: int = 1
    search_n_splits: int = 4

    num_workers: int = 6
    eval_batch_size: int = 64

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


def _search_checkpoint_path(output_dir: Path, outer_fold_id: int) -> Path:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"outer_fold_{outer_fold_id}_search_progress.json"


def _load_search_checkpoint(output_dir: Path, outer_fold_id: int) -> Dict[str, float]:
    path = _search_checkpoint_path(output_dir, outer_fold_id)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def _save_search_checkpoint(output_dir: Path, outer_fold_id: int, rank_performance: Dict[int, float]) -> None:
    path = _search_checkpoint_path(output_dir, outer_fold_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in rank_performance.items()}, f, indent=2)


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


def run_exp5_nested_rank(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    config: Exp5Config,
    output_dir: Path,
    resume: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("EXP-5 HEFT fine-tuning requires a CUDA device.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "exp5_nested_run_state.json"

    configure_huggingface_cache(config.hf_cache_dir)
    tokenizer = load_code_tokenizer(DEFAULT_CODE_TOKENIZER, hf_cache_dir=config.hf_cache_dir)

    fold_ids = sorted(development_manifest[config.fold_column].unique().tolist())
    print(f"[nested] Starting EXP-5 nested rank search over {len(fold_ids)} outer folds, rank_grid={config.rank_grid}")
    print(
        f"[nested] Search phase: {config.search_epochs} epoch(s)/phase, single held-out split | "
        f"Refit phase: {config.lora_epochs} LoRA + {config.reft_epochs} ReFT epoch(s), full training"
    )

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
                print(f"[nested] Outer fold {outer_fold_id}: loaded from checkpoint, skipping.")
            continue

        fold_t0 = time.time()
        print(f"\n=================== OUTER FOLD {outer_fold_id} ({len(completed_folds)+1}/{len(fold_ids)}) ===================")

        outer_train_ids = development_manifest.loc[
            development_manifest[config.fold_column] != outer_fold_id, config.source_id_column
        ]
        outer_val_ids = development_manifest.loc[
            development_manifest[config.fold_column] == outer_fold_id, config.source_id_column
        ]
        outer_train_df = development_frame[development_frame[config.source_id_column].isin(outer_train_ids)].reset_index(drop=True)
        outer_val_df = development_frame[development_frame[config.source_id_column].isin(outer_val_ids)].reset_index(drop=True)
        print(f"[nested] outer_train={len(outer_train_df)} rows | outer_val={len(outer_val_df)} rows")

        search_split_config = split_manifest.SplitConfig(
            n_splits=config.search_n_splits,
            random_state=config.inner_random_state,
            shuffle=True,
            source_id_column=config.source_id_column,
            label_column=config.label_column,
            group_column=config.project_column,
        )
        search_manifest = split_manifest.create_project_grouped_manifest(
            outer_train_df[[config.source_id_column, config.label_column, config.project_column]],
            config=search_split_config,
        )
        search_train_ids = search_manifest.loc[search_manifest["fold"] != 0, config.source_id_column]
        search_val_ids = search_manifest.loc[search_manifest["fold"] == 0, config.source_id_column]
        search_train_df = outer_train_df[outer_train_df[config.source_id_column].isin(search_train_ids)]
        search_val_df = outer_train_df[outer_train_df[config.source_id_column].isin(search_val_ids)]
        print(f"[nested] rank search split: train={len(search_train_df)} rows | val={len(search_val_df)} rows")

        rank_performance = _load_search_checkpoint(output_dir, outer_fold_id) if resume else {}
        if rank_performance:
            print(f"[nested] resuming rank search, already have: {rank_performance}")

        for rank_candidate in config.rank_grid:
            if rank_candidate in rank_performance:
                print(f"[nested] rank={rank_candidate}: already evaluated (PR-AUC={rank_performance[rank_candidate]:.4f}), skipping.")
                continue

            rank_t0 = time.time()
            print(f"[nested] --- evaluating rank candidate {rank_candidate} ---")

            val_scores, tmp_model = train_heft_model_safe(
                search_train_df, search_val_df, tokenizer, rank=rank_candidate,
                lora_epochs=config.search_epochs, reft_epochs=config.search_epochs,
                batch_size=config.train_batch_size,
                grad_accum_steps=config.grad_accum_steps, eval_batch_size=config.eval_batch_size,
                num_workers=config.num_workers, device=device,
                hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
                max_length=config.max_length, log_prefix="    ",
                reft_rank=config.reft_rank, layer_target=config.layer_target,
            )
            prauc = float(average_precision_score(search_val_df[config.label_column].values, val_scores))
            rank_performance[rank_candidate] = prauc

            print(f"[nested] rank={rank_candidate} | PR-AUC={prauc:.4f} | {(time.time()-rank_t0)/60:.1f} min")

            _save_search_checkpoint(output_dir, outer_fold_id, rank_performance)

            del tmp_model
            gc.collect()
            torch.cuda.empty_cache()

        optimal_rank = max(rank_performance, key=rank_performance.get)
        print(f"[nested] Selected rank={optimal_rank} for outer fold {outer_fold_id} | scores={rank_performance}")

        print(
            f"[nested] --- final refit on full outer_train, rank={optimal_rank}, "
            f"{config.lora_epochs} LoRA + {config.reft_epochs} ReFT epochs ---"
        )
        refit_t0 = time.time()
        outer_val_scores, final_outer_model = train_heft_model_safe(
            outer_train_df, outer_val_df, tokenizer, rank=optimal_rank,
            lora_epochs=config.lora_epochs, reft_epochs=config.reft_epochs,
            batch_size=config.train_batch_size,
            grad_accum_steps=config.grad_accum_steps, eval_batch_size=config.eval_batch_size,
            num_workers=config.num_workers, device=device,
            hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
            max_length=config.max_length, log_prefix="    ",
            reft_rank=config.reft_rank, layer_target=config.layer_target,
        )
        print(f"[nested] refit done in {(time.time()-refit_t0)/60:.1f} min")

        fold_oof = pd.DataFrame({
            config.source_id_column: outer_val_df[config.source_id_column].values,
            config.project_column: outer_val_df[config.project_column].values,
            "label": outer_val_df[config.label_column].astype(int).values,
            "y_score": outer_val_scores,
            "fold": outer_fold_id,
        })

        selected_row = {"outer_fold_id": outer_fold_id, "selected_rank": optimal_rank, "search_scores": rank_performance}
        training_row = {
            "outer_fold_id": outer_fold_id,
            "selected_rank": optimal_rank,
            "n_train": int(len(outer_train_df)),
            "n_val": int(len(outer_val_df)),
            "elapsed_minutes": (time.time() - fold_t0) / 60,
        }

        if outer_fold_id == fold_ids[-1]:
            final_outer_model.save_pretrained(output_dir / "final_exp5_heft_adapter")
            print(f"[nested] saved final fold HEFT adapter to {output_dir / 'final_exp5_heft_adapter'}")

        _write_outer_checkpoint(output_dir, outer_fold_id, fold_oof, selected_row, training_row)

        oof_parts.append(fold_oof)
        selected_rows.append(selected_row)
        outer_training_rows.append(training_row)
        completed_folds.append(outer_fold_id)

        _update_run_state(state_path, completed_folds, status="running")

        del outer_train_df, outer_val_df, search_train_df, search_val_df, final_outer_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[nested] Outer fold {outer_fold_id} done in {training_row['elapsed_minutes']:.1f} min | checkpoint saved | total elapsed {(time.time()-t0)/60:.1f} min")

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)

    eval_config = EvaluationConfig(threshold=config.decision_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    selected_df = pd.DataFrame(selected_rows)
    outer_training_df = pd.DataFrame(outer_training_rows)

    artifacts = {
        "oof_predictions": output_dir / "exp5_nested_oof_predictions.parquet",
        "selected_rank_per_fold": output_dir / "exp5_selected_rank_per_outer_fold.csv",
        "outer_training_audit": output_dir / "exp5_outer_training_audit.csv",
        "run_metadata": output_dir / "exp5_nested_run_metadata.json",
    }
    oof_predictions.to_parquet(artifacts["oof_predictions"], index=False)
    selected_df.to_csv(artifacts["selected_rank_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)

    metadata = {
        "exp5_version": EXP5_VERSION,
        "config": {**asdict(config), "rank_grid": list(config.rank_grid)},
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    _update_run_state(state_path, completed_folds, status="completed")

    print(f"\n[nested] EXP-5 nested rank search complete in {(time.time()-t0)/60:.1f} min")

    return {
        "oof_predictions": oof_predictions,
        "evaluation": eval_results,
        "selected_rank": selected_df,
        "outer_fold_training": outer_training_df,
        "artifacts": artifacts,
        "tokenizer": tokenizer,
    }


def run_exp5_canonical_retrain(
    development_frame: pd.DataFrame,
    tokenizer,
    selected_rank: int,
    holdout_frame: pd.DataFrame,
    config: Exp5Config,
    output_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[canonical] Retraining on full development set ({len(development_frame)} rows), "
        f"rank={selected_rank}, {config.lora_epochs} LoRA + {config.reft_epochs} ReFT epochs"
    )
    t0 = time.time()

    holdout_scores, global_model = train_heft_model_safe(
        development_frame, holdout_frame, tokenizer, rank=selected_rank,
        lora_epochs=config.lora_epochs, reft_epochs=config.reft_epochs,
        batch_size=config.train_batch_size,
        grad_accum_steps=config.grad_accum_steps, eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers, device=device,
        hf_cache_dir=config.hf_cache_dir, code_column=config.code_column,
        max_length=config.max_length, log_prefix="  ",
        reft_rank=config.reft_rank, layer_target=config.layer_target,
    )
    global_model.save_pretrained(output_dir / "final_canonical_heft_model")
    print(f"[canonical] Done in {(time.time()-t0)/60:.1f} min | model saved to {output_dir / 'final_canonical_heft_model'}")

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


def run_exp5_holdout_evaluation(
    holdout_predictions: pd.DataFrame,
    config: Exp5Config,
    output_dir: Path,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_predictions.to_csv(output_dir / "exp5_holdout_predictions.csv", index=False)

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
    with open(output_dir / "exp5_holdout_pr_auc_bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci_result.as_dict(), f, indent=2)
    print(format_ci_report(ci_result))

    y_true = holdout_predictions["label"].values
    y_score = holdout_predictions["y_score"].values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = holdout_metrics["pooled_metrics"]["average_precision_pr_auc"]
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="b", label=f"EXP-5 HEFT (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Frozen Outer Holdout")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "exp5_outer_holdout_pr_curve.png")
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
    plt.savefig(output_dir / "exp5_outer_holdout_confusion_matrix.png")
    plt.close()

    return {
        "holdout_metrics": holdout_metrics,
        "bootstrap_ci": ci_result.as_dict(),
        "y_pred": y_pred,
    }