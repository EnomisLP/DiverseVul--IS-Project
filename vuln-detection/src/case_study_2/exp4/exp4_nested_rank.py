from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from case_study_2.models import configure_huggingface_cache, load_code_tokenizer, DEFAULT_NEOBERT_TOKENIZER
from case_study_2.exp4.exp4_lora import train_lora_model_safe
from utils import split_manifest
from utils import evaluation
from utils.evaluation import EvaluationConfig, select_f1_threshold


EXP4_VERSION = "cs2-exp4-neobert-lora-v3-fixed-rank-rotating-5fold"


@dataclass(frozen=True)
class Exp4Config:
    """Declared reproducible configuration for CS2-EXP4 (NeoBERT LoRA fine-tuning)."""

    experiment_name: str = "cs2_exp4_neobert_lora"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    hf_cache_dir: Optional[str] = None
    max_length: int = 512
    train_batch_size: int = 16
    grad_accum_steps: int = 2

    # `epochs` is a ceiling, not a fixed count: training early-stops on
    # validation PR-AUC and returns the best checkpoint.
    epochs: int = 10
    early_stopping: bool = True
    patience: int = 2
    min_epochs: int = 2

    # Rank fixed at the literature-standard default for encoder models of
    # this size (Hu et al. 2021 LoRA paper; HuggingFace PEFT docs), not
    # searched, to keep NeoBERT fine-tuning within a practical compute
    # budget (a 3-candidate x 3-inner-fold rank search would need 10x more
    # NeoBERT trainings for a gain the literature says is usually marginal).
    rank: int = 16
    # Standard LoRA convention (alpha = 2 x rank), same reasoning as rank above.
    lora_alpha_multiplier: int = 2
    # 2e-4 is a standard LoRA fine-tuning learning rate; fixed for the same reason.
    learning_rate: float = 2e-4

    # Inner CV is still used, but only to calibrate the decision threshold
    # on validation folds the outer test fold never touches.
    inner_n_splits: int = 3
    inner_random_state: int = 20260707

    num_workers: int = 6
    eval_batch_size: int = 64

    n_splits: int = 5
    random_state: int = 42
    verbose: bool = True


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> Dict[str, Path]:
    """Filesystem locations for one outer fold's resumable checkpoint."""
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "selected": root / f"{prefix}_selected.json",
        "training": root / f"{prefix}_outer_training.json",
        "history": root / f"{prefix}_training_history.parquet",
    }


def _write_outer_checkpoint(output_dir: Path, outer_fold_id: int, predictions: pd.DataFrame, selected: dict, training: dict, history: pd.DataFrame) -> None:
    """Persist one outer fold's final result so a later run can resume without refitting."""
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    predictions.to_parquet(paths["predictions"], index=False)
    with paths["selected"].open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, default=str)
    with paths["training"].open("w", encoding="utf-8") as f:
        json.dump(training, f, indent=2, default=str)
    history.to_parquet(paths["history"], index=False)


def _load_outer_checkpoint(output_dir: Path, outer_fold_id: int) -> Optional[dict]:
    """Load one outer fold's checkpoint if it exists and is complete."""
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    if not all(p.exists() for p in paths.values()):
        return None
    with paths["selected"].open("r", encoding="utf-8") as f:
        selected = json.load(f)
    with paths["training"].open("r", encoding="utf-8") as f:
        training = json.load(f)
    return {
        "predictions": pd.read_parquet(paths["predictions"]), "selected": selected, "training": training,
        "history": pd.read_parquet(paths["history"]),
    }


def _update_run_state(state_path: Path, completed_folds, status: str) -> None:
    """Persist which outer folds are done, for resumability and progress inspection."""
    state = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_outer_folds": sorted(int(f) for f in completed_folds),
    }
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _inner_manifest(frame: pd.DataFrame, config: Exp4Config, outer_fold_id: int) -> pd.DataFrame:
    """Project-grouped, stratified 3-fold split of one outer fold's training rows."""
    inner_split_config = split_manifest.SplitConfig(
        n_splits=config.inner_n_splits,
        random_state=config.inner_random_state + int(outer_fold_id),
        source_id_column=config.source_id_column,
        label_column=config.label_column,
        group_column=config.project_column,
    )
    return split_manifest.create_project_grouped_manifest(
        frame[[config.source_id_column, config.label_column, config.project_column]], config=inner_split_config
    )


def _calibrate_threshold(
    outer_train_df: pd.DataFrame,
    tokenizer,
    device: torch.device,
    config: Exp4Config,
    outer_fold_id: int,
) -> dict:
    """Calibrate the decision threshold via 3-fold inner CV at the fixed rank, on this fold's training projects only."""
    lora_alpha = config.rank * config.lora_alpha_multiplier
    inner_manifest_df = _inner_manifest(outer_train_df, config, outer_fold_id)

    fold_predictions: List[pd.DataFrame] = []
    for inner_fold_id in range(config.inner_n_splits):
        tr_ids = set(inner_manifest_df.loc[inner_manifest_df["fold"] != inner_fold_id, config.source_id_column])
        va_ids = set(inner_manifest_df.loc[inner_manifest_df["fold"] == inner_fold_id, config.source_id_column])
        tr_frame = outer_train_df[outer_train_df[config.source_id_column].isin(tr_ids)].reset_index(drop=True)
        va_frame = outer_train_df[outer_train_df[config.source_id_column].isin(va_ids)].reset_index(drop=True)

        val_scores, tmp_model, _ = train_lora_model_safe(
            tr_frame, va_frame, tokenizer, rank=config.rank, lora_alpha=lora_alpha,
            learning_rate=config.learning_rate, epochs=config.epochs,
            batch_size=config.train_batch_size, grad_accum_steps=config.grad_accum_steps,
            eval_batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device,
            hf_cache_dir=config.hf_cache_dir, code_column=config.code_column, max_length=config.max_length,
            log_prefix="    ", early_stopping=config.early_stopping, patience=config.patience,
            min_epochs=config.min_epochs,
        )
        fold_predictions.append(
            pd.DataFrame({"source_row_id": va_frame[config.source_id_column].values,
                          "label": va_frame[config.label_column].astype(int).values, "y_score": val_scores})
        )
        del tmp_model
        gc.collect()
        torch.cuda.empty_cache()

    pooled = pd.concat(fold_predictions, ignore_index=True)
    selected_threshold, threshold_metrics = select_f1_threshold(pooled["label"], pooled["y_score"])

    print(f"[nested] rank={config.rank} (alpha={lora_alpha}, fixed) | threshold={selected_threshold:.2f} "
          f"(inner validation F1={threshold_metrics['f1']:.4f})")

    return {
        "rank": config.rank,
        "lora_alpha": lora_alpha,
        "decision_threshold": selected_threshold,
        "inner_validation_f1": float(threshold_metrics["f1"]),
    }


def run_exp4_nested_rank(
    dataset_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Exp4Config,
    output_dir: Path,
    resume: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the official rotating 5-fold CS2-EXP4 experiment, with a 3-fold inner-CV threshold calibration per outer fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("EXP-4 LoRA fine-tuning requires a CUDA device.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "exp4_nested_run_state.json"

    configure_huggingface_cache(config.hf_cache_dir)
    tokenizer = load_code_tokenizer(DEFAULT_NEOBERT_TOKENIZER, hf_cache_dir=config.hf_cache_dir)

    fold_ids = sorted(manifest[config.fold_column].unique().tolist())
    print(f"[nested] Starting EXP-4 rotating {len(fold_ids)}-fold run (NeoBERT LoRA), rank={config.rank} (fixed)")
    print(f"[nested] Threshold calibration: {config.inner_n_splits}-fold inner CV | "
          f"Refit phase: {config.epochs} epoch(s) ceiling, full outer-train")

    oof_parts = []
    selected_rows = []
    outer_training_rows = []
    training_histories = []
    completed_folds = []
    t0 = time.time()

    for outer_fold_id in fold_ids:
        checkpoint = _load_outer_checkpoint(output_dir, outer_fold_id) if resume else None
        if checkpoint is not None:
            oof_parts.append(checkpoint["predictions"])
            selected_rows.append(checkpoint["selected"])
            outer_training_rows.append(checkpoint["training"])
            training_histories.append(checkpoint["history"])
            completed_folds.append(outer_fold_id)
            if config.verbose:
                print(f"[nested] Outer fold {outer_fold_id}: loaded from checkpoint, skipping.")
            continue

        fold_t0 = time.time()
        print(f"\n[nested] Outer fold {outer_fold_id} ({len(completed_folds)+1}/{len(fold_ids)}) starting")

        outer_train_ids = manifest.loc[manifest[config.fold_column] != outer_fold_id, config.source_id_column]
        outer_val_ids = manifest.loc[manifest[config.fold_column] == outer_fold_id, config.source_id_column]
        outer_train_df = dataset_frame[dataset_frame[config.source_id_column].isin(outer_train_ids)].reset_index(drop=True)
        outer_val_df = dataset_frame[dataset_frame[config.source_id_column].isin(outer_val_ids)].reset_index(drop=True)
        print(f"[nested] outer_train={len(outer_train_df)} rows | outer_val={len(outer_val_df)} rows")

        selection = _calibrate_threshold(outer_train_df, tokenizer, device, config, outer_fold_id)

        print(f"[nested] --- final refit on full outer_train, rank={selection['rank']}, "
              f"alpha={selection['lora_alpha']}, {config.epochs} epochs ---")
        refit_t0 = time.time()
        outer_val_scores, final_outer_model, refit_history = train_lora_model_safe(
            outer_train_df, outer_val_df, tokenizer, rank=selection["rank"],
            lora_alpha=selection["lora_alpha"], learning_rate=config.learning_rate,
            epochs=config.epochs, batch_size=config.train_batch_size, grad_accum_steps=config.grad_accum_steps,
            eval_batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device,
            hf_cache_dir=config.hf_cache_dir, code_column=config.code_column, max_length=config.max_length,
            log_prefix="    ", early_stopping=config.early_stopping, patience=config.patience,
            min_epochs=config.min_epochs,
        )
        fold_history = pd.DataFrame(refit_history)
        fold_history.insert(0, "fold", int(outer_fold_id))
        print(f"[nested] refit done in {(time.time()-refit_t0)/60:.1f} min")

        fold_oof = pd.DataFrame({
            config.source_id_column: outer_val_df[config.source_id_column].values,
            config.project_column: outer_val_df[config.project_column].values,
            "label": outer_val_df[config.label_column].astype(int).values,
            "y_score": outer_val_scores,
            "fold": outer_fold_id,
        })

        selected_row = {"outer_fold_id": outer_fold_id, **selection}
        training_row = {
            **selected_row,
            "n_train": int(len(outer_train_df)),
            "n_val": int(len(outer_val_df)),
            "elapsed_minutes": (time.time() - fold_t0) / 60,
        }

        if outer_fold_id == fold_ids[-1]:
            final_outer_model.save_pretrained(output_dir / "final_exp4_lora_adapter")
            print(f"[nested] saved final fold LoRA adapter to {output_dir / 'final_exp4_lora_adapter'}")

        _write_outer_checkpoint(output_dir, outer_fold_id, fold_oof, selected_row, training_row, fold_history)

        oof_parts.append(fold_oof)
        selected_rows.append(selected_row)
        outer_training_rows.append(training_row)
        training_histories.append(fold_history)
        completed_folds.append(outer_fold_id)

        _update_run_state(state_path, completed_folds, status="running")

        del outer_train_df, outer_val_df, final_outer_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[nested] Outer fold {outer_fold_id} done in {training_row['elapsed_minutes']:.1f} min | "
              f"total elapsed {(time.time()-t0)/60:.1f} min")

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows)
    outer_training_df = pd.DataFrame(outer_training_rows)
    training_history_df = pd.concat(training_histories, ignore_index=True).sort_values(["fold", "epoch"]).reset_index(drop=True)

    mean_threshold = float(selected_df["decision_threshold"].mean())
    eval_config = EvaluationConfig(threshold=mean_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    artifacts = {
        "oof_predictions": output_dir / "exp4_nested_oof_predictions.parquet",
        "selected_per_fold": output_dir / "exp4_selected_per_outer_fold.csv",
        "outer_training_audit": output_dir / "exp4_outer_training_audit.csv",
        "training_history": output_dir / "exp4_training_history.csv",
        "run_metadata": output_dir / "exp4_nested_run_metadata.json",
    }
    eval_results["predictions"].to_parquet(artifacts["oof_predictions"], index=False)
    selected_df.to_csv(artifacts["selected_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)
    training_history_df.to_csv(artifacts["training_history"], index=False)

    metadata = {
        "exp4_version": EXP4_VERSION,
        "config": asdict(config),
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    _update_run_state(state_path, completed_folds, status="completed")
    print(f"\n[nested] EXP-4 rotating {len(fold_ids)}-fold run complete in {(time.time()-t0)/60:.1f} min")

    return {
        "oof_predictions": eval_results["predictions"],
        "evaluation": eval_results,
        "selected": selected_df,
        "outer_fold_training": outer_training_df,
        "training_history": training_history_df,
        "artifacts": artifacts,
        "tokenizer": tokenizer,
    }


__all__ = [
    "EXP4_VERSION",
    "Exp4Config",
    "run_exp4_nested_rank",
]
