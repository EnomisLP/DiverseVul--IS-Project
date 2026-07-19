"""
EXP-3: NeoBERT frozen-encoder linear probe (Case Study 2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

from case_study_2.data_loader import create_dataloader
from case_study_2.models import (
    DEFAULT_NEOBERT_MODEL,
    DEFAULT_NEOBERT_TOKENIZER,
    configure_huggingface_cache,
    load_neobert_tokenizer,
    load_neobert_encoder,
)

# Shared, experiment-agnostic modules -- same ones EXP-0/1/2 use.
from case_study_1 import split_manifest
from case_study_1 import evaluation
from case_study_1 import confidence_intervals


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #

@dataclass
class Exp3Config:
    """Static, non-tuned configuration for the EXP-3 linear probe."""

    experiment_name: str = "cs2_exp3_neobert_linear_probe"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    hf_cache_dir: Optional[str] = None
    max_length: int = 512
    dtype_policy: str = "bfloat16"
    embedding_batch_size: int = 64

    logistic_max_iter: int = 2000
    logistic_solver: str = "lbfgs"
    class_weight: str = "balanced"
    decision_threshold: float = 0.50

    random_state: int = 42
    verbose: bool = True


@dataclass
class NestedProbeConfig:
    """Nested-CV tuning configuration, mirrors NestedAlphaConfig in EXP-0."""

    experiment_name: str = "cs2_exp3_nested_probe_dev_grouped"
    C_grid: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    selection_metric: str = "average_precision_pr_auc"
    decision_threshold: float = 0.50
    tie_break_rule: str = "higher_C_then_grid_order"
    verbose: bool = True


def resolve_device(require_cuda: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            "EXP-3 linear probe requires a CUDA GPU for NeoBERT embedding "
            "extraction. No CUDA device was found."
        )
    return "cpu"


# --------------------------------------------------------------------------- #
# Embedding extraction (the only step that touches the GPU/encoder)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_embeddings(
    encoder,
    tokenizer,
    frame: pd.DataFrame,
    config: Exp3Config,
    device: str,
    cache_path: Optional[Path] = None,
    checkpoint_every: int = 100,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        print(f"  [embed] Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    if device != "cuda":
        raise RuntimeError("extract_embeddings must run on a CUDA device.")

   
    frame = frame.reset_index(drop=True)
    code_lengths = frame[config.code_column].fillna("").astype(str).str.len()
    sort_order = code_lengths.sort_values(kind="mergesort").index.to_numpy() 
    sorted_frame = frame.iloc[sort_order].reset_index(drop=True)
   
    inverse_order = np.argsort(sort_order)

    n_rows = len(sorted_frame)
    n_batches_total = -(-n_rows // config.embedding_batch_size)  

    
    checkpoint_path = cache_path.with_suffix(".checkpoint.npz") if cache_path is not None else None
    all_embeddings: List[np.ndarray] = []
    start_batch = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        all_embeddings = [ckpt["embeddings"]]
        start_batch = int(ckpt["n_batches"])
        print(f"  [embed] Resuming from checkpoint: {start_batch}/{n_batches_total} batches already done")

    loader = create_dataloader(
        sorted_frame,
        tokenizer,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
        shuffle=False,
        code_column=config.code_column,
        label_column=config.label_column,
        source_id_column=config.source_id_column,
        project_column=config.project_column,
        num_workers=2,
    )

    encoder.eval()
    t0 = time.time()
    t_checkpoint = time.time()

    pbar = tqdm(total=n_batches_total, initial=start_batch, desc="  [embed] batches")

    for i, batch in enumerate(loader):
        if i < start_batch:
            continue

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(pooled.float().detach().cpu().numpy())
        pbar.update(1)

        
        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            partial = np.concatenate(all_embeddings, axis=0)
            np.savez(checkpoint_path, embeddings=partial, n_batches=i + 1)
            elapsed_min = (time.time() - t0) / 60
            rate = (i + 1 - start_batch) / max(time.time() - t_checkpoint, 1e-6)
            eta_min = (n_batches_total - (i + 1)) / max(rate, 1e-6) / 60
            print(
                f"  [embed] checkpoint @ batch {i+1}/{n_batches_total} | "
                f"elapsed {elapsed_min:.1f} min | ETA ~{eta_min:.1f} min | "
                f"rows so far {partial.shape[0]}"
            )

    pbar.close()

    embeddings_sorted = np.concatenate(all_embeddings, axis=0)
   
    embeddings = embeddings_sorted[inverse_order]

    total_min = (time.time() - t0) / 60
    print(f"  [embed] Extracted {embeddings.shape} in {total_min:.2f} min")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  [embed] Final cache saved, checkpoint removed: {cache_path}")

    return embeddings


# --------------------------------------------------------------------------- #
# Probe fitting helpers
# --------------------------------------------------------------------------- #

def _fit_probe(X: np.ndarray, y: np.ndarray, C: float, config: Exp3Config) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=C,
        max_iter=config.logistic_max_iter,
        solver=config.logistic_solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    clf.fit(X_s, y)
    return scaler, clf


def _predict_probe(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def _grouped_inner_folds(
    frame: pd.DataFrame,
    project_column: str,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Project-grouped inner K-fold split (shuffled, deterministic)."""
    shuffled = frame.sample(frac=1.0, random_state=random_state)
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_pos, val_pos in gkf.split(shuffled, groups=shuffled[project_column]):
        train_idx = shuffled.index.to_numpy()[train_pos]
        val_idx = shuffled.index.to_numpy()[val_pos]
        splits.append((train_idx, val_idx))
    return splits


# --------------------------------------------------------------------------- #
# Nested CV (development-only, no outer holdout touched)
# --------------------------------------------------------------------------- #

def run_exp3_nested_inner_profile(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    development_manifest: pd.DataFrame,
    outer_fold_id: int,
    base_config: Exp3Config,
    nested_config: NestedProbeConfig,
) -> Dict[str, Any]:
    """
    Profile a single outer-development fold: run the inner project-grouped C
    grid search only, without producing an OOF prediction.

    Includes gc.collect() after every (inner_fold, C) fit to avoid RAM
    accumulation across the 3 x 5 = 15 fits per outer fold.
    """
    import gc

    t0 = time.time()

    id_to_pos = {rid: pos for pos, rid in enumerate(development_frame[base_config.source_id_column].values)}

    outer_train_ids = set(
        development_manifest.loc[
            development_manifest[base_config.fold_column] != outer_fold_id, base_config.source_id_column
        ]
    )
    train_frame = development_frame[
        development_frame[base_config.source_id_column].isin(outer_train_ids)
    ].reset_index(drop=True)

    inner_splits = _grouped_inner_folds(
        train_frame, base_config.project_column, nested_config.inner_n_splits, nested_config.inner_random_state
    )

    rows = []
    for inner_id, (tr_idx, va_idx) in enumerate(inner_splits):
        tr_frame = train_frame.loc[tr_idx]
        va_frame = train_frame.loc[va_idx]
        tr_pos = [id_to_pos[rid] for rid in tr_frame[base_config.source_id_column].values]
        va_pos = [id_to_pos[rid] for rid in va_frame[base_config.source_id_column].values]

        X_tr = development_embeddings[tr_pos]
        y_tr = tr_frame[base_config.label_column].astype(int).values
        X_va = development_embeddings[va_pos]
        y_va = va_frame[base_config.label_column].astype(int).values

        for C in nested_config.C_grid:
            scaler, clf = _fit_probe(X_tr, y_tr, C, base_config)
            scores = _predict_probe(scaler, clf, X_va)
            ap = average_precision_score(y_va, scores) if len(np.unique(y_va)) > 1 else float("nan")
            rows.append({"inner_fold": inner_id, "C": C, "average_precision_pr_auc": ap, "n_val": len(y_va)})

            del scaler, clf, scores
            gc.collect()

        del tr_frame, va_frame, X_tr, X_va, y_tr, y_va
        gc.collect()

    alpha_summary = (
        pd.DataFrame(rows)
        .groupby("C", as_index=False)["average_precision_pr_auc"]
        .mean()
        .sort_values("average_precision_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    selected_C = float(alpha_summary.iloc[0]["C"])

    del train_frame
    gc.collect()

    return {
        "outer_fold_id": outer_fold_id,
        "selected_C": {"outer_fold_id": outer_fold_id, "selected_C": selected_C},
        "C_summary": alpha_summary,
        "inner_split_audit": pd.DataFrame(rows),
        "total_profile_seconds": time.time() - t0,
    }


def run_exp3_nested_probe(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    development_manifest: pd.DataFrame,
    base_config: Exp3Config,
    nested_config: NestedProbeConfig,
    output_dir: Path,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import gc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    checkpoint_path = output_dir / "exp3_nested_checkpoint.joblib"

    id_to_pos = {rid: pos for pos, rid in enumerate(development_frame[base_config.source_id_column].values)}
    fold_ids = sorted(development_manifest[base_config.fold_column].unique().tolist())

    oof_parts: List[pd.DataFrame] = []
    selected_alpha_rows: List[Dict[str, Any]] = []
    outer_training_audit: List[Dict[str, Any]] = []
    completed_folds: set = set()

    # --- Resume from checkpoint if present ---
    if checkpoint_path.exists():
        ckpt = joblib.load(checkpoint_path)
        oof_parts = ckpt["oof_parts"]
        selected_alpha_rows = ckpt["selected_alpha_rows"]
        outer_training_audit = ckpt["outer_training_audit"]
        completed_folds = ckpt["completed_folds"]
        if nested_config.verbose:
            print(f"  [nested] Resuming: {len(completed_folds)}/{len(fold_ids)} outer folds already completed")

    for outer_fold_id in fold_ids:
        if outer_fold_id in completed_folds:
            if nested_config.verbose:
                print(f"  [nested] Outer fold {outer_fold_id}: already completed, skipping.")
            continue

        fold_t0 = time.time()
        if nested_config.verbose:
            print(f"  [nested] Outer fold {outer_fold_id}: inner C grid search...")

        profile = run_exp3_nested_inner_profile(
            development_frame, development_embeddings, development_manifest,
            outer_fold_id, base_config, nested_config,
        )
        selected_C = profile["selected_C"]["selected_C"]
        selected_alpha_rows.append(profile["selected_C"])

        train_ids = set(
            development_manifest.loc[
                development_manifest[base_config.fold_column] != outer_fold_id, base_config.source_id_column
            ]
        )
        val_ids = set(
            development_manifest.loc[
                development_manifest[base_config.fold_column] == outer_fold_id, base_config.source_id_column
            ]
        )
        if train_ids.intersection(val_ids):
            raise RuntimeError(f"Outer fold {outer_fold_id}: train/val ID leakage detected.")

        train_frame = development_frame[development_frame[base_config.source_id_column].isin(train_ids)]
        val_frame = development_frame[development_frame[base_config.source_id_column].isin(val_ids)]

        tr_pos = [id_to_pos[rid] for rid in train_frame[base_config.source_id_column].values]
        va_pos = [id_to_pos[rid] for rid in val_frame[base_config.source_id_column].values]

        X_tr = development_embeddings[tr_pos]
        y_tr = train_frame[base_config.label_column].astype(int).values
        X_va = development_embeddings[va_pos]

        scaler, clf = _fit_probe(X_tr, y_tr, selected_C, base_config)
        val_scores = _predict_probe(scaler, clf, X_va)

        fold_oof = pd.DataFrame({
            base_config.source_id_column: val_frame[base_config.source_id_column].values,
            base_config.project_column: val_frame[base_config.project_column].values,
            "label": val_frame[base_config.label_column].astype(int).values,
            "y_score": val_scores,
            "fold": outer_fold_id,
        })
        oof_parts.append(fold_oof)

        outer_training_audit.append({
            "outer_fold_id": outer_fold_id,
            "selected_C": selected_C,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "train_projects": int(train_frame[base_config.project_column].nunique()),
            "val_projects": int(val_frame[base_config.project_column].nunique()),
        })

        completed_folds.add(outer_fold_id)

        # --- Free intermediate objects before checkpointing / next fold ---
        del train_frame, val_frame, X_tr, X_va, y_tr, scaler, clf, val_scores, profile
        gc.collect()

        # --- Checkpoint after every completed outer fold ---
        joblib.dump({
            "oof_parts": oof_parts,
            "selected_alpha_rows": selected_alpha_rows,
            "outer_training_audit": outer_training_audit,
            "completed_folds": completed_folds,
        }, checkpoint_path)

        if nested_config.verbose:
            fold_min = (time.time() - fold_t0) / 60
            total_min = (time.time() - t0) / 60
            print(
                f"  [nested] Outer fold {outer_fold_id} done in {fold_min:.1f} min "
                f"| total {total_min:.1f} min | checkpoint saved"
            )

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)

    eval_config = evaluation.EvaluationConfig(threshold=nested_config.decision_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    selected_alpha_df = pd.DataFrame(selected_alpha_rows)
    outer_training_df = pd.DataFrame(outer_training_audit)

    artifacts = {
        "oof_predictions": output_dir / "exp3_nested_oof_predictions.parquet",
        "selected_C_per_fold": output_dir / "exp3_selected_C_per_fold.csv",
        "outer_training_audit": output_dir / "exp3_outer_training_audit.csv",
        "run_metadata": output_dir / "exp3_nested_run_metadata.json",
    }
    oof_predictions.to_parquet(artifacts["oof_predictions"], index=False)
    selected_alpha_df.to_csv(artifacts["selected_C_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)

    metadata = {
        "base_config": asdict(base_config),
        "nested_config": {**asdict(nested_config), "C_grid": list(nested_config.C_grid)},
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # --- Clean up checkpoint on full success ---
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return {
        "oof_predictions": oof_predictions,
        "evaluation": eval_results,
        "selected_alpha": selected_alpha_df,
        "outer_fold_training": outer_training_df,
        "artifacts": artifacts,
    }


# --------------------------------------------------------------------------- #
# Canonical retraining + frozen outer-holdout scoring
# --------------------------------------------------------------------------- #

def run_exp3_canonical_retrain(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    selected_C: float,
    base_config: Exp3Config,
    output_dir: Path,
) -> Tuple[StandardScaler, LogisticRegression]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_dev = development_frame[base_config.label_column].astype(int).values
    scaler, clf = _fit_probe(development_embeddings, y_dev, selected_C, base_config)

    joblib.dump(clf, output_dir / "final_exp3_linear_probe_model.joblib")
    joblib.dump(scaler, output_dir / "final_exp3_scaler.joblib")

    return scaler, clf


def run_exp3_holdout_evaluation(
    holdout_frame: pd.DataFrame,
    holdout_embeddings: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    base_config: Exp3Config,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_holdout = holdout_frame[base_config.label_column].astype(int).values
    y_scores = _predict_probe(scaler, clf, holdout_embeddings)

    holdout_predictions = pd.DataFrame({
        base_config.source_id_column: holdout_frame[base_config.source_id_column].values,
        base_config.project_column: holdout_frame[base_config.project_column].values,
        "label": y_holdout,
        "y_score": y_scores,
        "fold": 0,
    })
    holdout_predictions.to_csv(output_dir / "exp3_holdout_predictions.csv", index=False)

    eval_config = evaluation.EvaluationConfig(threshold=base_config.decision_threshold, expected_n_folds=1)
    holdout_metrics = evaluation.evaluate_oof_predictions(holdout_predictions, config=eval_config)
    print(evaluation.format_metric_report(holdout_metrics["pooled_metrics"]))

    precision, recall, _ = precision_recall_curve(y_holdout, y_scores)
    ap = holdout_metrics["pooled_metrics"]["average_precision_pr_auc"]
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="b", label=f"EXP-3 Linear Probe (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Frozen Outer Holdout")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "exp3_outer_holdout_pr_curve.png")
    plt.close()

    y_pred = (y_scores >= base_config.decision_threshold).astype(int)
    cm = confusion_matrix(y_holdout, y_pred)
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
    plt.savefig(output_dir / "exp3_outer_holdout_confusion_matrix.png")
    plt.close()

    return {
        "holdout_predictions": holdout_predictions,
        "holdout_metrics": holdout_metrics,
        "y_pred": y_pred,
    }
