from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from case_study_2.data_loader import create_dataloader
from case_study_2.models import DEFAULT_NEOBERT_MODEL, DEFAULT_NEOBERT_TOKENIZER
from utils import evaluation
from utils import split_manifest
from utils.evaluation import EvaluationConfig, select_f1_threshold


EXP3_VERSION = "cs2-exp3-neobert-linear-probe-v2-nested-rotating-5fold"


@dataclass(frozen=True)
class Exp3Config:
    """Declared reproducible configuration for CS2-EXP3 (frozen NeoBERT embeddings + logistic probe)."""

    experiment_name: str = "cs2_exp3_neobert_linear_probe"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    hf_cache_dir: Optional[str] = None
    # NeoBERT supports up to 4096 tokens; kept at 512 since most functions fit.
    max_length: int = 512
    dtype_policy: str = "bfloat16"
    embedding_batch_size: int = 64
    pooling: str = "mean"
    trust_remote_code: bool = True  # NeoBERT ships as trust_remote_code on the Hub

    logistic_max_iter: int = 2000
    logistic_solver: str = "lbfgs"
    class_weight: str = "balanced"

    n_splits: int = 5
    random_state: int = 42
    verbose: bool = True


@dataclass(frozen=True)
class NestedProbeConfig:
    """Inner-CV search configuration: C and decision threshold, selected per outer fold."""

    C_grid: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    selection_metric: str = "average_precision_pr_auc"
    verbose: bool = True


def resolve_device(require_cuda: bool = True) -> str:
    """Pick the GPU when available; raise if a GPU is required but absent."""
    if torch.cuda.is_available():
        return "cuda:0"
    if require_cuda:
        raise RuntimeError("EXP-3 linear probe requires a CUDA GPU for embedding extraction.")
    return "cpu"


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
    """Extract frozen mean/CLS-pooled NeoBERT embeddings for every row, once for the whole dataset."""
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    if not str(device).startswith("cuda"):
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

    for i, batch in enumerate(loader):
        if i < start_batch:
            continue

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Pooling in fp32 guards against the known NeoBERT bf16 NaN/Inf
            # instability (chandar-lab/NeoBERT GitHub Issue #11).
            last_hidden = outputs.last_hidden_state.float()
            if config.pooling == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom

        if not torch.isfinite(pooled).all():
            n_bad = (~torch.isfinite(pooled)).any(dim=-1).sum().item()
            raise RuntimeError(
                f"[embed] NaN/Inf detected in pooled embeddings for {n_bad}/{pooled.shape[0]} "
                f"rows in batch {i} (known NeoBERT numerical instability, not a data problem)."
            )

        all_embeddings.append(pooled.detach().cpu().numpy())

        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            partial = np.concatenate(all_embeddings, axis=0)
            np.savez(checkpoint_path, embeddings=partial, n_batches=i + 1)
            if config.verbose:
                elapsed_min = (time.time() - t0) / 60
                print(f"  [embed] checkpoint @ batch {i+1}/{n_batches_total} | elapsed {elapsed_min:.1f} min")

    embeddings_sorted = np.concatenate(all_embeddings, axis=0)
    embeddings = embeddings_sorted[inverse_order]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()

    return embeddings


def _fit_probe(X: np.ndarray, y: np.ndarray, C: float, config: Exp3Config) -> Tuple[StandardScaler, LogisticRegression]:
    """Fit a StandardScaler + L2 logistic regression probe at one C value."""
    scaler = StandardScaler(copy=False)
    X_s = scaler.fit_transform(X.astype(np.float32, copy=False))
    clf = LogisticRegression(
        C=C, max_iter=config.logistic_max_iter, solver=config.logistic_solver,
        class_weight=config.class_weight, random_state=config.random_state,
    )
    clf.fit(X_s, y)
    return scaler, clf


def _predict_probe(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Score embeddings with an already-fit scaler + probe."""
    return clf.predict_proba(scaler.transform(X.astype(np.float32, copy=False)))[:, 1]


def _inner_manifest(frame: pd.DataFrame, config: Exp3Config, nested_config: NestedProbeConfig, outer_fold_id: int) -> pd.DataFrame:
    """Project-grouped, stratified inner-fold assignment for one outer fold's training rows."""
    inner_split_config = split_manifest.SplitConfig(
        n_splits=nested_config.inner_n_splits,
        random_state=nested_config.inner_random_state + int(outer_fold_id),
        source_id_column=config.source_id_column,
        label_column=config.label_column,
        group_column=config.project_column,
    )
    return split_manifest.create_project_grouped_manifest(
        frame[[config.source_id_column, config.label_column, config.project_column]], config=inner_split_config
    )


def run_exp3_nested_inner_profile(
    dataset_frame: pd.DataFrame,
    dataset_embeddings: np.ndarray,
    manifest: pd.DataFrame,
    outer_fold_id: int,
    base_config: Exp3Config,
    nested_config: NestedProbeConfig,
) -> Dict[str, Any]:
    """Select C and the decision threshold via 3-fold inner CV (micro-averaged/pooled PR-AUC) on this fold's training rows."""
    t0 = time.time()
    id_to_pos = {rid: pos for pos, rid in enumerate(dataset_frame[base_config.source_id_column].values)}

    outer_train_ids = set(
        manifest.loc[manifest[base_config.fold_column] != outer_fold_id, base_config.source_id_column]
    )
    train_frame = dataset_frame[dataset_frame[base_config.source_id_column].isin(outer_train_ids)].reset_index(drop=True)

    inner_manifest_df = _inner_manifest(train_frame, base_config, nested_config, outer_fold_id)

    rows = []
    predictions_by_C: Dict[float, List[pd.DataFrame]] = {C: [] for C in nested_config.C_grid}
    for inner_id in range(nested_config.inner_n_splits):
        tr_ids = set(inner_manifest_df.loc[inner_manifest_df["fold"] != inner_id, "source_row_id"])
        va_ids = set(inner_manifest_df.loc[inner_manifest_df["fold"] == inner_id, "source_row_id"])
        tr_frame = train_frame[train_frame[base_config.source_id_column].isin(tr_ids)]
        va_frame = train_frame[train_frame[base_config.source_id_column].isin(va_ids)]
        tr_pos = [id_to_pos[rid] for rid in tr_frame[base_config.source_id_column].values]
        va_pos = [id_to_pos[rid] for rid in va_frame[base_config.source_id_column].values]

        X_tr = dataset_embeddings[tr_pos]
        y_tr = tr_frame[base_config.label_column].astype(int).values
        X_va = dataset_embeddings[va_pos]
        y_va = va_frame[base_config.label_column].astype(int).values

        for C in nested_config.C_grid:
            scaler, clf = _fit_probe(X_tr, y_tr, C, base_config)
            scores = _predict_probe(scaler, clf, X_va)
            predictions_by_C[C].append(
                pd.DataFrame({"source_row_id": va_frame[base_config.source_id_column].values, "label": y_va, "y_score": scores})
            )
            del scaler, clf, scores
            gc.collect()

        del tr_frame, va_frame, X_tr, X_va, y_tr, y_va
        gc.collect()

    # Micro-averaged (pooled) selection: concatenate every inner fold's held-out
    # predictions for a given C and score once, instead of averaging separately
    # computed per-fold scores (macro-average). PR-AUC/ROC-AUC are rank-based and
    # not additive across folds, so macro-averaging them is a biased/noisy proxy
    # when positives are rare; pooling first is the standard fix (as in sklearn's
    # own average_precision_score computed over a full held-out set).
    for C in nested_config.C_grid:
        pooled = pd.concat(predictions_by_C[C], ignore_index=True)
        pooled_pr_auc = float(average_precision_score(pooled["label"], pooled["y_score"]))
        rows.append({"C": C, "inner_pooled_pr_auc": pooled_pr_auc, "n_val": len(pooled)})

    C_summary = (
        pd.DataFrame(rows)
        .sort_values(["inner_pooled_pr_auc", "C"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    selected_C = float(C_summary.iloc[0]["C"])
    pooled_predictions = pd.concat(predictions_by_C[selected_C], ignore_index=True)
    selected_threshold, threshold_metrics = select_f1_threshold(pooled_predictions["label"], pooled_predictions["y_score"])

    del train_frame
    gc.collect()

    return {
        "outer_fold_id": outer_fold_id,
        "selected": {
            "outer_fold_id": outer_fold_id,
            "selected_C": selected_C,
            "decision_threshold": selected_threshold,
            "inner_pooled_pr_auc": float(C_summary.iloc[0]["inner_pooled_pr_auc"]),
            "inner_validation_f1": float(threshold_metrics["f1"]),
        },
        "C_summary": C_summary,
        "total_profile_seconds": time.time() - t0,
    }


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> Dict[str, Path]:
    """Filesystem locations for one outer fold's resumable checkpoint."""
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "selected": root / f"{prefix}_selected.json",
        "training": root / f"{prefix}_outer_training.json",
    }


def _write_outer_checkpoint(output_dir: Path, outer_fold_id: int, predictions: pd.DataFrame, selected: dict, training: dict) -> None:
    """Persist one outer fold's result so a later run can resume without refitting."""
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    predictions.to_parquet(paths["predictions"], index=False)
    with paths["selected"].open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, default=str)
    with paths["training"].open("w", encoding="utf-8") as f:
        json.dump(training, f, indent=2, default=str)


def _load_outer_checkpoint(output_dir: Path, outer_fold_id: int) -> Optional[dict]:
    """Load one outer fold's checkpoint if it exists and is complete."""
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    if not all(p.exists() for p in paths.values()):
        return None
    with paths["selected"].open("r", encoding="utf-8") as f:
        selected = json.load(f)
    with paths["training"].open("r", encoding="utf-8") as f:
        training = json.load(f)
    return {"predictions": pd.read_parquet(paths["predictions"]), "selected": selected, "training": training}


def _update_run_state(state_path: Path, completed_folds, status: str) -> None:
    """Persist which outer folds are done, for resumability and progress inspection."""
    state = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_outer_folds": sorted(int(f) for f in completed_folds),
    }
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_exp3_nested_probe(
    dataset_frame: pd.DataFrame,
    dataset_embeddings: np.ndarray,
    manifest: pd.DataFrame,
    base_config: Exp3Config,
    nested_config: NestedProbeConfig,
    output_dir: Path,
    additional_metadata: Optional[Dict[str, Any]] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run the official rotating 5-fold CS2-EXP3 experiment, with a 3-fold inner-CV search per outer fold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "exp3_nested_run_state.json"
    t0 = time.time()

    id_to_pos = {rid: pos for pos, rid in enumerate(dataset_frame[base_config.source_id_column].values)}
    fold_ids = sorted(manifest[base_config.fold_column].unique().tolist())

    oof_parts = []
    selected_rows = []
    outer_training_rows = []
    completed_folds = []

    for outer_fold_id in fold_ids:
        checkpoint = _load_outer_checkpoint(output_dir, outer_fold_id) if resume else None
        if checkpoint is not None:
            oof_parts.append(checkpoint["predictions"])
            selected_rows.append(checkpoint["selected"])
            outer_training_rows.append(checkpoint["training"])
            completed_folds.append(outer_fold_id)
            if nested_config.verbose:
                print(f"  [nested] Outer fold {outer_fold_id}: loaded from checkpoint.")
            continue

        if nested_config.verbose:
            print(f"  [nested] Outer fold {outer_fold_id}: inner C + threshold search...")

        profile = run_exp3_nested_inner_profile(dataset_frame, dataset_embeddings, manifest, outer_fold_id, base_config, nested_config)
        selected = profile["selected"]

        train_ids = set(manifest.loc[manifest[base_config.fold_column] != outer_fold_id, base_config.source_id_column])
        val_ids = set(manifest.loc[manifest[base_config.fold_column] == outer_fold_id, base_config.source_id_column])
        if train_ids.intersection(val_ids):
            raise RuntimeError(f"Outer fold {outer_fold_id}: train/val ID leakage detected.")

        train_frame = dataset_frame[dataset_frame[base_config.source_id_column].isin(train_ids)]
        val_frame = dataset_frame[dataset_frame[base_config.source_id_column].isin(val_ids)]

        tr_pos = [id_to_pos[rid] for rid in train_frame[base_config.source_id_column].values]
        va_pos = [id_to_pos[rid] for rid in val_frame[base_config.source_id_column].values]

        X_tr = dataset_embeddings[tr_pos]
        y_tr = train_frame[base_config.label_column].astype(int).values
        X_va = dataset_embeddings[va_pos]

        scaler, clf = _fit_probe(X_tr, y_tr, selected["selected_C"], base_config)
        val_scores = _predict_probe(scaler, clf, X_va)

        fold_oof = pd.DataFrame({
            base_config.source_id_column: val_frame[base_config.source_id_column].values,
            base_config.project_column: val_frame[base_config.project_column].values,
            "label": val_frame[base_config.label_column].astype(int).values,
            "y_score": val_scores,
            "fold": outer_fold_id,
        })

        training_row = {
            **selected,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "train_projects": int(train_frame[base_config.project_column].nunique()),
            "val_projects": int(val_frame[base_config.project_column].nunique()),
        }

        _write_outer_checkpoint(output_dir, outer_fold_id, fold_oof, selected, training_row)

        oof_parts.append(fold_oof)
        selected_rows.append(selected)
        outer_training_rows.append(training_row)
        completed_folds.append(outer_fold_id)

        _update_run_state(state_path, completed_folds, status="running")

        del train_frame, val_frame, X_tr, X_va, y_tr, scaler, clf, val_scores, profile
        gc.collect()

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows)
    outer_training_df = pd.DataFrame(outer_training_rows)

    mean_threshold = float(selected_df["decision_threshold"].mean())
    eval_config = EvaluationConfig(threshold=mean_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    artifacts = {
        "oof_predictions": output_dir / "exp3_nested_oof_predictions.parquet",
        "selected_per_fold": output_dir / "exp3_selected_per_outer_fold.csv",
        "outer_training_audit": output_dir / "exp3_outer_training_audit.csv",
        "run_metadata": output_dir / "exp3_nested_run_metadata.json",
    }
    eval_results["predictions"].to_parquet(artifacts["oof_predictions"], index=False)
    selected_df.to_csv(artifacts["selected_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)

    metadata = {
        "exp3_version": EXP3_VERSION,
        "base_config": asdict(base_config),
        "nested_config": {**asdict(nested_config), "C_grid": list(nested_config.C_grid)},
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    _update_run_state(state_path, completed_folds, status="completed")

    return {
        "oof_predictions": eval_results["predictions"],
        "evaluation": eval_results,
        "selected": selected_df,
        "outer_fold_training": outer_training_df,
        "artifacts": artifacts,
    }


__all__ = [
    "EXP3_VERSION",
    "Exp3Config",
    "NestedProbeConfig",
    "extract_embeddings",
    "resolve_device",
    "run_exp3_nested_inner_profile",
    "run_exp3_nested_probe",
]
