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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from case_study_2.data_loader import create_dataloader
from case_study_2.models import (
    DEFAULT_NEOBERT_MODEL,
    DEFAULT_NEOBERT_TOKENIZER,
    configure_huggingface_cache,
    load_code_tokenizer,
    load_code_encoder,
)
from case_study_1 import evaluation
from case_study_1 import split_manifest
from case_study_1.evaluation import EvaluationConfig
from case_study_1.confidence_intervals import bootstrap_metric_ci, format_ci_report


EXP6_VERSION = "cs2-exp6-neobert-linear-probe-v1"


@dataclass(frozen=True)
class Exp6Config:
    experiment_name: str = "cs2_exp6_neobert_linear_probe"

    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    hf_cache_dir: Optional[str] = None
    # NeoBERT supports up to 4096 tokens (RoPE/YaRE), unlike CodeBERTa's 512
    # positional-embedding ceiling. Kept at 512 by default for apples-to-apples
    # comparison with EXP-6/4/5/7 and because most DiverseVul functions are
    # under 512 tokens anyway (see project discussion) -- raise this
    # deliberately (and re-check VRAM/runtime) if you want to test whether
    # the longer context recovers anything on the minority of longer
    # functions that get truncated at 512.
    max_length: int = 512
    dtype_policy: str = "bfloat16"
    embedding_batch_size: int = 64
    pooling: str = "mean"
    trust_remote_code: bool = True  # NeoBERT ships as trust_remote_code on the Hub

    logistic_max_iter: int = 2000
    logistic_solver: str = "lbfgs"
    class_weight: str = "balanced"
    decision_threshold: float = 0.50

    n_splits: int = 5
    random_state: int = 42
    verbose: bool = True


@dataclass(frozen=True)
class NestedProbeConfig:
    experiment_name: str = "cs2_exp6_nested_probe_dev_grouped"
    C_grid: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    selection_metric: str = "average_precision_pr_auc"
    decision_threshold: float = 0.50
    tie_break_rule: str = "higher_C_then_grid_order"
    verbose: bool = True


def resolve_device(require_cuda: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if require_cuda:
        raise RuntimeError(
            "EXP-6 linear probe requires a CUDA GPU for embedding extraction."
        )
    return "cpu"


@torch.no_grad()
def extract_embeddings(
    encoder,
    tokenizer,
    frame: pd.DataFrame,
    config: Exp6Config,
    device: str,
    cache_path: Optional[Path] = None,
    checkpoint_every: int = 100,
) -> np.ndarray:
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
            # --- Guardrail (PDD sec. 5.2 / chandar-lab/NeoBERT GitHub Issue #11) ---
            # NeoBERT has a known numerical-instability bug that can produce
            # NaN/Inf hidden states under bf16 in some configurations. The
            # bug report is specifically about training, but a frozen
            # forward-only pass in bf16 is not immune in principle, and this
            # is cheap insurance: pooling arithmetic is done in fp32 (matching
            # the enforce_fp32_head safeguard already used for NeoBERT's
            # trainable head in CodeSequenceClassifier), and any NaN/Inf that
            # does slip through fails loudly here instead of silently
            # poisoning the linear probe's training data.
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
                f"rows in batch {i} (see PDD sec. 5.2 / chandar-lab/NeoBERT GitHub Issue #11). "
                f"This is the known NeoBERT numerical-instability bug, not a data problem -- "
                f"do not silently drop/zero these rows, investigate the encoder/dtype config first."
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


def _fit_probe(X: np.ndarray, y: np.ndarray, C: float, config: Exp6Config) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(copy=False)
    X_s = scaler.fit_transform(X.astype(np.float32, copy=False))
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
    return clf.predict_proba(scaler.transform(X.astype(np.float32, copy=False)))[:, 1]


def _grouped_inner_folds(
    frame: pd.DataFrame,
    source_id_column: str,
    label_column: str,
    project_column: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    inner_split_config = split_manifest.SplitConfig(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
        source_id_column=source_id_column,
        label_column=label_column,
        group_column=project_column,
    )
    return split_manifest.create_project_grouped_manifest(
        frame[[source_id_column, label_column, project_column]],
        config=inner_split_config,
    )


def run_exp6_nested_inner_profile(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    development_manifest: pd.DataFrame,
    outer_fold_id: int,
    base_config: Exp6Config,
    nested_config: NestedProbeConfig,
) -> Dict[str, Any]:
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

    inner_manifest = _grouped_inner_folds(
        train_frame, base_config.source_id_column, base_config.label_column,
        base_config.project_column, nested_config.inner_n_splits, nested_config.inner_random_state,
    )

    rows = []
    for inner_id in range(nested_config.inner_n_splits):
        tr_ids = set(inner_manifest.loc[inner_manifest["fold"] != inner_id, "source_row_id"])
        va_ids = set(inner_manifest.loc[inner_manifest["fold"] == inner_id, "source_row_id"])
        tr_frame = train_frame[train_frame[base_config.source_id_column].isin(tr_ids)]
        va_frame = train_frame[train_frame[base_config.source_id_column].isin(va_ids)]
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
        .sort_values(["average_precision_pr_auc", "C"], ascending=[False, False])
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


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> Dict[str, Path]:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "selected": root / f"{prefix}_selected_C.json",
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


def run_exp6_nested_probe(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    development_manifest: pd.DataFrame,
    base_config: Exp6Config,
    nested_config: NestedProbeConfig,
    output_dir: Path,
    additional_metadata: Optional[Dict[str, Any]] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "exp6_nested_run_state.json"
    t0 = time.time()

    id_to_pos = {rid: pos for pos, rid in enumerate(development_frame[base_config.source_id_column].values)}
    fold_ids = sorted(development_manifest[base_config.fold_column].unique().tolist())

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
            print(f"  [nested] Outer fold {outer_fold_id}: inner C grid search...")

        profile = run_exp6_nested_inner_profile(
            development_frame, development_embeddings, development_manifest,
            outer_fold_id, base_config, nested_config,
        )
        selected_C = profile["selected_C"]["selected_C"]

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

        selected_row = {"outer_fold_id": outer_fold_id, "selected_C": selected_C}
        training_row = {
            "outer_fold_id": outer_fold_id,
            "selected_C": selected_C,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "train_projects": int(train_frame[base_config.project_column].nunique()),
            "val_projects": int(val_frame[base_config.project_column].nunique()),
        }

        _write_outer_checkpoint(output_dir, outer_fold_id, fold_oof, selected_row, training_row)

        oof_parts.append(fold_oof)
        selected_rows.append(selected_row)
        outer_training_rows.append(training_row)
        completed_folds.append(outer_fold_id)

        _update_run_state(state_path, completed_folds, status="running")

        del train_frame, val_frame, X_tr, X_va, y_tr, scaler, clf, val_scores, profile
        gc.collect()

    oof_predictions = pd.concat(oof_parts, axis=0).reset_index(drop=True)

    eval_config = EvaluationConfig(threshold=nested_config.decision_threshold, expected_n_folds=len(fold_ids))
    eval_results = evaluation.evaluate_oof_predictions(oof_predictions, config=eval_config)

    selected_df = pd.DataFrame(selected_rows)
    outer_training_df = pd.DataFrame(outer_training_rows)

    artifacts = {
        "oof_predictions": output_dir / "exp6_nested_oof_predictions.parquet",
        "selected_C_per_fold": output_dir / "exp6_selected_C_per_fold.csv",
        "outer_training_audit": output_dir / "exp6_outer_training_audit.csv",
        "run_metadata": output_dir / "exp6_nested_run_metadata.json",
    }
    oof_predictions.to_parquet(artifacts["oof_predictions"], index=False)
    selected_df.to_csv(artifacts["selected_C_per_fold"], index=False)
    outer_training_df.to_csv(artifacts["outer_training_audit"], index=False)

    metadata = {
        "exp6_version": EXP6_VERSION,
        "base_config": asdict(base_config),
        "nested_config": {**asdict(nested_config), "C_grid": list(nested_config.C_grid)},
        "runtime_seconds": time.time() - t0,
        **(additional_metadata or {}),
    }
    with open(artifacts["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    _update_run_state(state_path, completed_folds, status="completed")

    return {
        "oof_predictions": oof_predictions,
        "evaluation": eval_results,
        "selected_C": selected_df,
        "outer_fold_training": outer_training_df,
        "artifacts": artifacts,
    }


def run_exp6_canonical_retrain(
    development_frame: pd.DataFrame,
    development_embeddings: np.ndarray,
    selected_C: float,
    base_config: Exp6Config,
    output_dir: Path,
) -> Tuple[StandardScaler, LogisticRegression]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_dev = development_frame[base_config.label_column].astype(int).values
    scaler, clf = _fit_probe(development_embeddings, y_dev, selected_C, base_config)

    joblib.dump(clf, output_dir / "final_exp6_linear_probe_model.joblib")
    joblib.dump(scaler, output_dir / "final_exp6_scaler.joblib")

    return scaler, clf


def run_exp6_holdout_evaluation(
    holdout_frame: pd.DataFrame,
    holdout_embeddings: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    base_config: Exp6Config,
    output_dir: Path,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_holdout = holdout_frame[base_config.label_column].astype(int).values
    y_scores = _predict_probe(scaler, clf, holdout_embeddings)

    holdout_predictions = pd.DataFrame({
        base_config.source_id_column: holdout_frame[base_config.source_id_column].values,
        "project": holdout_frame[base_config.project_column].values,
        "label": y_holdout,
        "y_score": y_scores,
        "fold": 0,
    })
    holdout_predictions.to_csv(output_dir / "exp6_holdout_predictions.csv", index=False)

    eval_config = EvaluationConfig(threshold=base_config.decision_threshold, expected_n_folds=1)
    holdout_metrics = evaluation.evaluate_oof_predictions(holdout_predictions, config=eval_config)
    print(evaluation.format_metric_report(holdout_metrics["pooled_metrics"]))

    ci_result = bootstrap_metric_ci(
        holdout_predictions,
        metric="average_precision_pr_auc",
        group_column="project",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=base_config.random_state,
    )
    with open(output_dir / "exp6_holdout_pr_auc_bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci_result.as_dict(), f, indent=2)
    print(format_ci_report(ci_result))

    precision, recall, _ = precision_recall_curve(y_holdout, y_scores)
    ap = holdout_metrics["pooled_metrics"]["average_precision_pr_auc"]
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="b", label=f"EXP-6 Linear Probe (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Frozen Outer Holdout")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "exp6_outer_holdout_pr_curve.png")
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
    plt.savefig(output_dir / "exp6_outer_holdout_confusion_matrix.png")
    plt.close()

    return {
        "holdout_predictions": holdout_predictions,
        "holdout_metrics": holdout_metrics,
        "bootstrap_ci": ci_result.as_dict(),
        "y_pred": y_pred,
    }