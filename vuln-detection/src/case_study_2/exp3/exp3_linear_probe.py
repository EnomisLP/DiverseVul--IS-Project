"""Case Study 2 - EXP-3 Linear Probe.

Goal
----
Evaluate frozen NeoBERT representations for C/C++ vulnerability detection.
This experiment is part of a comparative study, not a model-selection-only
pipeline.  Therefore it reports both:

1. Development project-disjoint performance.
2. Frozen outer-holdout performance from a canonical model trained on the
   development partition.

Important methodological note
-----------------------------
This module can automatically score the frozen outer holdout because the project
is a comparative analysis.  To keep the interpretation honest, the report should
state that all model definitions and hyperparameters are fixed before looking at
holdout results, and that holdout results are used for comparison/reporting, not
for repeatedly redesigning the experiments.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from case_study_2.data_loader import (
    DatasetColumns,
    create_dataloader,
    make_project_disjoint_threshold_split,
    prepare_code_frame,
    sample_with_optional_positive_fraction,
)
from case_study_2.models import (
    DEFAULT_NEOBERT_MODEL,
    DEFAULT_NEOBERT_TOKENIZER,
    configure_huggingface_cache,
    load_neobert_encoder,
    load_neobert_tokenizer,
)


@dataclass
class Exp3LinearProbeConfig:
    """Explicit configuration for EXP-3 Linear Probe."""

    # Shared Drive data root.  This is the same root that contains processed
    # dataset files, manifests, and outputs.
    data_root: str = "/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData"

    # We reuse the same frozen project-disjoint split protocol as CS1 because
    # CS2 uses the same C/C++ vulnerability dataset and labels.
    split_id: str = "cs1_project_holdout20_innercv_v1"

    # Input representation.  Start with normalized_code for the main CS2
    # transformer baseline.  A later supplementary run can use abstracted_code_v1.
    data_filename: str = "rdiversevul_cs1_normalized_v1.parquet"
    code_column: str = "normalized_code"
    label_column: str = "label"
    source_id_column: str = "source_row_id"
    project_column: str = "project"

    # NeoBERT/tokenization.
    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    max_length: int = 512
    dtype_policy: str = "float16"
    embedding_batch_size: int = 4
    hf_cache_dir: str = "/content/hf_cache"

    # Run mode.  smoke/pilot are sanity checks and not reportable as final
    # results.  one_fold and full_cv are development-comparison runs.
    mode: str = "smoke"  # smoke, pilot, one_fold, full_cv
    selected_fold: int = 0

    # Smoke/pilot sampling.
    smoke_train_rows: int = 512
    smoke_threshold_rows: int = 256
    smoke_valid_rows: int = 256
    smoke_holdout_rows: int = 256

    pilot_train_rows: int = 8192
    pilot_threshold_rows: int = 2048
    pilot_valid_rows: int = 4096
    pilot_holdout_rows: int = 4096
    pilot_positive_fraction: Optional[float] = 0.25

    # Threshold selection inside development only.
    threshold_validation_fraction: float = 0.20
    threshold_metric: str = "f1"

    # Linear probe.  class_weight='balanced' is appropriate for the strong class
    # imbalance in RDiverseVul.
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = "lbfgs"
    class_weight: Optional[str] = "balanced"

    # Confidence intervals.
    bootstrap_resamples: int = 1000
    bootstrap_random_state: int = 20260707

    # Output organization.  CS2 outputs are intentionally separate from CS1
    # outputs even though the split_id name is reused for manifests.
    output_subdir: str = "exp3_linear_probe_dev_and_holdout_v1"

    # Comparative-analysis setting: score outer holdout automatically after the
    # development evaluation.  Keep this true only when experiment definitions
    # are fixed and not being changed adaptively from holdout feedback.
    evaluate_outer_holdout: bool = True

    # Reproducibility.
    random_state: int = 42


def build_default_paths(config: Exp3LinearProbeConfig) -> Dict[str, Path]:
    data_root = Path(config.data_root)
    paths = {
        "data_root": data_root,
        "data": data_root / "processed" / config.data_filename,
        "outer_manifest": data_root / "manifests" / config.split_id / "outer_holdout" / "cs1_outer_project_holdout_manifest.parquet",
        "inner_manifest": data_root / "manifests" / config.split_id / "inner_cv" / "cs1_project_grouped_5fold_manifest.parquet",
        "output_root": data_root / "outputs" / "case_study_2" / config.output_subdir,
    }
    paths["run_output_dir"] = paths["output_root"] / config.mode
    paths["embedding_cache_dir"] = paths["run_output_dir"] / "embedding_cache"
    return paths


def audit_paths(paths: Dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": name, "path": str(path), "exists": path.exists()}
            for name, path in paths.items()
        ]
    )


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "f1",
) -> Tuple[float, pd.DataFrame]:
    """Select a threshold using labels from development data only."""

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        row = {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
            "predicted_positive_rate": float(y_pred.mean()),
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    metric = str(metric).lower()
    if metric not in table.columns:
        raise ValueError(f"Unknown threshold metric: {metric}")
    best_idx = table[metric].astype(float).idxmax()
    return float(table.loc[best_idx, "threshold"]), table


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "threshold": float(threshold),
        "pr_auc": _safe_average_precision(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "specificity": float(specificity),
        "npv": float(npv),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else float("nan"),
    }


def project_block_bootstrap_pr_auc_ci(
    predictions: pd.DataFrame,
    *,
    label_column: str = "label",
    score_column: str = "y_score",
    project_column: str = "project",
    n_resamples: int = 1000,
    random_state: int = 20260707,
) -> Dict[str, float]:
    """Project-block bootstrap CI for PR-AUC."""

    rng = np.random.default_rng(random_state)
    projects = np.array(sorted(predictions[project_column].astype(str).unique()))
    if len(projects) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "valid_resamples": 0, "n_projects": 0}

    by_project = {p: predictions[predictions[project_column].astype(str) == p] for p in projects}
    values: List[float] = []
    for _ in range(int(n_resamples)):
        sampled_projects = rng.choice(projects, size=len(projects), replace=True)
        sample = pd.concat([by_project[p] for p in sampled_projects], ignore_index=True)
        y_true = sample[label_column].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_score = sample[score_column].astype(float).to_numpy()
        values.append(float(average_precision_score(y_true, y_score)))

    if not values:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "valid_resamples": 0, "n_projects": int(len(projects))}

    return {
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "valid_resamples": int(len(values)),
        "n_projects": int(len(projects)),
    }


def plot_precision_recall(predictions: pd.DataFrame, output_path: Path, *, title: str) -> None:
    y_true = predictions["label"].astype(int).to_numpy()
    y_score = predictions["y_score"].astype(float).to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    baseline = float(y_true.mean()) if len(y_true) else float("nan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.axhline(baseline, linestyle="--", label=f"baseline={baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion(metrics: Dict[str, float], output_path: Path, *, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _load_frames(config: Exp3LinearProbeConfig, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for key in ["data", "outer_manifest", "inner_manifest"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required file for {key}: {paths[key]}")

    full_df = pd.read_parquet(paths["data"])
    outer_manifest = pd.read_parquet(paths["outer_manifest"])
    inner_manifest = pd.read_parquet(paths["inner_manifest"])

    columns = DatasetColumns(
        code=config.code_column,
        label=config.label_column,
        source_id=config.source_id_column,
        project=config.project_column,
    )
    full_df = prepare_code_frame(full_df, columns)

    dev_ids = set(outer_manifest.loc[outer_manifest["partition"] == "development", config.source_id_column].tolist())
    holdout_ids = set(outer_manifest.loc[outer_manifest["partition"] == "outer_holdout", config.source_id_column].tolist())

    dev_df = full_df[full_df[config.source_id_column].isin(dev_ids)].copy().reset_index(drop=True)
    holdout_df = full_df[full_df[config.source_id_column].isin(holdout_ids)].copy().reset_index(drop=True)

    if dev_df.empty:
        raise ValueError("Development frame is empty. Check outer manifest partition names.")
    if holdout_df.empty:
        raise ValueError("Outer holdout frame is empty. Check outer manifest partition names.")

    return full_df, dev_df, holdout_df, inner_manifest


def _frame_summary(frame: pd.DataFrame, config: Exp3LinearProbeConfig) -> dict:
    labels = frame[config.label_column].astype(int)
    return {
        "rows": int(len(frame)),
        "projects": int(frame[config.project_column].nunique()),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(frame) else float("nan"),
    }


def extract_embeddings(
    frame: pd.DataFrame,
    *,
    encoder,
    tokenizer,
    config: Exp3LinearProbeConfig,
    device: torch.device,
    description: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract CLS embeddings from frozen NeoBERT."""

    columns = DatasetColumns(
        code=config.code_column,
        label=config.label_column,
        source_id=config.source_id_column,
        project=config.project_column,
    )
    loader = create_dataloader(
        frame,
        tokenizer,
        columns=columns,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()

            all_embeddings.append(cls_embeddings)
            all_labels.append(batch["labels"].detach().cpu().numpy().astype(int))
            all_meta.append(
                pd.DataFrame(
                    {
                        "source_row_id": batch["source_row_id"].detach().cpu().numpy().astype(int),
                        "project": batch["project"],
                    }
                )
            )

    X = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=int)
    meta = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame(columns=["source_row_id", "project"])
    return X, y, meta


def _fit_linear_probe(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    config: Exp3LinearProbeConfig,
) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)

    classifier = LogisticRegression(
        C=float(config.logistic_C),
        max_iter=int(config.logistic_max_iter),
        solver=config.logistic_solver,
        class_weight=config.class_weight,
        random_state=int(config.random_state),
    )
    classifier.fit(X_fit_scaled, y_fit)
    return scaler, classifier


def _predict_linear_probe(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return classifier.predict_proba(X_scaled)[:, 1]


def _run_single_fold(
    *,
    fold_id: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    encoder,
    tokenizer,
    config: Exp3LinearProbeConfig,
    device: torch.device,
    output_dir: Path,
    prefix: str,
) -> Tuple[pd.DataFrame, dict, float, pd.DataFrame]:
    fit_df, threshold_df = make_project_disjoint_threshold_split(
        train_df,
        project_column=config.project_column,
        validation_fraction=config.threshold_validation_fraction,
        random_state=config.random_state + fold_id,
    )

    X_fit, y_fit, _ = extract_embeddings(
        fit_df,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        device=device,
        description=f"{prefix} fold {fold_id} fit embeddings",
    )
    X_thr, y_thr, _ = extract_embeddings(
        threshold_df,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        device=device,
        description=f"{prefix} fold {fold_id} threshold embeddings",
    )
    X_val, y_val, meta_val = extract_embeddings(
        valid_df,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        device=device,
        description=f"{prefix} fold {fold_id} valid embeddings",
    )

    scaler, classifier = _fit_linear_probe(X_fit, y_fit, config=config)
    threshold_scores = _predict_linear_probe(scaler, classifier, X_thr)
    selected_threshold, threshold_table = select_threshold(
        y_thr,
        threshold_scores,
        metric=config.threshold_metric,
    )

    valid_scores = _predict_linear_probe(scaler, classifier, X_val)
    predictions = meta_val.copy()
    predictions["label"] = y_val.astype(int)
    predictions["y_score"] = valid_scores.astype(float)
    predictions["fold"] = int(fold_id)

    metrics = compute_metrics(y_val, valid_scores, threshold=selected_threshold)
    metrics.update(
        {
            "fold": int(fold_id),
            "fit_rows": int(len(fit_df)),
            "threshold_rows": int(len(threshold_df)),
            "valid_rows": int(len(valid_df)),
            "selected_threshold": float(selected_threshold),
        }
    )

    threshold_table["fold"] = int(fold_id)
    return predictions, metrics, selected_threshold, threshold_table


def _make_mode_frames(
    config: Exp3LinearProbeConfig,
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    inner_manifest: pd.DataFrame,
) -> Tuple[List[int], pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Prepare development/holdout frames according to run mode.

    Returns:
      fold_ids, dev_frame_for_cv, holdout_frame_for_eval, explicit_train, explicit_valid

    explicit_train/valid are used for smoke/pilot one-fold style runs.
    full_cv uses the full manifest and returns explicit_train/valid=None.
    """

    mode = config.mode.lower()
    if mode not in {"smoke", "pilot", "one_fold", "full_cv"}:
        raise ValueError("mode must be one of: smoke, pilot, one_fold, full_cv")

    if mode == "full_cv":
        return [0, 1, 2, 3, 4], dev_df, holdout_df, None, None

    fold = int(config.selected_fold)
    train_ids = set(inner_manifest.loc[inner_manifest["fold"] != fold, config.source_id_column])
    valid_ids = set(inner_manifest.loc[inner_manifest["fold"] == fold, config.source_id_column])
    train_df = dev_df[dev_df[config.source_id_column].isin(train_ids)].copy().reset_index(drop=True)
    valid_df = dev_df[dev_df[config.source_id_column].isin(valid_ids)].copy().reset_index(drop=True)

    if mode == "one_fold":
        return [fold], dev_df, holdout_df, train_df, valid_df

    if mode == "smoke":
        train_df = sample_with_optional_positive_fraction(
            train_df,
            n_rows=config.smoke_train_rows,
            label_column=config.label_column,
            positive_fraction=0.25,
            random_state=config.random_state,
        )
        valid_df = sample_with_optional_positive_fraction(
            valid_df,
            n_rows=config.smoke_valid_rows,
            label_column=config.label_column,
            positive_fraction=0.25,
            random_state=config.random_state + 1,
        )
        holdout_eval_df = sample_with_optional_positive_fraction(
            holdout_df,
            n_rows=config.smoke_holdout_rows,
            label_column=config.label_column,
            positive_fraction=0.25,
            random_state=config.random_state + 2,
        )
        return [fold], dev_df, holdout_eval_df, train_df, valid_df

    # pilot
    train_df = sample_with_optional_positive_fraction(
        train_df,
        n_rows=config.pilot_train_rows,
        label_column=config.label_column,
        positive_fraction=config.pilot_positive_fraction,
        random_state=config.random_state,
    )
    valid_df = sample_with_optional_positive_fraction(
        valid_df,
        n_rows=config.pilot_valid_rows,
        label_column=config.label_column,
        positive_fraction=None,
        random_state=config.random_state + 1,
    )
    holdout_eval_df = sample_with_optional_positive_fraction(
        holdout_df,
        n_rows=config.pilot_holdout_rows,
        label_column=config.label_column,
        positive_fraction=None,
        random_state=config.random_state + 2,
    )
    return [fold], dev_df, holdout_eval_df, train_df, valid_df


def _canonical_holdout_evaluation(
    *,
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    encoder,
    tokenizer,
    config: Exp3LinearProbeConfig,
    device: torch.device,
    output_dir: Path,
    threshold: float,
) -> Tuple[pd.DataFrame, dict]:
    """Train canonical linear probe on development and score outer holdout."""

    X_dev, y_dev, _ = extract_embeddings(
        dev_df,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        device=device,
        description="canonical development embeddings",
    )
    X_holdout, y_holdout, meta_holdout = extract_embeddings(
        holdout_df,
        encoder=encoder,
        tokenizer=tokenizer,
        config=config,
        device=device,
        description="canonical outer holdout embeddings",
    )

    scaler, classifier = _fit_linear_probe(X_dev, y_dev, config=config)
    holdout_scores = _predict_linear_probe(scaler, classifier, X_holdout)

    predictions = meta_holdout.copy()
    predictions["label"] = y_holdout.astype(int)
    predictions["y_score"] = holdout_scores.astype(float)
    predictions["fold"] = -1

    metrics = compute_metrics(y_holdout, holdout_scores, threshold=threshold)
    metrics.update({"threshold_source": "development_predictions", "scope": "outer_holdout"})
    return predictions, metrics


def run_exp3_pipeline(config: Exp3LinearProbeConfig) -> Dict[str, object]:
    """Run EXP-3 and save development + optional holdout outputs."""

    start_time = time.time()
    paths = build_default_paths(config)
    run_output_dir = paths["run_output_dir"]
    run_output_dir.mkdir(parents=True, exist_ok=True)
    paths["embedding_cache_dir"].mkdir(parents=True, exist_ok=True)

    print("[EXP3-LP] Comparative development + holdout pipeline")
    print(f"[EXP3-LP] Data root: {paths['data_root']}")
    print(f"[EXP3-LP] Output dir: {run_output_dir}")
    print(f"[EXP3-LP] Mode={config.mode}; selected fold={config.selected_fold}")
    print(f"[EXP3-LP] Representation: {config.code_column} from {config.data_filename}")

    _save_json(asdict(config), run_output_dir / "exp3_config.json")
    audit_paths(paths).to_csv(run_output_dir / "path_audit.csv", index=False)

    full_df, dev_df, holdout_df, inner_manifest = _load_frames(config, paths)
    split_summary = {
        "full": _frame_summary(full_df, config),
        "development": _frame_summary(dev_df, config),
        "outer_holdout": _frame_summary(holdout_df, config),
    }
    _save_json(split_summary, run_output_dir / "split_summary.json")
    print("[EXP3-LP] Split summary:")
    print(json.dumps(split_summary, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_huggingface_cache(config.hf_cache_dir)
    tokenizer = load_neobert_tokenizer(config.tokenizer_name)
    print(f"[EXP3-LP] Loading frozen NeoBERT encoder on {device}; dtype_policy={config.dtype_policy}")
    encoder = load_neobert_encoder(
        config.model_name,
        dtype_policy=config.dtype_policy,
        device=device,
        freeze=True,
        hf_cache_dir=config.hf_cache_dir,
    )

    fold_ids, dev_frame_for_cv, holdout_eval_df, explicit_train, explicit_valid = _make_mode_frames(
        config, dev_df, holdout_df, inner_manifest
    )

    oof_predictions: List[pd.DataFrame] = []
    fold_metrics: List[dict] = []
    threshold_tables: List[pd.DataFrame] = []
    selected_thresholds: List[float] = []

    if config.mode.lower() in {"smoke", "pilot", "one_fold"}:
        assert explicit_train is not None and explicit_valid is not None
        fold_id = int(config.selected_fold)
        print(
            f"[EXP3-LP] Running {config.mode} fold={fold_id}: "
            f"train rows={len(explicit_train)}, valid rows={len(explicit_valid)}, "
            f"holdout eval rows={len(holdout_eval_df)}"
        )
        pred, metrics, threshold, threshold_table = _run_single_fold(
            fold_id=fold_id,
            train_df=explicit_train,
            valid_df=explicit_valid,
            encoder=encoder,
            tokenizer=tokenizer,
            config=config,
            device=device,
            output_dir=run_output_dir,
            prefix=config.mode,
        )
        oof_predictions.append(pred)
        fold_metrics.append(metrics)
        threshold_tables.append(threshold_table)
        selected_thresholds.append(threshold)

    else:
        for fold_id in fold_ids:
            train_ids = set(inner_manifest.loc[inner_manifest["fold"] != fold_id, config.source_id_column])
            valid_ids = set(inner_manifest.loc[inner_manifest["fold"] == fold_id, config.source_id_column])
            train_df = dev_frame_for_cv[dev_frame_for_cv[config.source_id_column].isin(train_ids)].copy().reset_index(drop=True)
            valid_df = dev_frame_for_cv[dev_frame_for_cv[config.source_id_column].isin(valid_ids)].copy().reset_index(drop=True)
            print(f"[EXP3-LP] Full-CV fold {fold_id}: train={len(train_df)}, valid={len(valid_df)}")
            pred, metrics, threshold, threshold_table = _run_single_fold(
                fold_id=int(fold_id),
                train_df=train_df,
                valid_df=valid_df,
                encoder=encoder,
                tokenizer=tokenizer,
                config=config,
                device=device,
                output_dir=run_output_dir,
                prefix="full_cv",
            )
            oof_predictions.append(pred)
            fold_metrics.append(metrics)
            threshold_tables.append(threshold_table)
            selected_thresholds.append(threshold)

    dev_predictions = pd.concat(oof_predictions, ignore_index=True)
    dev_predictions_path = run_output_dir / "exp3_development_predictions.parquet"
    dev_predictions.to_parquet(dev_predictions_path, index=False)
    dev_predictions.to_csv(run_output_dir / "exp3_development_predictions.csv", index=False)

    # Development threshold for pooled metrics and holdout operating point.
    # This uses development predictions only, never holdout labels.
    pooled_dev_threshold, pooled_dev_threshold_table = select_threshold(
        dev_predictions["label"].astype(int).to_numpy(),
        dev_predictions["y_score"].astype(float).to_numpy(),
        metric=config.threshold_metric,
    )
    pooled_dev_threshold_table.to_csv(run_output_dir / "exp3_development_pooled_threshold_table.csv", index=False)

    dev_metrics = compute_metrics(
        dev_predictions["label"].astype(int).to_numpy(),
        dev_predictions["y_score"].astype(float).to_numpy(),
        threshold=pooled_dev_threshold,
    )
    dev_metrics.update(
        {
            "scope": "development",
            "threshold_source": "development_predictions",
            "mode": config.mode,
        }
    )
    dev_ci = project_block_bootstrap_pr_auc_ci(
        dev_predictions,
        n_resamples=config.bootstrap_resamples,
        random_state=config.bootstrap_random_state,
    )
    dev_metrics.update({"pr_auc_ci_low": dev_ci["ci_low"], "pr_auc_ci_high": dev_ci["ci_high"], "ci_valid_resamples": dev_ci["valid_resamples"], "ci_n_projects": dev_ci["n_projects"]})

    pd.DataFrame(fold_metrics).to_csv(run_output_dir / "exp3_development_fold_metrics.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(run_output_dir / "exp3_threshold_tables_by_fold.csv", index=False)
    _save_json(dev_metrics, run_output_dir / "exp3_development_metrics.json")
    plot_precision_recall(dev_predictions, run_output_dir / "exp3_development_pr_curve.png", title="EXP-3 development PR curve")
    plot_confusion(dev_metrics, run_output_dir / "exp3_development_confusion_matrix.png", title="EXP-3 development confusion matrix")

    holdout_predictions = None
    holdout_metrics = None
    if config.evaluate_outer_holdout:
        print("[EXP3-LP] Evaluating canonical model on frozen outer holdout.")
        print("[EXP3-LP] Threshold is selected from development predictions only:", pooled_dev_threshold)
        canonical_dev_df = dev_df
        if config.mode.lower() == "smoke":
            canonical_dev_df = sample_with_optional_positive_fraction(
                dev_df,
                n_rows=config.smoke_train_rows + config.smoke_threshold_rows,
                label_column=config.label_column,
                positive_fraction=0.25,
                random_state=config.random_state + 100,
            )
        elif config.mode.lower() == "pilot":
            canonical_dev_df = sample_with_optional_positive_fraction(
                dev_df,
                n_rows=config.pilot_train_rows + config.pilot_threshold_rows,
                label_column=config.label_column,
                positive_fraction=config.pilot_positive_fraction,
                random_state=config.random_state + 100,
            )

        holdout_predictions, holdout_metrics = _canonical_holdout_evaluation(
            dev_df=canonical_dev_df,
            holdout_df=holdout_eval_df,
            encoder=encoder,
            tokenizer=tokenizer,
            config=config,
            device=device,
            output_dir=run_output_dir,
            threshold=pooled_dev_threshold,
        )
        holdout_predictions.to_parquet(run_output_dir / "exp3_outer_holdout_predictions.parquet", index=False)
        holdout_predictions.to_csv(run_output_dir / "exp3_outer_holdout_predictions.csv", index=False)
        holdout_ci = project_block_bootstrap_pr_auc_ci(
            holdout_predictions,
            n_resamples=config.bootstrap_resamples,
            random_state=config.bootstrap_random_state + 1,
        )
        holdout_metrics.update({"pr_auc_ci_low": holdout_ci["ci_low"], "pr_auc_ci_high": holdout_ci["ci_high"], "ci_valid_resamples": holdout_ci["valid_resamples"], "ci_n_projects": holdout_ci["n_projects"], "mode": config.mode})
        _save_json(holdout_metrics, run_output_dir / "exp3_outer_holdout_metrics.json")
        plot_precision_recall(holdout_predictions, run_output_dir / "exp3_outer_holdout_pr_curve.png", title="EXP-3 outer holdout PR curve")
        plot_confusion(holdout_metrics, run_output_dir / "exp3_outer_holdout_confusion_matrix.png", title="EXP-3 outer holdout confusion matrix")

    summary_rows = [dev_metrics]
    if holdout_metrics is not None:
        summary_rows.append(holdout_metrics)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_output_dir / "exp3_summary_metrics.csv", index=False)

    run_metadata = {
        "runtime_seconds": float(time.time() - start_time),
        "device": str(device),
        "config": asdict(config),
        "paths": {name: str(path) for name, path in paths.items()},
        "selected_fold_thresholds": [float(x) for x in selected_thresholds],
        "pooled_development_threshold": float(pooled_dev_threshold),
        "methodological_note": (
            "Both development and frozen outer-holdout results are produced for comparative analysis. "
            "The holdout threshold is selected from development predictions only. "
            "Holdout results should not be used to iteratively redesign the experiment."
        ),
    }
    _save_json(run_metadata, run_output_dir / "exp3_run_metadata.json")

    print("[EXP3-LP] Completed.")
    print("[EXP3-LP] Summary:")
    print(summary_df[["scope", "n", "positives", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "precision", "recall", "f1", "mcc"]].to_string(index=False))

    # Free GPU memory.
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "config": config,
        "paths": paths,
        "development_predictions": dev_predictions,
        "development_metrics": dev_metrics,
        "holdout_predictions": holdout_predictions,
        "holdout_metrics": holdout_metrics,
        "summary": summary_df,
        "output_dir": run_output_dir,
    }
