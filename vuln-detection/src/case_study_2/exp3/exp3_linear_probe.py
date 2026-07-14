"""Case Study 2 — EXP-3 NeoBERT Linear Probe.

Reusable module for the `notebooks/cs2_exp3_lp.ipynb` notebook.

EXP-3 purpose:
    Measure how much vulnerability signal exists in frozen NeoBERT embeddings
    before any adapter/fine-tuning.  The encoder is frozen; only a linear
    classifier is trained over CLS embeddings.

Methodology:
    - Use the same project-disjoint development folds as CS1.
    - Fit token/embedding transformations only after splitting.
    - Select operating threshold using an inner project-disjoint split of the
      outer training partition.
    - Keep frozen outer holdout locked by default.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from case_study_2.data_loader import create_dataloader
from case_study_2.models import (
    DEFAULT_NEOBERT_MODEL,
    DEFAULT_NEOBERT_TOKENIZER,
    NeoBertFrozenEncoder,
)


EXPERIMENT_ID = "cs1_project_holdout20_innercv_v1"


@dataclass
class Exp3LinearProbeConfig:
    # Data/layout
    data_root: Optional[str] = None
    experiment_id: str = EXPERIMENT_ID
    data_filename: str = "rdiversevul_cs1_normalized_v1.parquet"
    code_column: str = "normalized_code"
    label_column: str = "label"
    source_id_column: str = "source_row_id"
    project_column: str = "project"

    # Model/tokenization
    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    max_length: int = 512
    dtype_policy: str = "float16"
    embedding_batch_size: int = 4

    # Experiment mode
    mode: str = "smoke"  # smoke | pilot | one_fold | full_cv
    selected_fold: int = 4
    smoke_train_rows: int = 512
    smoke_valid_rows: int = 256
    pilot_train_rows: int = 8192
    pilot_valid_rows: int = 4096
    pilot_positive_fraction: float = 0.25
    inner_validation_fraction: float = 0.20

    # Linear probe
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = "lbfgs"
    class_weight: str = "balanced"

    # Reproducibility/output
    random_state: int = 42
    output_subdir: str = "case_study_2_exp3_linear_probe_dev_v1"
    run_final_outer_holdout: bool = False


def log(message: str) -> None:
    print(f"[EXP3-LP] {message}", flush=True)


def discover_data_root(data_root: Optional[str] = None) -> Path:
    """Find the VulnerabilityDetectionData root robustly in Colab/Drive."""

    if data_root:
        root = Path(data_root)
        if root.exists():
            return root
        raise FileNotFoundError(f"Configured data_root does not exist: {root}")

    candidates = [
        Path("/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData"),
        Path("/content/drive/My Drive/IntelligentSystemProject/VulnerabilityDetectionData"),
        Path.cwd() / "VulnerabilityDetectionData",
        Path.cwd().parent / "VulnerabilityDetectionData",
    ]

    for root in candidates:
        if (root / "processed").exists() and (root / "manifests").exists():
            return root

    # Last-resort targeted search.  This avoids a hard-coded path failure when
    # the Drive folder is nested differently.
    search_roots = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive"), Path.cwd()]
    target_name = "rdiversevul_cs1_normalized_v1.parquet"
    for search_root in search_roots:
        if not search_root.exists():
            continue
        try:
            matches = list(search_root.rglob(target_name))
        except Exception:
            matches = []
        if matches:
            return matches[0].parents[1]  # .../processed/file -> data root

    raise FileNotFoundError(
        "Could not locate VulnerabilityDetectionData. Mount Google Drive and/or set "
        "Exp3LinearProbeConfig(data_root='.../VulnerabilityDetectionData')."
    )


def build_default_paths(config: Exp3LinearProbeConfig) -> Dict[str, Path]:
    root = discover_data_root(config.data_root)
    exp_root = root / "manifests" / config.experiment_id
    output_root = root / "outputs" / config.experiment_id / config.output_subdir
    return {
        "data_root": root,
        "data": root / "processed" / config.data_filename,
        "outer_manifest": exp_root / "outer_holdout" / "cs1_outer_project_holdout_manifest.parquet",
        "inner_manifest": exp_root / "inner_cv" / "cs1_project_grouped_5fold_manifest.parquet",
        "output_dir": output_root,
    }


def audit_paths(paths: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists()})
    return pd.DataFrame(rows)


def require_paths(paths: Dict[str, Path], names: Iterable[str]) -> None:
    missing = [(name, paths[name]) for name in names if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files. Check Drive mount and paths:\n"
            + "\n".join(f"- {name}: {path}" for name, path in missing)
        )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_development_data(config: Exp3LinearProbeConfig, paths: Dict[str, Path]):
    require_paths(paths, ["data", "outer_manifest", "inner_manifest"])

    full_df = pd.read_parquet(paths["data"])
    outer_manifest = pd.read_parquet(paths["outer_manifest"])
    inner_manifest = pd.read_parquet(paths["inner_manifest"])

    required_cols = [
        config.source_id_column,
        config.project_column,
        config.label_column,
        config.code_column,
    ]
    missing = [c for c in required_cols if c not in full_df.columns]
    if missing:
        raise KeyError(f"Data file missing columns: {missing}. Available: {list(full_df.columns)[:20]}")

    dev_ids = set(
        outer_manifest.loc[
            outer_manifest["partition"].eq("development"),
            config.source_id_column,
        ].tolist()
    )
    holdout_ids = set(
        outer_manifest.loc[
            outer_manifest["partition"].eq("outer_holdout"),
            config.source_id_column,
        ].tolist()
    )

    development = full_df[full_df[config.source_id_column].isin(dev_ids)].copy().reset_index(drop=True)
    holdout = full_df[full_df[config.source_id_column].isin(holdout_ids)].copy().reset_index(drop=True)

    return development, holdout, inner_manifest, outer_manifest


def _sample_positive_enriched(
    frame: pd.DataFrame,
    n_rows: int,
    positive_fraction: float,
    label_column: str,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= n_rows:
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    pos = frame[frame[label_column].eq(1)]
    neg = frame[frame[label_column].eq(0)]

    n_pos = min(len(pos), int(round(n_rows * positive_fraction)))
    n_neg = min(len(neg), n_rows - n_pos)
    sampled = []
    if n_pos > 0:
        sampled.append(pos.sample(n=n_pos, random_state=random_state, replace=False))
    if n_neg > 0:
        sampled.append(neg.sample(n=n_neg, random_state=random_state + 1, replace=False))

    out = pd.concat(sampled, axis=0).sample(frac=1.0, random_state=random_state + 2)
    return out.reset_index(drop=True)


def _sample_natural(frame: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= n_rows:
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return frame.sample(n=n_rows, random_state=random_state, replace=False).reset_index(drop=True)


def make_outer_and_inner_split(
    development: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    config: Exp3LinearProbeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Return fit, threshold-validation, and outer-validation frames."""

    fold_id = int(config.selected_fold)
    train_ids = set(inner_manifest.loc[inner_manifest["fold"].ne(fold_id), config.source_id_column].tolist())
    outer_valid_ids = set(inner_manifest.loc[inner_manifest["fold"].eq(fold_id), config.source_id_column].tolist())

    outer_train = development[development[config.source_id_column].isin(train_ids)].copy().reset_index(drop=True)
    outer_valid = development[development[config.source_id_column].isin(outer_valid_ids)].copy().reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.inner_validation_fraction,
        random_state=config.random_state + fold_id,
    )
    fit_idx, threshold_idx = next(
        splitter.split(
            outer_train,
            y=outer_train[config.label_column],
            groups=outer_train[config.project_column],
        )
    )

    fit_df = outer_train.iloc[fit_idx].copy().reset_index(drop=True)
    threshold_df = outer_train.iloc[threshold_idx].copy().reset_index(drop=True)

    # Mode-specific subsetting AFTER project-disjoint splitting.
    if config.mode == "smoke":
        fit_df = _sample_positive_enriched(
            fit_df, config.smoke_train_rows, 0.25, config.label_column, config.random_state
        )
        threshold_df = _sample_positive_enriched(
            threshold_df, min(config.smoke_valid_rows, len(threshold_df)), 0.25, config.label_column, config.random_state + 10
        )
        outer_valid = _sample_positive_enriched(
            outer_valid, min(config.smoke_valid_rows, len(outer_valid)), 0.25, config.label_column, config.random_state + 20
        )
    elif config.mode == "pilot":
        fit_df = _sample_positive_enriched(
            fit_df, config.pilot_train_rows, config.pilot_positive_fraction, config.label_column, config.random_state
        )
        threshold_df = _sample_natural(
            threshold_df, min(config.pilot_valid_rows, len(threshold_df)), config.random_state + 10
        )
        outer_valid = _sample_natural(
            outer_valid, min(config.pilot_valid_rows, len(outer_valid)), config.random_state + 20
        )
    elif config.mode in {"one_fold", "full_cv"}:
        pass
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    meta = {
        "fold": fold_id,
        "outer_train_rows": int(len(outer_train)),
        "outer_valid_rows": int(len(outer_valid)),
        "inner_fit_rows": int(len(fit_df)),
        "inner_threshold_rows": int(len(threshold_df)),
        "outer_train_projects": int(outer_train[config.project_column].nunique()),
        "outer_valid_projects": int(outer_valid[config.project_column].nunique()),
        "inner_fit_projects": int(fit_df[config.project_column].nunique()),
        "inner_threshold_projects": int(threshold_df[config.project_column].nunique()),
        "mode": config.mode,
    }
    return fit_df, threshold_df, outer_valid, meta


@torch.no_grad()
def extract_cls_embeddings(
    frame: pd.DataFrame,
    tokenizer,
    encoder: NeoBertFrozenEncoder,
    config: Exp3LinearProbeConfig,
    device: torch.device,
) -> np.ndarray:
    loader = create_dataloader(
        frame,
        tokenizer,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
        shuffle=False,
        code_column=config.code_column,
        label_column=config.label_column,
        num_workers=0,
    )

    encoder.eval()
    embeddings = []
    total = len(loader)
    for idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda" and config.dtype_policy.lower() in {"auto", "float16", "fp16"}),
        ):
            cls = encoder(input_ids=input_ids, attention_mask=attention_mask)

        embeddings.append(cls.detach().float().cpu().numpy())
        if idx == 1 or idx % 50 == 0 or idx == total:
            log(f"Embedding batches {idx}/{total}")

    return np.concatenate(embeddings, axis=0)


def positive_scores_from_classifier(classifier: LogisticRegression, x: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x)[:, 1]
    scores = classifier.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def select_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        current_f1 = f1_score(y_true, pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n_samples": int(len(y_true)),
        "vulnerable_1": int(y_true.sum()),
        "non_vulnerable_0": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def run_exp3_pipeline(config: Exp3LinearProbeConfig):
    """Run EXP-3 according to the supplied config.

    Returns a dictionary with metrics and paths.  For `mode=full_cv`, this
    function currently runs the selected fold only by design; use the notebook
    to iterate folds after the pilot is validated.
    """

    start = time.time()
    paths = build_default_paths(config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    log(f"Data root: {paths['data_root']}")
    log(f"Output dir: {paths['output_dir']}")

    development, holdout, inner_manifest, outer_manifest = load_development_data(config, paths)

    split_summary = {
        "development_rows": int(len(development)),
        "development_projects": int(development[config.project_column].nunique()),
        "development_positives": int(development[config.label_column].sum()),
        "development_positive_rate": float(development[config.label_column].mean()),
        "holdout_rows_locked": int(len(holdout)),
        "holdout_projects_locked": int(holdout[config.project_column].nunique()),
        "holdout_positives_locked": int(holdout[config.label_column].sum()),
        "holdout_positive_rate_locked": float(holdout[config.label_column].mean()),
    }
    save_json(paths["output_dir"] / "split_summary.json", split_summary)

    if config.run_final_outer_holdout:
        raise RuntimeError(
            "EXP-3 final outer-holdout evaluation is intentionally locked. "
            "Compare CS2 candidates on development first, then evaluate the selected CS2 model once."
        )

    fit_df, threshold_df, outer_valid_df, split_meta = make_outer_and_inner_split(
        development, inner_manifest, config
    )

    log(f"Mode={config.mode}; selected fold={config.selected_fold}")
    log(f"Inner fit rows={len(fit_df):,}; threshold rows={len(threshold_df):,}; outer valid rows={len(outer_valid_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading frozen NeoBERT encoder on {device}; dtype_policy={config.dtype_policy}")
    encoder = NeoBertFrozenEncoder(model_name=config.model_name, dtype_policy=config.dtype_policy, freeze_backbone=True)
    encoder.to(device)

    log("Extracting fit embeddings...")
    x_fit = extract_cls_embeddings(fit_df, tokenizer, encoder, config, device)
    log("Extracting threshold-validation embeddings...")
    x_threshold = extract_cls_embeddings(threshold_df, tokenizer, encoder, config, device)
    log("Extracting outer-validation embeddings...")
    x_outer = extract_cls_embeddings(outer_valid_df, tokenizer, encoder, config, device)

    y_fit = fit_df[config.label_column].values.astype(int)
    y_threshold = threshold_df[config.label_column].values.astype(int)
    y_outer = outer_valid_df[config.label_column].values.astype(int)

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    x_threshold_scaled = scaler.transform(x_threshold)
    x_outer_scaled = scaler.transform(x_outer)

    clf = LogisticRegression(
        C=config.logistic_C,
        max_iter=config.logistic_max_iter,
        solver=config.logistic_solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    log("Training linear probe on frozen CLS embeddings...")
    clf.fit(x_fit_scaled, y_fit)

    threshold_scores = positive_scores_from_classifier(clf, x_threshold_scaled)
    selected_threshold, selected_f1 = select_threshold_by_f1(y_threshold, threshold_scores)

    outer_scores = positive_scores_from_classifier(clf, x_outer_scaled)
    metrics = compute_metrics(y_outer, outer_scores, selected_threshold)
    metrics.update({
        "case_study": "CS2",
        "experiment": "EXP-3 NeoBERT Linear Probe",
        "mode": config.mode,
        "fold": int(config.selected_fold),
        "selected_threshold_strategy": "inner_project_validation_f1",
        "selected_threshold_inner_f1": float(selected_f1),
        "model_name": config.model_name,
        "tokenizer_name": config.tokenizer_name,
        "code_column": config.code_column,
        "max_length": int(config.max_length),
        "runtime_seconds": float(time.time() - start),
    })

    predictions = pd.DataFrame({
        config.source_id_column: outer_valid_df[config.source_id_column].values,
        config.project_column: outer_valid_df[config.project_column].values,
        "label": y_outer,
        "y_score": outer_scores,
        "fold": int(config.selected_fold),
        "mode": config.mode,
    })

    predictions.to_parquet(paths["output_dir"] / "exp3_linear_probe_oof_predictions.parquet", index=False)
    predictions.to_csv(paths["output_dir"] / "exp3_linear_probe_oof_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(paths["output_dir"] / "exp3_linear_probe_summary.csv", index=False)
    save_json(paths["output_dir"] / "exp3_linear_probe_metrics.json", metrics)
    save_json(paths["output_dir"] / "exp3_linear_probe_split_metadata.json", split_meta)
    save_json(paths["output_dir"] / "exp3_linear_probe_config.json", asdict(config))
    joblib.dump(scaler, paths["output_dir"] / "exp3_linear_probe_scaler.joblib")
    joblib.dump(clf, paths["output_dir"] / "exp3_linear_probe_classifier.joblib")

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

        PrecisionRecallDisplay.from_predictions(y_outer, outer_scores)
        plt.title("EXP-3 Linear Probe — Project-Disjoint Validation PR Curve")
        plt.tight_layout()
        plt.savefig(paths["output_dir"] / "exp3_linear_probe_pr_curve.png", dpi=180)
        plt.close()

        y_pred = (outer_scores >= selected_threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_outer, y_pred, labels=[0, 1])
        plt.title("EXP-3 Linear Probe — Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(paths["output_dir"] / "exp3_linear_probe_confusion_matrix.png", dpi=180)
        plt.close()
    except Exception as exc:
        log(f"Plot export skipped: {exc}")

    log("EXP-3 Linear Probe completed.")
    log(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "paths": {k: str(v) for k, v in paths.items()}, "split_summary": split_summary}
