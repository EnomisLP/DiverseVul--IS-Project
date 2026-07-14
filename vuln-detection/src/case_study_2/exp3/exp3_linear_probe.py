"""
EXP-3: NeoBERT frozen-encoder linear probe.

Experiment-specific file. Shared utilities are imported from:
  - case_study_2.data_loader
  - case_study_2.models
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from case_study_2.data_loader import (
    create_dataloader,
    sample_with_optional_positive_fraction,
    make_project_disjoint_threshold_split,
)
from case_study_2.models import (
    DEFAULT_NEOBERT_MODEL,
    DEFAULT_NEOBERT_TOKENIZER,
    configure_huggingface_cache,
    load_neobert_tokenizer,
    load_neobert_encoder,
)


@dataclass
class Exp3LinearProbeConfig:
    data_root: str
    split_id: str = "cs1_project_holdout20_innercv_v1"

    data_filename: str = "rdiversevul_cs1_normalized_v1.parquet"
    code_column: str = "normalized_code"
    label_column: str = "label"
    source_id_column: str = "source_row_id"
    project_column: str = "project"

    model_name: str = DEFAULT_NEOBERT_MODEL
    tokenizer_name: str = DEFAULT_NEOBERT_TOKENIZER
    hf_cache_dir: str = "/content/hf_cache"
    max_length: int = 512
    dtype_policy: str = "auto"
    embedding_batch_size: int = 4

    mode: str = "smoke"  # smoke, pilot, one_fold, full_cv
    selected_fold: int = 0

    smoke_train_rows: int = 512
    smoke_valid_rows: int = 256
    smoke_holdout_rows: int = 256

    pilot_train_rows: int = 8192
    pilot_valid_rows: int = 4096
    pilot_holdout_rows: int = 4096
    pilot_positive_fraction: float = 0.25

    threshold_fraction: float = 0.20

    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = "lbfgs"
    class_weight: str = "balanced"

    output_subdir: str = "exp3_linear_probe_dev_and_holdout_v1"
    random_state: int = 42


def build_paths(config: Exp3LinearProbeConfig) -> Dict[str, Path]:
    root = Path(config.data_root)
    return {
        "data": root / "processed" / config.data_filename,
        "outer_manifest": root / "manifests" / config.split_id / "outer_holdout" / "cs1_outer_project_holdout_manifest.parquet",
        "inner_manifest": root / "manifests" / config.split_id / "inner_cv" / "cs1_project_grouped_5fold_manifest.parquet",
        "output_dir": root / "outputs" / "case_study_2" / config.output_subdir / config.mode,
    }


def validate_paths(paths: Dict[str, Path]) -> None:
    required = ["data", "outer_manifest", "inner_manifest"]
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(f"- {name}: {paths[name]}" for name in missing)
        )


def load_split_frames(config: Exp3LinearProbeConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = build_paths(config)
    validate_paths(paths)

    df = pd.read_parquet(paths["data"])
    outer = pd.read_parquet(paths["outer_manifest"])
    inner = pd.read_parquet(paths["inner_manifest"])

    dev_ids = set(outer.loc[outer["partition"] == "development", config.source_id_column].astype(int))
    holdout_ids = set(outer.loc[outer["partition"] == "outer_holdout", config.source_id_column].astype(int))

    dev = df[df[config.source_id_column].isin(dev_ids)].copy().reset_index(drop=True)
    holdout = df[df[config.source_id_column].isin(holdout_ids)].copy().reset_index(drop=True)

    for frame in [dev, holdout]:
        frame[config.code_column] = frame[config.code_column].fillna("").astype(str)
        empty_mask = frame[config.code_column].str.strip().eq("")
        if empty_mask.any():
            frame.loc[empty_mask, config.code_column] = "EMPTY_CODE_SAMPLE"

    return df, dev, holdout, inner


def split_summary(df: pd.DataFrame, dev: pd.DataFrame, holdout: pd.DataFrame, config: Exp3LinearProbeConfig) -> Dict[str, Any]:
    def summarize(frame: pd.DataFrame) -> Dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "projects": int(frame[config.project_column].nunique()),
            "positives": int(frame[config.label_column].sum()),
            "positive_rate": float(frame[config.label_column].mean()),
        }
    return {"full": summarize(df), "development": summarize(dev), "outer_holdout": summarize(holdout)}


def get_fold_frames(
    dev: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    config: Exp3LinearProbeConfig,
    fold_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = set(inner_manifest.loc[inner_manifest["fold"] != fold_id, config.source_id_column].astype(int))
    valid_ids = set(inner_manifest.loc[inner_manifest["fold"] == fold_id, config.source_id_column].astype(int))

    train = dev[dev[config.source_id_column].isin(train_ids)].copy().reset_index(drop=True)
    valid = dev[dev[config.source_id_column].isin(valid_ids)].copy().reset_index(drop=True)
    return train, valid


@torch.no_grad()
def extract_embeddings(
    encoder,
    tokenizer,
    frame: pd.DataFrame,
    config: Exp3LinearProbeConfig,
    device: str,
) -> np.ndarray:
    loader = create_dataloader(
        frame,
        tokenizer,
        batch_size=config.embedding_batch_size,
        max_length=config.max_length,
        shuffle=False,
        code_column=config.code_column,
        label_column=config.label_column,
        source_id_column=config.source_id_column,
        project_column=config.project_column,
    )

    all_embeddings = []
    encoder.eval()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            pooled = pooled.float().detach().cpu().numpy()

        all_embeddings.append(pooled)

    return np.concatenate(all_embeddings, axis=0)


def choose_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def project_block_pr_auc_ci(
    predictions: pd.DataFrame,
    label_column: str = "label",
    score_column: str = "y_score",
    project_column: str = "project",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    projects = predictions[project_column].dropna().unique()
    point = average_precision_score(predictions[label_column], predictions[score_column])
    values = []

    for _ in range(n_bootstrap):
        sampled_projects = rng.choice(projects, size=len(projects), replace=True)
        sample_parts = []
        for project in sampled_projects:
            sample_parts.append(predictions[predictions[project_column] == project])
        sample = pd.concat(sample_parts, axis=0)
        if sample[label_column].nunique() < 2:
            continue
        values.append(average_precision_score(sample[label_column], sample[score_column]))

    if values:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {
            "pr_auc_point": float(point),
            "pr_auc_ci_low": float(lo),
            "pr_auc_ci_high": float(hi),
            "bootstrap_valid_resamples": int(len(values)),
            "n_projects": int(len(projects)),
        }

    return {
        "pr_auc_point": float(point),
        "pr_auc_ci_low": None,
        "pr_auc_ci_high": None,
        "bootstrap_valid_resamples": 0,
        "n_projects": int(len(projects)),
    }


def plot_pr_curve(predictions: pd.DataFrame, output_path: Path, title: str) -> None:
    y_true = predictions["label"].astype(int).values
    y_score = predictions["y_score"].astype(float).values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.axhline(baseline, linestyle="--", label=f"Baseline={baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(metrics: Dict[str, Any], output_path: Path, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

    plt.figure(figsize=(4, 4))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def fit_probe_and_predict(
    encoder,
    tokenizer,
    fit_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    config: Exp3LinearProbeConfig,
    device: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()

    x_fit = extract_embeddings(encoder, tokenizer, fit_frame, config, device)
    x_threshold = extract_embeddings(encoder, tokenizer, threshold_frame, config, device)
    x_valid = extract_embeddings(encoder, tokenizer, valid_frame, config, device)
    x_holdout = extract_embeddings(encoder, tokenizer, holdout_frame, config, device)

    y_fit = fit_frame[config.label_column].astype(int).values
    y_threshold = threshold_frame[config.label_column].astype(int).values

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    x_threshold_s = scaler.transform(x_threshold)
    x_valid_s = scaler.transform(x_valid)
    x_holdout_s = scaler.transform(x_holdout)

    clf = LogisticRegression(
        C=config.logistic_C,
        max_iter=config.logistic_max_iter,
        solver=config.logistic_solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=None,
    )
    clf.fit(x_fit_s, y_fit)

    threshold_scores = clf.predict_proba(x_threshold_s)[:, 1]
    selected_threshold = choose_threshold_f1(y_threshold, threshold_scores)

    valid_scores = clf.predict_proba(x_valid_s)[:, 1]
    holdout_scores = clf.predict_proba(x_holdout_s)[:, 1]

    dev_predictions = valid_frame[
        [config.source_id_column, config.project_column, config.label_column]
    ].copy()
    dev_predictions = dev_predictions.rename(columns={config.label_column: "label"})
    dev_predictions["y_score"] = valid_scores
    dev_predictions["fold"] = int(config.selected_fold)

    holdout_predictions = holdout_frame[
        [config.source_id_column, config.project_column, config.label_column]
    ].copy()
    holdout_predictions = holdout_predictions.rename(columns={config.label_column: "label"})
    holdout_predictions["y_score"] = holdout_scores
    holdout_predictions["fold"] = -1

    metadata = {
        "selected_threshold": float(selected_threshold),
        "fit_rows": int(len(fit_frame)),
        "threshold_rows": int(len(threshold_frame)),
        "valid_rows": int(len(valid_frame)),
        "holdout_rows": int(len(holdout_frame)),
        "embedding_dim": int(x_fit.shape[1]),
        "runtime_seconds": float(time.time() - t0),
    }

    return dev_predictions, holdout_predictions, metadata


def run_exp3_pipeline(config: Exp3LinearProbeConfig) -> Dict[str, Any]:
    paths = build_paths(config)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[EXP3-LP] Comparative development + holdout pipeline")
    print("[EXP3-LP] Data root:", config.data_root)
    print("[EXP3-LP] Output dir:", output_dir)
    print(f"[EXP3-LP] Mode={config.mode}; selected fold={config.selected_fold}")
    print(f"[EXP3-LP] Representation: {config.code_column} from {config.data_filename}")

    df, dev, holdout, inner_manifest = load_split_frames(config)
    summary = split_summary(df, dev, holdout, config)
    print("[EXP3-LP] Split summary:")
    print(json.dumps(summary, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_huggingface_cache(config.hf_cache_dir)

    tokenizer = load_neobert_tokenizer(config.tokenizer_name, hf_cache_dir=config.hf_cache_dir)
    print(f"[EXP3-LP] Loading frozen NeoBERT encoder on {device}; dtype_policy={config.dtype_policy}")
    encoder = load_neobert_encoder(
        config.model_name,
        dtype_policy=config.dtype_policy,
        device=device,
        freeze=True,
        hf_cache_dir=config.hf_cache_dir,
    )

    dev_prediction_parts = []
    holdout_prediction_parts = []
    run_metadata = {
        "config": asdict(config),
        "split_summary": summary,
        "fold_metadata": [],
    }

    if config.mode not in {"smoke", "pilot", "one_fold", "full_cv"}:
        raise ValueError("mode must be one of: smoke, pilot, one_fold, full_cv")

    folds = [int(config.selected_fold)] if config.mode in {"smoke", "pilot", "one_fold"} else [0, 1, 2, 3, 4]

    for fold_id in folds:
        print(f"\n[EXP3-LP] Fold {fold_id}")
        outer_train, outer_valid = get_fold_frames(dev, inner_manifest, config, fold_id)

        fit_frame, threshold_frame = make_project_disjoint_threshold_split(
            outer_train,
            threshold_fraction=config.threshold_fraction,
            project_column=config.project_column,
            label_column=config.label_column,
            random_state=config.random_state + fold_id,
        )

        valid_frame = outer_valid.copy()
        holdout_frame = holdout.copy()

        if config.mode == "smoke":
            fit_frame = sample_with_optional_positive_fraction(
                fit_frame, config.smoke_train_rows, config.label_column, 0.25, config.random_state
            )
            threshold_frame = sample_with_optional_positive_fraction(
                threshold_frame, config.smoke_valid_rows, config.label_column, 0.25, config.random_state + 1
            )
            valid_frame = sample_with_optional_positive_fraction(
                valid_frame, config.smoke_valid_rows, config.label_column, 0.25, config.random_state + 2
            )
            holdout_frame = sample_with_optional_positive_fraction(
                holdout_frame, config.smoke_holdout_rows, config.label_column, 0.25, config.random_state + 3
            )

        elif config.mode == "pilot":
            fit_frame = sample_with_optional_positive_fraction(
                fit_frame, config.pilot_train_rows, config.label_column, config.pilot_positive_fraction, config.random_state
            )
            threshold_frame = sample_with_optional_positive_fraction(
                threshold_frame, config.pilot_valid_rows, config.label_column, config.pilot_positive_fraction, config.random_state + 1
            )
            valid_frame = sample_with_optional_positive_fraction(
                valid_frame, config.pilot_valid_rows, config.label_column, None, config.random_state + 2
            )
            holdout_frame = sample_with_optional_positive_fraction(
                holdout_frame, config.pilot_holdout_rows, config.label_column, None, config.random_state + 3
            )

        print(
            f"[EXP3-LP] fit={len(fit_frame)}, threshold={len(threshold_frame)}, "
            f"dev_valid={len(valid_frame)}, holdout={len(holdout_frame)}"
        )

        dev_pred, holdout_pred, meta = fit_probe_and_predict(
            encoder=encoder,
            tokenizer=tokenizer,
            fit_frame=fit_frame,
            threshold_frame=threshold_frame,
            valid_frame=valid_frame,
            holdout_frame=holdout_frame,
            config=config,
            device=device,
        )

        dev_pred["fold"] = fold_id
        holdout_pred["fold"] = fold_id
        dev_prediction_parts.append(dev_pred)
        holdout_prediction_parts.append(holdout_pred)
        meta["fold"] = fold_id
        run_metadata["fold_metadata"].append(meta)

    dev_predictions = pd.concat(dev_prediction_parts, axis=0).reset_index(drop=True)

    # For full_cv, holdout is scored once per fold. Average fold scores for a stable comparative estimate.
    holdout_all = pd.concat(holdout_prediction_parts, axis=0).reset_index(drop=True)
    holdout_predictions = (
        holdout_all
        .groupby([config.source_id_column, config.project_column, "label"], as_index=False)["y_score"]
        .mean()
    )
    holdout_predictions["fold"] = -1

    # Use the mean threshold selected over folds.
    selected_threshold = float(np.mean([m["selected_threshold"] for m in run_metadata["fold_metadata"]]))

    dev_metrics = compute_metrics(dev_predictions["label"], dev_predictions["y_score"], selected_threshold)
    holdout_metrics = compute_metrics(holdout_predictions["label"], holdout_predictions["y_score"], selected_threshold)

    dev_ci = project_block_pr_auc_ci(
        dev_predictions,
        project_column=config.project_column,
        n_bootstrap=1000 if config.mode == "full_cv" else 200,
        random_state=config.random_state,
    )
    holdout_ci = project_block_pr_auc_ci(
        holdout_predictions,
        project_column=config.project_column,
        n_bootstrap=1000 if config.mode == "full_cv" else 200,
        random_state=config.random_state + 10,
    )

    dev_metrics.update({
        "pr_auc_ci_low": dev_ci["pr_auc_ci_low"],
        "pr_auc_ci_high": dev_ci["pr_auc_ci_high"],
        "bootstrap_valid_resamples": dev_ci["bootstrap_valid_resamples"],
        "n_projects": dev_ci["n_projects"],
    })
    holdout_metrics.update({
        "pr_auc_ci_low": holdout_ci["pr_auc_ci_low"],
        "pr_auc_ci_high": holdout_ci["pr_auc_ci_high"],
        "bootstrap_valid_resamples": holdout_ci["bootstrap_valid_resamples"],
        "n_projects": holdout_ci["n_projects"],
    })

    dev_predictions.to_parquet(output_dir / "exp3_development_predictions.parquet", index=False)
    holdout_predictions.to_parquet(output_dir / "exp3_outer_holdout_predictions.parquet", index=False)

    with open(output_dir / "exp3_development_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2)
    with open(output_dir / "exp3_outer_holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_metrics, f, indent=2)

    summary_table = pd.DataFrame([
        {"scope": "development", **dev_metrics},
        {"scope": "outer_holdout", **holdout_metrics},
    ])
    summary_table.to_csv(output_dir / "exp3_summary_metrics.csv", index=False)

    plot_pr_curve(dev_predictions, output_dir / "exp3_development_pr_curve.png", "EXP-3 Linear Probe Development PR Curve")
    plot_pr_curve(holdout_predictions, output_dir / "exp3_outer_holdout_pr_curve.png", "EXP-3 Linear Probe Outer Holdout PR Curve")
    plot_confusion_matrix(dev_metrics, output_dir / "exp3_development_confusion_matrix.png", "EXP-3 Development Confusion Matrix")
    plot_confusion_matrix(holdout_metrics, output_dir / "exp3_outer_holdout_confusion_matrix.png", "EXP-3 Outer Holdout Confusion Matrix")

    with open(output_dir / "exp3_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print("\n[EXP3-LP] Completed.")
    print("[EXP3-LP] Development PR-AUC:", dev_metrics["pr_auc"])
    print("[EXP3-LP] Outer holdout PR-AUC:", holdout_metrics["pr_auc"])
    print("[EXP3-LP] Output dir:", output_dir)

    # Free GPU memory.
    del encoder
    torch.cuda.empty_cache()

    return {
        "output_dir": str(output_dir),
        "development_metrics": dev_metrics,
        "holdout_metrics": holdout_metrics,
        "summary": summary_table,
        "run_metadata": run_metadata,
    }
