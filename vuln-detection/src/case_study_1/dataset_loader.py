"""Robust RDiverseVul loading and audit utilities for Case Study 1."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_COLUMN_CANDIDATES = (
    "func", "function", "code", "source_code", "code_sequence", "source",
)
LABEL_COLUMN_CANDIDATES = (
    "target", "label", "vulnerable", "is_vulnerable", "is_vuln", "y",
)
PROJECT_COLUMN_CANDIDATES = (
    "project", "project_name", "repository", "repo", "repo_name",
)
CWE_COLUMN_CANDIDATES = ("cwe", "cwe_class", "cwe_id", "cwe_ids")
RECORD_CONTAINER_CANDIDATES = (
    "data", "records", "samples", "items", "dataset", "functions",
)


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be converted into tabular records."""


class DatasetSchemaError(ValueError):
    """Raised when required semantic fields are absent."""


def _is_missing_scalar(value: Any) -> bool:
    """True only for scalar missing values; never raises for lists/dicts."""
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalise_column_name(column: Any) -> str:
    return str(column).strip().lower()


def _find_column(columns: Sequence[Any], candidates: Sequence[str]) -> str | None:
    lookup = {_normalise_column_name(column): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _looks_like_record_list(value: Any) -> bool:
    return _is_non_string_sequence(value) and (
        len(value) == 0 or all(isinstance(item, Mapping) for item in value)
    )


def _mapping_to_dataframe(root: Mapping[str, Any]) -> pd.DataFrame:
    """Convert common JSON-object layouts into a dataframe."""
    columns = list(root.keys())
    code_key = _find_column(columns, CODE_COLUMN_CANDIDATES)
    label_key = _find_column(columns, LABEL_COLUMN_CANDIDATES)

    # A. Single record or column-oriented object.
    if code_key is not None and label_key is not None:
        code_value = root[code_key]
        if isinstance(code_value, str) or _is_missing_scalar(code_value):
            return pd.json_normalize([root], sep=".")
        try:
            return pd.DataFrame(root)
        except (TypeError, ValueError):
            pass

    # B. Named list of records, e.g. {"data": [{...}, {...}]}.
    for key in RECORD_CONTAINER_CANDIDATES:
        if key in root and _looks_like_record_list(root[key]):
            return pd.json_normalize(root[key], sep=".")

    # C. Dictionary of records, e.g. {"0": {"func": ...}, "1": {...}}.
    if root and all(isinstance(value, Mapping) for value in root.values()):
        try:
            candidate = pd.DataFrame.from_dict(root, orient="index")
            if _find_column(list(candidate.columns), CODE_COLUMN_CANDIDATES) is not None:
                return candidate.reset_index(drop=True)
        except (TypeError, ValueError):
            pass

    # D. One unrecognised list of record objects.
    record_lists = [value for value in root.values() if _looks_like_record_list(value)]
    if len(record_lists) == 1:
        return pd.json_normalize(record_lists[0], sep=".")

    raise DatasetLoadError(
        "The JSON root is an object, but its record layout is ambiguous. "
        f"Top-level keys: {columns[:20]}. Supported layouts: list of records, "
        "single record, named record container, dictionary of records, or "
        "column-oriented object."
    )


def _read_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    if isinstance(root, list):
        if not root:
            return pd.DataFrame()
        if not all(isinstance(item, Mapping) for item in root):
            raise DatasetLoadError("JSON root is a list but contains non-record values.")
        return pd.json_normalize(root, sep=".")
    if isinstance(root, Mapping):
        return _mapping_to_dataframe(root)
    raise DatasetLoadError(
        f"Unsupported JSON root type: {type(root).__name__}; expected list or object."
    )


def load_raw_table(path: str | Path) -> pd.DataFrame:
    """Load CSV, JSON, JSONL, or Parquet into an unmodified dataframe."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(source)
    elif suffix in {".jsonl", ".ndjson"}:
        dataframe = pd.read_json(source, lines=True)
    elif suffix == ".json":
        dataframe = _read_json(source)
    elif suffix in {".parquet", ".pq"}:
        dataframe = pd.read_parquet(source)
    else:
        raise DatasetLoadError(
            f"Unsupported file type '{suffix}'. Use .csv, .json, .jsonl, .ndjson, .parquet, or .pq."
        )

    if dataframe.empty:
        raise DatasetLoadError(f"The loaded dataset is empty: {source}")
    dataframe = dataframe.reset_index(drop=True)
    dataframe.attrs["source_path"] = str(source)
    return dataframe


def _coerce_binary_label(value: Any) -> int | pd.NA:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else pd.NA
    if isinstance(value, (float, np.floating)):
        return int(value) if np.isfinite(value) and float(value) in (0.0, 1.0) else pd.NA
    mapping = {
        "0": 0, "false": 0, "safe": 0, "secure": 0, "benign": 0,
        "non-vulnerable": 0, "non_vulnerable": 0, "not_vulnerable": 0,
        "1": 1, "true": 1, "vulnerable": 1, "vuln": 1,
    }
    return mapping.get(str(value).strip().lower(), pd.NA)


def _coerce_code(value: Any) -> str | pd.NA:
    return value if isinstance(value, str) else pd.NA


def _coerce_project(value: Any) -> str | pd.NA:
    if _is_missing_scalar(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _coerce_cwe_list(value: Any) -> list[str]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                raw_values = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw_values = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    raw_values = [text]
            if not _is_non_string_sequence(raw_values):
                raw_values = [raw_values]
        else:
            raw_values = [text]
    else:
        raw_values = [value]
    return sorted({str(item).strip() for item in raw_values if not _is_missing_scalar(item) and str(item).strip()})


def standardize_schema(raw_dataframe: pd.DataFrame, *, require_project: bool = True) -> pd.DataFrame:
    """Add canonical code, label, project, cwe, and source_row_id columns."""
    if raw_dataframe.empty:
        raise DatasetSchemaError("Cannot standardize an empty dataframe.")

    source = raw_dataframe.copy()
    raw_columns = [str(column) for column in source.columns]
    code_column = _find_column(source.columns, CODE_COLUMN_CANDIDATES)
    label_column = _find_column(source.columns, LABEL_COLUMN_CANDIDATES)
    project_column = _find_column(source.columns, PROJECT_COLUMN_CANDIDATES)
    cwe_column = _find_column(source.columns, CWE_COLUMN_CANDIDATES)

    missing = []
    if code_column is None:
        missing.append(f"code field (tried: {CODE_COLUMN_CANDIDATES})")
    if label_column is None:
        missing.append(f"label field (tried: {LABEL_COLUMN_CANDIDATES})")
    if require_project and project_column is None:
        missing.append(f"project field (tried: {PROJECT_COLUMN_CANDIDATES})")
    if missing:
        raise DatasetSchemaError("Required fields missing: " + "; ".join(missing))

    source["code"] = source[code_column].map(_coerce_code).astype("string")
    source["label"] = source[label_column].map(_coerce_binary_label).astype("Int64")
    source["project"] = (
        source[project_column].map(_coerce_project).astype("string")
        if project_column is not None
        else pd.Series(pd.NA, index=source.index, dtype="string")
    )
    source["cwe"] = (
        source[cwe_column].map(_coerce_cwe_list)
        if cwe_column is not None
        else [[] for _ in range(len(source))]
    )
    source["source_row_id"] = np.arange(len(source), dtype=np.int64)
    source.attrs["source_path"] = raw_dataframe.attrs.get("source_path", "")
    source.attrs["raw_columns"] = raw_columns
    source.attrs["source_columns"] = {
        "code": code_column, "label": label_column,
        "project": project_column, "cwe": cwe_column,
    }
    return source


def load_dataset(path: str | Path, *, require_project: bool = True) -> pd.DataFrame:
    return standardize_schema(load_raw_table(path), require_project=require_project)


def _summary(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"count": 0, "min": 0, "q25": 0, "median": 0, "q75": 0, "max": 0, "mean": 0.0}
    quantiles = values.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(values.size), "min": int(values.min()),
        "q25": round(float(quantiles.loc[0.25]), 2),
        "median": round(float(quantiles.loc[0.5]), 2),
        "q75": round(float(quantiles.loc[0.75]), 2),
        "max": int(values.max()), "mean": round(float(values.mean()), 2),
    }


def _code_digest(code: str) -> str:
    return sha256(code.encode("utf-8", errors="replace")).hexdigest()


def audit_dataset(dataframe: pd.DataFrame, *, top_n_projects: int = 15, top_n_cwes: int = 15) -> dict[str, Any]:
    """Audit standardized data without dropping, deduplicating, or relabelling records."""
    required = {"code", "label", "project", "cwe", "source_row_id"}
    absent = required.difference(dataframe.columns)
    if absent:
        raise DatasetSchemaError("Audit needs standardized columns. Missing: " + ", ".join(sorted(absent)))

    code = dataframe["code"].astype("string")
    project = dataframe["project"].astype("string")
    label = dataframe["label"].astype("Int64")
    valid_code = code.notna() & code.str.strip().ne("")
    valid_project = project.notna() & project.str.strip().ne("")
    valid_label = label.isin([0, 1])

    label_counts = label[valid_label].value_counts().sort_index()
    n_vulnerable = int(label_counts.get(1, 0))
    n_non_vulnerable = int(label_counts.get(0, 0))
    n_valid_labels = int(valid_label.sum())

    hashes = code.where(valid_code).map(lambda value: _code_digest(str(value)) if pd.notna(value) else pd.NA).astype("string")
    duplicate_mask = hashes.notna() & hashes.duplicated(keep=False)
    duplicate_excess = int((hashes.notna() & hashes.duplicated(keep="first")).sum())
    hash_labels = pd.DataFrame({"hash": hashes, "label": label}).dropna()
    conflicting_hashes = (hash_labels.groupby("hash")["label"].nunique() > 1).sum()

    usable_code = code[valid_code]
    char_lengths = usable_code.str.len().astype(int)
    line_counts = usable_code.str.count("\n").add(1).astype(int)
    project_counts = project[valid_project].value_counts()
    cwes = dataframe["cwe"].map(_coerce_cwe_list).explode().dropna()
    cwes = cwes[cwes.astype(str).str.strip().ne("")].astype(str)
    cwe_counts = cwes.value_counts()

    warnings = []
    if n_valid_labels != len(dataframe):
        warnings.append("Some labels are missing or outside the accepted binary set {0, 1}.")
    if int((~valid_code).sum()):
        warnings.append("Some code entries are missing, non-string, or blank.")
    if int((~valid_project).sum()):
        warnings.append("Some project entries are missing or blank.")
    if duplicate_excess:
        warnings.append("Exact duplicate code rows exist; duplicate policy must be fixed before evaluation.")
    if conflicting_hashes:
        warnings.append("Some exact duplicate functions have conflicting labels; they require an explicit policy.")
    if n_valid_labels and n_vulnerable / n_valid_labels < 0.10:
        warnings.append("Positive class is rare; accuracy must not be a primary metric.")

    return {
        "source": {
            "path": dataframe.attrs.get("source_path", ""),
            "raw_columns": dataframe.attrs.get("raw_columns", []),
            "canonical_source_columns": dataframe.attrs.get("source_columns", {}),
        },
        "rows": {
            "total": int(len(dataframe)),
            "valid_code": int(valid_code.sum()),
            "valid_binary_label": n_valid_labels,
            "valid_project": int(valid_project.sum()),
            "fully_usable_for_grouped_cv": int((valid_code & valid_label & valid_project).sum()),
        },
        "labels": {
            "non_vulnerable_0": n_non_vulnerable,
            "vulnerable_1": n_vulnerable,
            "vulnerable_rate": round(n_vulnerable / n_valid_labels, 6) if n_valid_labels else None,
            "invalid_or_missing": int((~valid_label).sum()),
        },
        "missing_or_blank": {"code": int((~valid_code).sum()), "project": int((~valid_project).sum())},
        "projects": {
            "unique_valid_projects": int(project_counts.size),
            "top_projects": {str(key): int(value) for key, value in project_counts.head(top_n_projects).items()},
        },
        "code_length": {"characters": _summary(char_lengths), "lines": _summary(line_counts)},
        "duplicates": {
            "rows_in_exact_duplicate_groups": int(duplicate_mask.sum()),
            "duplicate_rows_beyond_first_occurrence": duplicate_excess,
            "exact_code_hashes_with_conflicting_labels": int(conflicting_hashes),
        },
        "cwe": {
            "rows_with_at_least_one_cwe": int(dataframe["cwe"].map(_coerce_cwe_list).map(bool).sum()),
            "unique_observed_cwe_values": int(cwe_counts.size),
            "top_cwes": {str(key): int(value) for key, value in cwe_counts.head(top_n_cwes).items()},
        },
        "warnings": warnings,
    }


def save_audit_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return destination


def format_audit_report(report: Mapping[str, Any]) -> str:
    rows, labels = report["rows"], report["labels"]
    projects, duplicates = report["projects"], report["duplicates"]
    lengths = report["code_length"]["characters"]
    lines = [
        "=" * 72, "CASE STUDY 1 — DATASET AUDIT", "=" * 72,
        f"Rows loaded:                 {rows['total']:,}",
        f"Fully usable grouped-CV rows: {rows['fully_usable_for_grouped_cv']:,}",
        f"Vulnerable / non-vulnerable: {labels['vulnerable_1']:,} / {labels['non_vulnerable_0']:,}",
        f"Vulnerable rate:             {labels['vulnerable_rate']}",
        f"Unique projects:             {projects['unique_valid_projects']:,}",
        f"Exact duplicate excess rows: {duplicates['duplicate_rows_beyond_first_occurrence']:,}",
        f"Conflicting duplicate hashes: {duplicates['exact_code_hashes_with_conflicting_labels']:,}",
        f"Code length (characters):    median={lengths['median']}, q75={lengths['q75']}, max={lengths['max']}",
    ]
    warnings = report.get("warnings", [])
    lines.extend(["Warnings:"] + [f"  - {warning}" for warning in warnings] if warnings else ["Warnings: none"])
    lines.append("=" * 72)
    return "\n".join(lines)


def load_and_audit(path: str | Path, *, audit_output_path: str | Path | None = None, require_project: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataframe = load_dataset(path, require_project=require_project)
    report = audit_dataset(dataframe)
    if audit_output_path is not None:
        save_audit_report(report, audit_output_path)
    return dataframe, report
