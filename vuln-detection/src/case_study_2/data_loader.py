"""Shared data utilities for Case Study 2 experiments.

This module is intentionally experiment-neutral.  It should be imported by
EXP-3 (linear probe), EXP-4 (LoRA), and EXP-5 (HEFT/ReFT) so that all
transformer-based experiments use the same data conventions.

The utilities here do not decide which experiment to run.  They only handle:
- safe code/label dataframe preparation,
- PyTorch dataset/dataloader creation,
- dynamic tokenizer padding,
- class-weight computation,
- simple reproducible sampling for smoke/pilot runs,
- project-aware threshold splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit


EMPTY_CODE_SENTINEL = "EMPTY_CODE_SAMPLE"


@dataclass(frozen=True)
class DatasetColumns:
    """Column names used across CS2 experiments."""

    code: str = "normalized_code"
    label: str = "label"
    source_id: str = "source_row_id"
    project: str = "project"


def validate_required_columns(frame: pd.DataFrame, columns: DatasetColumns) -> None:
    """Raise a clear error if expected dataframe columns are missing."""

    required = [columns.code, columns.label, columns.source_id, columns.project]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(
            "Missing required dataframe columns: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(frame.columns)}"
        )


def prepare_code_frame(frame: pd.DataFrame, columns: DatasetColumns) -> pd.DataFrame:
    """Return a safe copy with string code and integer labels.

    Empty code strings are replaced with a deterministic sentinel so tokenizer
    calls never receive an empty string.  This is the same idea used in CS1 for
    the one empty identifier-abstraction row, but the helper remains generic.
    """

    validate_required_columns(frame, columns)
    out = frame.copy().reset_index(drop=True)
    out[columns.code] = out[columns.code].fillna("").astype(str)
    empty_mask = out[columns.code].str.strip().eq("")
    if empty_mask.any():
        out.loc[empty_mask, columns.code] = EMPTY_CODE_SENTINEL
    out[columns.label] = out[columns.label].astype(int)
    return out


class CodeTextDataset(Dataset):
    """Dataset returning raw code text plus labels/metadata.

    Tokenization is intentionally performed in the collator, not in
    ``__getitem__``.  This allows dynamic padding and avoids duplicating token
    padding logic across EXP-3/EXP-4/EXP-5.
    """

    def __init__(self, frame: pd.DataFrame, columns: DatasetColumns):
        self.columns = columns
        self.frame = prepare_code_frame(frame, columns)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.frame.iloc[idx]
        return {
            "code": row[self.columns.code],
            "label": int(row[self.columns.label]),
            "source_row_id": int(row[self.columns.source_id]),
            "project": str(row[self.columns.project]),
        }


class TransformerBatchCollator:
    """Batch tokenizer/collator shared by all CS2 transformer experiments."""

    def __init__(self, tokenizer, max_length: int = 512, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        texts = [str(item["code"]) for item in examples]
        labels = torch.tensor([int(item["label"]) for item in examples], dtype=torch.long)
        source_row_ids = torch.tensor([int(item["source_row_id"]) for item in examples], dtype=torch.long)
        projects = [str(item["project"]) for item in examples]

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        tokenized["labels"] = labels
        tokenized["source_row_id"] = source_row_ids
        tokenized["project"] = projects
        return tokenized


def create_dataloader(
    frame: pd.DataFrame,
    tokenizer,
    *,
    columns: DatasetColumns,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_to_multiple_of: Optional[int] = 8,
) -> DataLoader:
    """Create a shared CS2 DataLoader with dynamic padding."""

    dataset = CodeTextDataset(frame, columns)
    collator = TransformerBatchCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collator,
        drop_last=False,
    )


def get_pos_weight(frame: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    """Return BCEWithLogits positive-class weight: negatives / positives."""

    labels = frame[label_column].astype(int).to_numpy()
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def get_class_weights(frame: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    """Backward-compatible alias used by older EXP-4 code."""

    return get_pos_weight(frame, label_column=label_column)


def sample_with_optional_positive_fraction(
    frame: pd.DataFrame,
    *,
    n_rows: int,
    label_column: str = "label",
    positive_fraction: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows reproducibly, optionally enriching positives.

    This is intended for smoke/pilot experiments only.  Reportable full results
    should use the natural project-disjoint folds, not enriched samples.
    """

    n_rows = min(int(n_rows), len(frame))
    if n_rows <= 0:
        return frame.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)

    if positive_fraction is None:
        return frame.sample(n=n_rows, random_state=random_state).reset_index(drop=True)

    positive_fraction = float(positive_fraction)
    positive_fraction = max(0.0, min(1.0, positive_fraction))

    pos = frame[frame[label_column].astype(int) == 1]
    neg = frame[frame[label_column].astype(int) == 0]

    target_pos = min(len(pos), int(round(n_rows * positive_fraction)))
    target_neg = min(len(neg), n_rows - target_pos)

    # If one class is too small, fill the remaining rows from the other class.
    remaining = n_rows - (target_pos + target_neg)
    if remaining > 0:
        extra_pos = min(len(pos) - target_pos, remaining)
        target_pos += extra_pos
        remaining -= extra_pos
    if remaining > 0:
        extra_neg = min(len(neg) - target_neg, remaining)
        target_neg += extra_neg

    parts = []
    if target_pos > 0:
        parts.append(pos.sample(n=target_pos, random_state=random_state))
    if target_neg > 0:
        parts.append(neg.sample(n=target_neg, random_state=random_state + 1))

    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state + 2)
    return sampled.reset_index(drop=True)


def make_project_disjoint_threshold_split(
    train_frame: pd.DataFrame,
    *,
    project_column: str = "project",
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split an outer-train frame into fit and threshold subsets by project.

    The threshold subset is used only for threshold selection.  It is not the
    outer validation fold and it is never the frozen holdout.
    """

    if train_frame[project_column].nunique() < 2:
        # Very rare fallback: if there is only one project, row split is the
        # only possible option.  This should not happen for normal folds.
        threshold_frame = train_frame.sample(frac=validation_fraction, random_state=random_state)
        fit_frame = train_frame.drop(index=threshold_frame.index)
        return fit_frame.reset_index(drop=True), threshold_frame.reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(validation_fraction),
        random_state=int(random_state),
    )
    indices = np.arange(len(train_frame))
    groups = train_frame[project_column].astype(str).to_numpy()
    fit_idx, threshold_idx = next(splitter.split(indices, groups=groups))

    fit_frame = train_frame.iloc[fit_idx].reset_index(drop=True)
    threshold_frame = train_frame.iloc[threshold_idx].reset_index(drop=True)
    return fit_frame, threshold_frame
