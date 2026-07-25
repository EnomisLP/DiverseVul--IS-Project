from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


EMPTY_CODE_SENTINEL = "EMPTY_CODE_SAMPLE"


class CodeTextDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        code_column: str = "normalized_code",
        label_column: str = "label",
        source_id_column: str = "source_row_id",
        project_column: str = "project",
    ) -> None:
        self.df = dataframe.copy().reset_index(drop=True)
        self.code_column = code_column
        self.label_column = label_column
        self.source_id_column = source_id_column
        self.project_column = project_column

        self.df[self.code_column] = self.df[self.code_column].fillna("").astype(str)
        empty_mask = self.df[self.code_column].str.strip().eq("")
        if empty_mask.any():
            self.df.loc[empty_mask, self.code_column] = EMPTY_CODE_SENTINEL

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "code": str(row[self.code_column]),
            "label": int(row[self.label_column]),
            "source_row_id": int(row[self.source_id_column]),
            "project": str(row[self.project_column]),
        }


@dataclass
class TransformerBatchCollator:
    tokenizer: Any
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [feature["code"] for feature in features]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = torch.tensor([feature["label"] for feature in features], dtype=torch.float32)
        source_row_ids = torch.tensor([feature["source_row_id"] for feature in features], dtype=torch.long)
        projects = [feature["project"] for feature in features]

        enc["labels"] = labels
        enc["label"] = labels
        enc["source_row_id"] = source_row_ids
        enc["project"] = projects
        return enc


def create_dataloader(
    dataframe: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = 16,
    max_length: int = 512,
    shuffle: bool = False,
    code_column: str = "normalized_code",
    label_column: str = "label",
    source_id_column: str = "source_row_id",
    project_column: str = "project",
    num_workers: int = 0,
) -> DataLoader:
    dataset = CodeTextDataset(
        dataframe=dataframe,
        code_column=code_column,
        label_column=label_column,
        source_id_column=source_id_column,
        project_column=project_column,
    )
    collator = TransformerBatchCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def get_pos_weight(dataframe: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    y = dataframe[label_column].astype(int).values
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([neg / pos], dtype=torch.float32)


def get_class_weights(dataframe: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    return get_pos_weight(dataframe, label_column=label_column)


def sample_with_optional_positive_fraction(
    frame: pd.DataFrame,
    n_rows: int,
    label_column: str = "label",
    positive_fraction: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_rows is None or n_rows <= 0 or len(frame) <= n_rows:
        return frame.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)

    if positive_fraction is None:
        indices = rng.choice(frame.index.to_numpy(), size=n_rows, replace=False)
        return frame.loc[indices].copy().reset_index(drop=True)

    positives = frame[frame[label_column].astype(int) == 1]
    negatives = frame[frame[label_column].astype(int) == 0]

    n_pos = min(len(positives), max(1, int(round(n_rows * positive_fraction))))
    n_neg = min(len(negatives), n_rows - n_pos)

    pos_idx = rng.choice(positives.index.to_numpy(), size=n_pos, replace=False) if n_pos else []
    neg_idx = rng.choice(negatives.index.to_numpy(), size=n_neg, replace=False) if n_neg else []
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)

    return frame.loc[idx].copy().reset_index(drop=True)


def make_project_disjoint_threshold_split(
    train_frame: pd.DataFrame,
    threshold_fraction: float = 0.20,
    project_column: str = "project",
    label_column: str = "label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    projects = train_frame[[project_column, label_column]].groupby(project_column)[label_column].agg(["count", "sum"])
    project_names = projects.index.to_numpy()

    rng = np.random.default_rng(random_state)
    shuffled = project_names.copy()
    rng.shuffle(shuffled)

    target_rows = int(round(len(train_frame) * threshold_fraction))
    selected = []
    count = 0

    for project in shuffled:
        selected.append(project)
        count += int(projects.loc[project, "count"])
        if count >= target_rows:
            break

    selected = set(selected)
    threshold_mask = train_frame[project_column].isin(selected)
    threshold_frame = train_frame[threshold_mask].copy().reset_index(drop=True)
    fit_frame = train_frame[~threshold_mask].copy().reset_index(drop=True)

    if (
        fit_frame[label_column].sum() == 0
        or threshold_frame[label_column].sum() == 0
        or len(fit_frame) == 0
        or len(threshold_frame) == 0
    ):
        shuffled_rows = train_frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        cut = max(1, int(round(len(shuffled_rows) * (1.0 - threshold_fraction))))
        fit_frame = shuffled_rows.iloc[:cut].copy().reset_index(drop=True)
        threshold_frame = shuffled_rows.iloc[cut:].copy().reset_index(drop=True)

    return fit_frame, threshold_frame