from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


EMPTY_CODE_SENTINEL = "EMPTY_CODE_SAMPLE"


class CodeTextDataset(Dataset):
    """PyTorch Dataset yielding raw code text plus label/id/project for one row."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        code_column: str = "normalized_code",
        label_column: str = "label",
        source_id_column: str = "source_row_id",
        project_column: str = "project",
    ) -> None:
        """Copy the frame and replace empty code with a sentinel token."""
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
        """Return the number of rows."""
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one row as a plain dict."""
        row = self.df.iloc[idx]
        return {
            "code": str(row[self.code_column]),
            "label": int(row[self.label_column]),
            "source_row_id": int(row[self.source_id_column]),
            "project": str(row[self.project_column]),
        }


@dataclass
class TransformerBatchCollator:
    """Tokenize a batch of raw-code dicts into padded model inputs."""

    tokenizer: Any
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenize and collate a list of row dicts into one batch."""
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
    """Build a DataLoader that tokenizes code rows on the fly."""
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
    """Compute the negative/positive ratio for BCEWithLogitsLoss's pos_weight."""
    y = dataframe[label_column].astype(int).values
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([neg / pos], dtype=torch.float32)


def get_class_weights(dataframe: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    """Alias for get_pos_weight, used as the BCE positive-class weight."""
    return get_pos_weight(dataframe, label_column=label_column)

