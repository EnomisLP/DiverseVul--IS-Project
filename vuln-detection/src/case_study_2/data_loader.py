"""Shared dataset and dataloader utilities for Case Study 2.

This file is intentionally reusable by EXP-3 linear probe, EXP-4 LoRA, and
EXP-5 HEFT/ReFT.  It does not perform any split logic; notebooks/experiment
modules should split first, then call these utilities on the training or
validation partition.

Key design choices:
- deterministic per-function empty-code sentinel;
- configurable code column (`normalized_code` by default);
- fixed tokenizer supplied by the caller;
- no label/statistical information is used during tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


EMPTY_CODE_SENTINEL = "EMPTY_ABSTRACTED_CODE_SAMPLE"


@dataclass(frozen=True)
class TokenizationConfig:
    code_column: str = "normalized_code"
    label_column: str = "label"
    max_length: int = 512
    padding: str = "max_length"


class DiverseVulDataset(Dataset):
    """PyTorch dataset for function-level vulnerability classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        config: Optional[TokenizationConfig] = None,
        *,
        max_length: Optional[int] = None,
        code_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ):
        self.config = config or TokenizationConfig()
        if max_length is not None or code_column is not None or label_column is not None:
            self.config = TokenizationConfig(
                code_column=code_column or self.config.code_column,
                label_column=label_column or self.config.label_column,
                max_length=max_length or self.config.max_length,
                padding=self.config.padding,
            )

        self.df = dataframe.copy().reset_index(drop=True)
        self.tokenizer = tokenizer

        required = [self.config.code_column, self.config.label_column]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise KeyError(f"DiverseVulDataset missing required columns: {missing}")

        self.df[self.config.code_column] = self.df[self.config.code_column].fillna("").astype(str)
        empty_mask = self.df[self.config.code_column].str.strip().eq("")
        if empty_mask.any():
            self.df.loc[empty_mask, self.config.code_column] = EMPTY_CODE_SENTINEL

        self.labels = self.df[self.config.label_column].values.astype(np.float32)
        self.codes = self.df[self.config.code_column].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.codes[idx],
            truncation=True,
            max_length=self.config.max_length,
            padding=self.config.padding,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def get_pos_weight(dataframe: pd.DataFrame, label_column: str = "label") -> torch.Tensor:
    """Return BCEWithLogitsLoss-compatible positive-class weight."""

    labels = dataframe[label_column].values.astype(int)
    neg_counts = int(np.sum(labels == 0))
    pos_counts = int(np.sum(labels == 1))
    if pos_counts == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([neg_counts / pos_counts], dtype=torch.float32)


# Backwards-compatible alias used by your earlier exp3 code.
get_class_weights = get_pos_weight


def create_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    shuffle: bool = True,
    code_column: str = "normalized_code",
    label_column: str = "label",
    num_workers: int = 0,
    pin_memory: bool = True,
):
    dataset = DiverseVulDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        config=TokenizationConfig(
            code_column=code_column,
            label_column=label_column,
            max_length=max_length,
            padding="max_length",
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
