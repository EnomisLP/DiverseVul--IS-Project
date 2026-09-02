from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_downsample(
    frame: pd.DataFrame,
    target_n: int,
    label_column: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return an exact target_n-row random subsample of frame, stratified by label_column."""
    if target_n >= len(frame):
        return frame.reset_index(drop=True)
    sampled, _ = train_test_split(
        frame,
        train_size=target_n,
        stratify=frame[label_column],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)
