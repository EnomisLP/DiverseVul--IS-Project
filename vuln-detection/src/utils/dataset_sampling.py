from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def cap_rows_per_group(
    frame: pd.DataFrame,
    max_rows_per_group: int,
    group_column: str = "project",
    random_state: int = 42,
) -> pd.DataFrame:
    parts = []
    for _, group_frame in frame.groupby(group_column, sort=False):
        if len(group_frame) > max_rows_per_group:
            group_frame = group_frame.sample(n=max_rows_per_group, random_state=random_state)
        parts.append(group_frame)
    return pd.concat(parts, ignore_index=True)


def stratified_downsample(
    frame: pd.DataFrame,
    target_n: int,
    label_column: str = "label",
    group_column: str = "project",
    max_rows_per_group: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    if target_n >= len(frame):
        return frame.reset_index(drop=True)

    capped_frame = cap_rows_per_group(
        frame, max_rows_per_group, group_column=group_column, random_state=random_state
    )

    if target_n >= len(capped_frame):
        return capped_frame.reset_index(drop=True)

    sampled, _ = train_test_split(
        capped_frame,
        train_size=target_n,
        stratify=capped_frame[label_column],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def summarize_group_sizes(
    frame: pd.DataFrame,
    group_column: str = "project",
) -> pd.Series:
    return frame.groupby(group_column).size().sort_values(ascending=False)
