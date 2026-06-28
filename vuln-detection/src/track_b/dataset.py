"""
PyTorch Dataset for Track B — tokenises C/C++ function strings for NeoBERT.

Design notes:
  - Tokenization is performed lazily (per __getitem__) by default, which is
    memory-efficient for 4096-token sequences.  An eager pre-tokenize path
    is available for small folds or when speed matters more than RAM.
  - max_seq_length is read from cfg.model.max_seq_length (default 4096).
  - Truncation strategy: longest-first (single sequence, so right-side
    truncation). Long C/C++ functions that exceed 4096 tokens are truncated
    at the tail — the function signature and early variable declarations,
    which carry the most vulnerability signal, are preserved.
  - No padding is applied at the Dataset level; padding is deferred to the
    DataLoader collate function (collate_fn) so variable-length sequences
    within a batch are padded to the batch maximum, not the global maximum.
    This is critical for memory efficiency with 4096-token sequences.
  - Labels are returned as torch.long tensors (required by CrossEntropyLoss).
  - Input IDs and attention masks are returned as torch.long tensors.

Public API:
  - VulnDataset            : main torch.utils.data.Dataset subclass
  - make_collate_fn()      : returns a DataLoader-compatible collate function
                             that left-pads or right-pads to batch maximum
  - build_dataloader()     : convenience factory wrapping DataLoader
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils import get_logger

_log = get_logger(__name__)

# Default model hub path — overridden by cfg.model.name
_DEFAULT_MODEL_NAME = "chandar-lab/NeoBERT"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VulnDataset(Dataset):
    """PyTorch Dataset for binary vulnerability classification with NeoBERT.

    Each item returns a dict with keys:
      - ``input_ids``      : LongTensor of shape ``(seq_len,)``
      - ``attention_mask`` : LongTensor of shape ``(seq_len,)``
      - ``label``          : LongTensor scalar (0 or 1)

    Sequence length varies per sample (no dataset-level padding); the
    collate function handles batch-level padding.

    Args:
        texts:          List of raw C/C++ function strings.
        labels:         Corresponding binary labels (0 or 1).
        tokenizer:      Fitted HuggingFace tokenizer instance.
        max_seq_length: Maximum number of tokens (default 4096).
        truncation:     Whether to truncate sequences exceeding
                        *max_seq_length* (default ``True``).
        eager:          If ``True``, tokenize all samples at construction
                        time and cache results.  Faster iteration but uses
                        more memory (default ``False``).
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int = 4096,
        truncation: bool = True,
        eager: bool = False,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length "
                f"(got {len(texts)} and {len(labels)})."
            )

        self.texts          = list(texts)
        self.labels         = list(labels)
        self.tokenizer      = tokenizer
        self.max_seq_length = max_seq_length
        self.truncation     = truncation

        self._cache: Optional[List[Dict[str, torch.Tensor]]] = None

        if eager:
            _log.info(
                "Eager tokenization: processing %d samples (max_seq_length=%d) …",
                len(self.texts), self.max_seq_length,
            )
            self._cache = [self._tokenize(i) for i in range(len(self.texts))]
            _log.info("Eager tokenization complete.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._tokenize(idx)

    # ------------------------------------------------------------------
    # Internal tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize a single sample and return tensor dict."""
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_length,
            truncation=self.truncation,
            padding=False,          # deferred to collate_fn
            return_attention_mask=True,
            return_token_type_ids=False,  # NeoBERT has no token type embeddings
        )

        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "label":          torch.tensor(self.labels[idx],           dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    def token_length_stats(self, sample_size: Optional[int] = 1000) -> Dict[str, float]:
        """Compute token length statistics over a random sample of the dataset.

        Useful for logging before training to check how many sequences are
        being truncated.

        Args:
            sample_size: Number of samples to inspect.  ``None`` = all.

        Returns:
            Dict with ``mean``, ``median``, ``p95``, ``max``, ``pct_truncated``.
        """
        n = len(self)
        indices = (
            np.random.choice(n, size=min(sample_size, n), replace=False)
            if sample_size is not None
            else np.arange(n)
        )

        lengths = []
        truncated = 0
        for i in indices:
            enc = self.tokenizer(
                self.texts[i],
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            raw_len = len(enc["input_ids"])
            lengths.append(min(raw_len, self.max_seq_length))
            if raw_len > self.max_seq_length:
                truncated += 1

        lengths_arr = np.array(lengths)
        stats = {
            "mean":          float(lengths_arr.mean()),
            "median":        float(np.median(lengths_arr)),
            "p95":           float(np.percentile(lengths_arr, 95)),
            "max":           float(lengths_arr.max()),
            "pct_truncated": float(100 * truncated / len(indices)),
        }
        _log.info(
            "Token lengths (n=%d) — mean=%.0f  median=%.0f  "
            "p95=%.0f  max=%.0f  truncated=%.1f%%",
            len(indices),
            stats["mean"], stats["median"],
            stats["p95"],  stats["max"], stats["pct_truncated"],
        )
        return stats


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    """Load the NeoBERT tokenizer from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier
                    (default ``"chandar-lab/NeoBERT"``).
        use_fast:   Use the Rust-backed fast tokenizer (default ``True``).

    Returns:
        Loaded :class:`transformers.PreTrainedTokenizerBase` instance.
    """
    _log.info("Loading tokenizer: %s (use_fast=%s)", model_name, use_fast)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    _log.info(
        "Tokenizer loaded — vocab_size=%d  model_max_length=%s",
        tokenizer.vocab_size,
        getattr(tokenizer, "model_max_length", "unset"),
    )
    return tokenizer


def load_tokenizer_from_config(cfg) -> PreTrainedTokenizerBase:
    """Load the tokenizer using parameters from the Track B config namespace.

    Reads ``cfg.model.name`` and ``cfg.model.max_seq_length``.

    Args:
        cfg: Config namespace from ``load_config("configs/track_b.yaml")``.

    Returns:
        Loaded tokenizer.
    """
    return load_tokenizer(model_name=cfg.model.name)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def make_collate_fn(
    pad_token_id: int,
    *,
    padding_side: str = "right",
) -> callable:
    """Return a collate function that pads a batch to its longest sequence.

    Padding is applied at batch level (not dataset level) to minimise wasted
    compute on short batches.  With 4096-token max length and batch_size=8,
    dataset-level padding would waste enormous memory.

    Args:
        pad_token_id:  Token ID used for padding (from ``tokenizer.pad_token_id``).
        padding_side:  ``"right"`` (default) or ``"left"``.  NeoBERT uses
                       absolute position embeddings replaced with RoPE, so
                       right-padding is standard.

    Returns:
        Callable suitable for ``DataLoader(collate_fn=...)``.
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got '{padding_side}'.")

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            if padding_side == "right":
                padded_ids  = torch.cat([
                    item["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
                padded_mask = torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            else:  # left padding
                padded_ids  = torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                    item["input_ids"],
                ])
                padded_mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    item["attention_mask"],
                ])

            input_ids_list.append(padded_ids)
            attention_mask_list.append(padded_mask)
            labels_list.append(item["label"])

        return {
            "input_ids":      torch.stack(input_ids_list),       # (B, max_len)
            "attention_mask": torch.stack(attention_mask_list),  # (B, max_len)
            "labels":         torch.stack(labels_list),          # (B,)
        }

    return collate_fn


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: VulnDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    padding_side: str = "right",
) -> DataLoader:
    """Build a :class:`torch.utils.data.DataLoader` for a :class:`VulnDataset`.

    Attaches the batch-level padding collate function automatically.

    Args:
        dataset:      A :class:`VulnDataset` instance.
        tokenizer:    The tokenizer used to build *dataset* (supplies
                      ``pad_token_id``).
        batch_size:   Samples per batch (default 8, matching track_b.yaml).
        shuffle:      Shuffle the dataset each epoch.  Use ``True`` for
                      training, ``False`` for validation / holdout.
        num_workers:  Dataloader worker processes.  Set to 0 in Colab
                      (multiprocessing is unreliable there).
        pin_memory:   Pin host memory for faster GPU transfer (default ``True``).
        padding_side: ``"right"`` (default) or ``"left"``.

    Returns:
        Configured :class:`torch.utils.data.DataLoader`.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Some tokenizers omit pad_token — fall back to eos_token_id
        pad_token_id = tokenizer.eos_token_id
        _log.warning(
            "tokenizer.pad_token_id is None — using eos_token_id (%d) for padding.",
            pad_token_id,
        )

    collate_fn = make_collate_fn(pad_token_id, padding_side=padding_side)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )

    _log.info(
        "DataLoader built — %d samples  batch_size=%d  shuffle=%s  "
        "num_workers=%d  pin_memory=%s",
        len(dataset), batch_size, shuffle,
        num_workers, pin_memory and torch.cuda.is_available(),
    )
    return loader


# ---------------------------------------------------------------------------
# Convenience builder (used by track_b/train.py)
# ---------------------------------------------------------------------------

def build_fold_dataloaders(
    texts_train: List[str],
    labels_train: Sequence[int],
    texts_val: List[str],
    labels_val: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    cfg,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for one CV fold.

    Args:
        texts_train:  Training fold raw texts.
        labels_train: Training fold labels.
        texts_val:    Validation fold raw texts.
        labels_val:   Validation fold labels.
        tokenizer:    Fitted tokenizer.
        cfg:          Track B config namespace
                      (``load_config("configs/track_b.yaml")``).

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    max_seq = cfg.model.max_seq_length
    bs      = cfg.training.batch_size

    train_ds = VulnDataset(
        texts_train, labels_train, tokenizer, max_seq_length=max_seq
    )
    val_ds = VulnDataset(
        texts_val, labels_val, tokenizer, max_seq_length=max_seq
    )

    train_loader = build_dataloader(
        train_ds, tokenizer, batch_size=bs, shuffle=True,  num_workers=0
    )
    val_loader = build_dataloader(
        val_ds,   tokenizer, batch_size=bs, shuffle=False, num_workers=0
    )

    return train_loader, val_loader