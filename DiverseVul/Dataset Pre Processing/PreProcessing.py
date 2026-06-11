"""
preprocess_stratified_split.py
================================
Data preprocessing pipeline for RDiverseVul with stratified train/val/test split.

Strategy chosen: Stratified Splitting (80/10/10)
- Preserves the natural 5.3% / 94.7% class ratio in every split
- No oversampling or undersampling applied
- Class imbalance handled downstream via weighted loss / Focal Loss

Usage:
    python preprocess_stratified_split.py \
        --input  path/to/rdiversevul.jsonl \
        --output path/to/splits/ \
        [--max_length 2048] \
        [--seed 42]

Expected input format (JSONL, one record per line):
    {"func": "int foo(...) {...}", "target": 0, "cwe": ["CWE-787"], ...}

Output:
    splits/
      train.jsonl
      val.jsonl
      test.jsonl
      split_stats.json
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so the script can be inspected without them
# ---------------------------------------------------------------------------
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

LABEL_COL   = "target"   # 0 = secure, 1 = vulnerable
FUNC_COL    = "func"
CWE_COL     = "cwe"


# ---------------------------------------------------------------------------
# 1. I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[load]  Loaded {len(records):,} records from '{path}'")
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save]  Wrote {len(records):,} records → '{path}'")


# ---------------------------------------------------------------------------
# 2. Deduplication (exact, by function body)
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    """
    Remove exact duplicate functions (same source text).
    When duplicates conflict in label (same code marked both secure AND
    vulnerable), keep the vulnerable label to be conservative.
    """
    seen: dict[str, dict] = {}
    conflicts = 0

    for rec in records:
        key = rec[FUNC_COL].strip()
        if key not in seen:
            seen[key] = rec
        else:
            existing_label = seen[key][LABEL_COL]
            new_label      = rec[LABEL_COL]
            if existing_label != new_label:
                conflicts += 1
                # Conservative: keep vulnerable label
                if new_label == 1:
                    seen[key] = rec

    deduped = list(seen.values())
    removed = len(records) - len(deduped)
    print(f"[dedup] Removed {removed:,} exact duplicates "
          f"({conflicts} label conflicts resolved → kept 'vulnerable').")
    print(f"[dedup] Remaining: {len(deduped):,} records.")
    return deduped


# ---------------------------------------------------------------------------
# 3. Token-length filtering
# ---------------------------------------------------------------------------

def char_length_proxy(text: str) -> int:
    """
    Cheap proxy for token count: character count.
    Qwen2.5-Coder tokeniser averages ~4 chars/token for C/C++.
    Multiply max_tokens by 4 to get the char cutoff.
    """
    return len(text)


def filter_by_length(records: list[dict], max_tokens: int) -> list[dict]:
    """
    Drop functions whose char-proxy exceeds max_tokens * 4.
    These are the extreme outliers (sparse tail > 5 000 tokens in the EDA).
    Truncation is intentionally NOT done here; it is left to the tokeniser
    so that the model receives a proper [BOS] … [EOS] sequence.
    """
    max_chars = max_tokens * 4
    before = len(records)
    records = [r for r in records if char_length_proxy(r[FUNC_COL]) <= max_chars]
    dropped = before - len(records)
    print(f"[len]   Dropped {dropped:,} samples exceeding {max_tokens} tokens "
          f"(char proxy > {max_chars:,}). Remaining: {len(records):,}.")
    return records


# ---------------------------------------------------------------------------
# 4. Stratified split (sklearn or fallback)
# ---------------------------------------------------------------------------

def stratified_split_sklearn(
    records: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """80 / 10 / 10 stratified split using sklearn."""
    labels = [r[LABEL_COL] for r in records]

    train_recs, temp_recs, _, temp_labels = train_test_split(
        records, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=seed,
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_recs, test_recs = train_test_split(
        temp_recs,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_recs, val_recs, test_recs


def stratified_split_manual(
    records: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Pure-Python fallback stratified split when sklearn is unavailable.
    Splits each class independently then shuffles.
    """
    rng = random.Random(seed)

    by_class: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_class[rec[LABEL_COL]].append(rec)

    train_all, val_all, test_all = [], [], []

    for label, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train_all.extend(group[:n_train])
        val_all.extend(group[n_train:n_train + n_val])
        test_all.extend(group[n_train + n_val:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)
    return train_all, val_all, test_all


def stratified_split(
    records: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    if SKLEARN_AVAILABLE:
        print("[split] Using sklearn stratified split.")
        return stratified_split_sklearn(records, seed)
    else:
        print("[split] sklearn not found — using manual stratified split.")
        return stratified_split_manual(records, seed)


# ---------------------------------------------------------------------------
# 5. Statistics & verification
# ---------------------------------------------------------------------------

def split_stats(
    train: list[dict],
    val:   list[dict],
    test:  list[dict],
) -> dict:
    """
    Compute and print class distribution for each split.
    Verifies that the imbalance ratio is preserved.
    """
    def stats(name: str, recs: list[dict]) -> dict:
        total = len(recs)
        n_vul = sum(1 for r in recs if r[LABEL_COL] == 1)
        n_sec = total - n_vul
        ratio = n_sec / n_vul if n_vul > 0 else float("inf")
        pct   = 100 * n_vul / total if total > 0 else 0
        print(
            f"  {name:6s}  total={total:>7,}  "
            f"secure={n_sec:>7,} ({100-pct:.1f}%)  "
            f"vuln={n_vul:>6,} ({pct:.1f}%)  "
            f"ratio={ratio:.1f}:1"
        )
        return {
            "total": total, "secure": n_sec, "vulnerable": n_vul,
            "vuln_pct": round(pct, 2), "imbalance_ratio": round(ratio, 2),
        }

    print("\n[stats] Split class distribution:")
    result = {
        "train": stats("train", train),
        "val":   stats("val",   val),
        "test":  stats("test",  test),
    }

    # CWE coverage check (optional, only if cwe field present)
    if CWE_COL in train[0]:
        train_cwes = set(c for r in train for c in (r[CWE_COL] or []))
        val_cwes   = set(c for r in val   for c in (r[CWE_COL] or []))
        test_cwes  = set(c for r in test  for c in (r[CWE_COL] or []))
        val_unseen  = val_cwes  - train_cwes
        test_unseen = test_cwes - train_cwes
        if val_unseen:
            print(f"  [warn] {len(val_unseen)} CWE(s) in val not seen in train: {val_unseen}")
        if test_unseen:
            print(f"  [warn] {len(test_unseen)} CWE(s) in test not seen in train: {test_unseen}")
        result["train_unique_cwes"] = len(train_cwes)
        result["val_unique_cwes"]   = len(val_cwes)
        result["test_unique_cwes"]  = len(test_cwes)

    return result


# ---------------------------------------------------------------------------
# 6. Compute class weights (for downstream loss function)
# ---------------------------------------------------------------------------

def compute_class_weights(train: list[dict]) -> dict:
    """
    Compute pos_weight for BCEWithLogitsLoss and balanced class weights.
    These values should be passed to your training script.
    """
    n_sec = sum(1 for r in train if r[LABEL_COL] == 0)
    n_vul = sum(1 for r in train if r[LABEL_COL] == 1)
    pos_weight = n_sec / n_vul  # ~17.8 before dedup/filtering
    balanced_w = {
        0: (n_sec + n_vul) / (2 * n_sec),
        1: (n_sec + n_vul) / (2 * n_vul),
    }
    print(f"\n[weights] pos_weight (BCEWithLogitsLoss) = {pos_weight:.4f}")
    print(f"[weights] balanced class weights: class-0={balanced_w[0]:.4f}, class-1={balanced_w[1]:.4f}")
    return {"pos_weight": round(pos_weight, 4), "balanced": balanced_w}


# ---------------------------------------------------------------------------
# 7. Format prompt (Qwen2.5-Coder instruction style)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Analyze the following C/C++ function and determine if it contains "
    "a security vulnerability.\n\n"
    "### Code:\n"
    "{code}\n\n"
    "### Response:\n"
)

def add_prompt_field(records: list[dict]) -> list[dict]:
    """Add a 'prompt' field with the instruction-style template."""
    for rec in records:
        rec["prompt"] = PROMPT_TEMPLATE.format(code=rec[FUNC_COL])
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_dir: str,
    max_length: int = 2048,
    seed: int = 42,
    add_prompts: bool = True,
) -> None:
    print("=" * 60)
    print("RDiverseVul — Stratified Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Load
    records = load_jsonl(input_path)
    print(f"         Raw class distribution: "
          f"vuln={sum(r[LABEL_COL]==1 for r in records):,}, "
          f"secure={sum(r[LABEL_COL]==0 for r in records):,}")

    # Step 2: Deduplicate
    records = deduplicate(records)

    # Step 3: Length filter (drop extreme outliers only)
    records = filter_by_length(records, max_tokens=max_length)

    # Step 4: Stratified split  ← class imbalance strategy chosen
    train, val, test = stratified_split(records, seed=seed)

    # Step 5: Stats & verification
    stats = split_stats(train, val, test)

    # Step 6: Class weights (saved for reference; not applied here)
    weights = compute_class_weights(train)

    # Step 7: Add prompt field
    if add_prompts:
        train = add_prompt_field(train)
        val   = add_prompt_field(val)
        test  = add_prompt_field(test)

    # Step 8: Save splits
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val,   os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test,  os.path.join(output_dir, "test.jsonl"))

    # Step 9: Save metadata
    meta = {
        "pipeline": "stratified_split",
        "seed": seed,
        "max_length_tokens": max_length,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "class_weights": weights,
        "splits": stats,
    }
    meta_path = os.path.join(output_dir, "split_stats.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[meta]  Metadata saved → '{meta_path}'")
    print("\n[done]  Preprocessing complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified preprocessing pipeline for RDiverseVul."
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to the raw RDiverseVul JSONL file."
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory where train/val/test splits will be saved."
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Max token length threshold for filtering outliers (default: 2048). "
             "Use 4096 for the 7B model."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--no_prompts", action="store_true",
        help="Skip adding the instruction-style 'prompt' field."
    )
    args = parser.parse_args()

    run_pipeline(
        input_path  = args.input,
        output_dir  = args.output,
        max_length  = args.max_length,
        seed        = args.seed,
        add_prompts = not args.no_prompts,
    )