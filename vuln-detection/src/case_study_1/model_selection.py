"""
Model selection across Case Study 1 experiments, using pooled 5-fold CV
metrics computed on the dev partition only. The holdout is never involved.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelCandidate:
    name: str          # e.g. "cs1_exp0_lr", "cs1_exp1_rf"
    cv_pooled_metrics: Mapping[str, object]
    final_holdout_runner: str  # e.g. "run_exp0_final_holdout"


def select_best_cs1_model(
    candidates: list[ModelCandidate],
    metric: str = "average_precision_pr_auc",
) -> ModelCandidate:
    """Pick the candidate with the highest pooled CV metric (dev only)."""
    if not candidates:
        raise ValueError("No candidates provided.")

    scored = [(float(c.cv_pooled_metrics[metric]), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    print("CV-based model comparison (dev partition, pooled OOF):")
    for score, candidate in scored:
        print(f"  {candidate.name:>15s}: {metric}={score:.4f}")

    best_score, best_candidate = scored[0]
    print(f"\nSelected: {best_candidate.name} ({metric}={best_score:.4f})")
    return best_candidate