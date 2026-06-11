"""
exp0_baseline.py
================
EXP_0 — Zero-Shot & Few-Shot Baseline
Base model: Qwen2.5-Coder-1.5B (no fine-tuning, inference only)

Evaluates the pre-trained model on the test split using:
  - Zero-shot prompting
  - Few-shot prompting (k=2, k=4 by default)

Metrics reported: Precision, Recall, F1-Score, PR-AUC, Accuracy
Results saved to: exp0_outputs/exp0_results.json
                  exp0_outputs/exp0_metrics.csv

Usage:
    python exp0_baseline.py \
        --test_path  data/splits/test.jsonl \
        --model_id   Qwen/Qwen2.5-Coder-1.5B-Instruct \
        [--train_path data/splits/train.jsonl]  # needed for few-shot examples
        [--shots 0 2 4]                          # which k values to run
        [--max_new_tokens 16]
        [--batch_size 8]
        [--device cuda]
        [--max_samples 1000]                     # subsample test set (speed)
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------
def _import_torch():
    import torch
    return torch

def _import_transformers():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoTokenizer, AutoModelForCausalLM

def _import_sklearn_metrics():
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        average_precision_score, accuracy_score,
        confusion_matrix,
    )
    return precision_score, recall_score, f1_score, average_precision_score, accuracy_score, confusion_matrix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_COL = "target"
FUNC_COL  = "func"

# Tokens the model is expected to generate for each class
VULNERABLE_TOKENS = ["vulnerable", "Vulnerable", "VULNERABLE", "1", "yes", "Yes"]
SECURE_TOKENS     = ["secure", "Secure", "SECURE", "0", "no", "No"]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a security-focused code analysis assistant. "
    "Your task is to determine whether a given C/C++ function contains a security vulnerability. "
    "Answer with exactly one word: 'Vulnerable' if the function has a vulnerability, "
    "or 'Secure' if it does not."
)

ZERO_SHOT_TEMPLATE = """\
### Instruction:
Analyze the following C/C++ function and determine whether it contains a security vulnerability.
Answer with exactly one word: 'Vulnerable' or 'Secure'.

### Code:
{code}

### Response:"""

FEW_SHOT_EXAMPLE_TEMPLATE = """\
### Code:
{code}

### Response:
{label}"""

FEW_SHOT_HEADER = """\
### Instruction:
Analyze the following C/C++ function and determine whether it contains a security vulnerability.
Answer with exactly one word: 'Vulnerable' or 'Secure'.

Below are some examples:

"""

FEW_SHOT_QUERY_TEMPLATE = """\

Now analyze this function:

### Code:
{code}

### Response:"""


def build_zero_shot_prompt(code: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(code=code)


def build_few_shot_prompt(code: str, examples: list[dict]) -> str:
    """
    Build a k-shot prompt from a list of example dicts with keys 'func' and 'target'.
    Examples are shown in alternating secure/vulnerable order for balance.
    """
    parts = [FEW_SHOT_HEADER]
    for ex in examples:
        label_str = "Vulnerable" if ex[LABEL_COL] == 1 else "Secure"
        parts.append(FEW_SHOT_EXAMPLE_TEMPLATE.format(
            code=ex[FUNC_COL][:800],  # truncate examples to keep prompt short
            label=label_str,
        ))
        parts.append("\n---\n")
    parts.append(FEW_SHOT_QUERY_TEMPLATE.format(code=code))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Example sampler (balanced for few-shot)
# ---------------------------------------------------------------------------

def sample_few_shot_examples(
    train_records: list[dict],
    k: int,
    seed: int = 42,
    max_char_len: int = 800,
) -> list[dict]:
    """
    Sample k examples balanced between secure and vulnerable.
    Filters out very long functions to keep prompts manageable.
    """
    rng = random.Random(seed)
    secure_pool = [
        r for r in train_records
        if r[LABEL_COL] == 0 and len(r[FUNC_COL]) <= max_char_len
    ]
    vuln_pool = [
        r for r in train_records
        if r[LABEL_COL] == 1 and len(r[FUNC_COL]) <= max_char_len
    ]

    k_each = k // 2
    sampled = (
        rng.sample(secure_pool, min(k_each, len(secure_pool))) +
        rng.sample(vuln_pool,   min(k_each, len(vuln_pool)))
    )
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(response_text: str) -> tuple[int, float]:
    """
    Parse model output to a binary prediction and a soft confidence score.

    Returns:
        (prediction, confidence)
        prediction : 0 (secure) or 1 (vulnerable)
        confidence : float in [0, 1] representing P(vulnerable)
    """
    text = response_text.strip().lower()

    # Check for vulnerable tokens first (positive class)
    for tok in VULNERABLE_TOKENS:
        if tok.lower() in text[:30]:
            return 1, 0.9

    for tok in SECURE_TOKENS:
        if tok.lower() in text[:30]:
            return 0, 0.1

    # Fallback: keyword search in full response
    vuln_count = sum(text.count(t.lower()) for t in VULNERABLE_TOKENS)
    sec_count  = sum(text.count(t.lower()) for t in SECURE_TOKENS)

    if vuln_count > sec_count:
        return 1, 0.7
    elif sec_count > vuln_count:
        return 0, 0.3
    else:
        # Ambiguous → default to secure (conservative FP avoidance)
        return 0, 0.5


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def run_inference(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 16,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[str]:
    """
    Run batched inference. Returns list of generated response strings.
    """
    torch = _import_torch()
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            new_tokens = out[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text)

        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{len(prompts)}] samples processed...")

    return responses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float],
    split_name: str = "",
) -> dict:
    (precision_score, recall_score, f1_score,
     average_precision_score, accuracy_score, confusion_matrix) = _import_sklearn_metrics()

    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    pr_auc     = average_precision_score(y_true, y_scores)
    accuracy   = accuracy_score(y_true, y_pred)
    cm         = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n  {'─'*40}")
    print(f"  Results [{split_name}]")
    print(f"  {'─'*40}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}  ← primary metric")
    print(f"  PR-AUC    : {pr_auc:.4f}  ← primary metric")
    print(f"  Confusion matrix (TN FP / FN TP):")
    print(f"    {cm[0]}  (TN={cm[0][0]}, FP={cm[0][1]})")
    print(f"    {cm[1]}  (FN={cm[1][0]}, TP={cm[1][1]})")

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "pr_auc":    round(pr_auc,    4),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp0(
    test_path:      str,
    model_id:       str,
    train_path:     str | None = None,
    shots:          list[int]  = (0, 2, 4),
    max_new_tokens: int        = 16,
    batch_size:     int        = 8,
    device:         str        = "cuda",
    max_samples:    int | None = None,
    output_dir:     str        = "exp0_outputs",
    seed:           int        = 42,
) -> None:
    torch = _import_torch()
    AutoTokenizer, AutoModelForCausalLM = _import_transformers()

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("EXP_0 — Zero/Few-Shot Baseline")
    print(f"Model  : {model_id}")
    print(f"Device : {device}")
    print(f"Shots  : {list(shots)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[data] Loading test set...")
    with open(test_path) as f:
        test_records = [json.loads(l) for l in f if l.strip()]
    print(f"       Test set: {len(test_records):,} samples")

    # Optionally subsample for speed
    if max_samples and max_samples < len(test_records):
        rng = random.Random(seed)
        # Stratified subsample
        vuln  = [r for r in test_records if r[LABEL_COL] == 1]
        sec   = [r for r in test_records if r[LABEL_COL] == 0]
        n_v   = int(max_samples * 0.053)   # preserve ~5.3% ratio
        n_s   = max_samples - n_v
        test_records = rng.sample(vuln, min(n_v, len(vuln))) + rng.sample(sec, min(n_s, len(sec)))
        rng.shuffle(test_records)
        print(f"       Subsampled to {len(test_records):,} (vuln={n_v}, secure={n_s})")

    train_records = []
    if train_path and max(shots) > 0:
        print("[data] Loading train set for few-shot examples...")
        with open(train_path) as f:
            train_records = [json.loads(l) for l in f if l.strip()]
        print(f"       Train set: {len(train_records):,} samples")

    y_true = [r[LABEL_COL] for r in test_records]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[model] Loading '{model_id}'...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] Loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Run each shot configuration
    # ------------------------------------------------------------------
    all_results = {}

    for k in shots:
        print(f"\n{'='*60}")
        print(f"  Running {k}-shot evaluation...")
        print(f"{'='*60}")

        # Build prompts
        if k == 0:
            prompts = [build_zero_shot_prompt(r[FUNC_COL]) for r in test_records]
        else:
            examples = sample_few_shot_examples(train_records, k=k, seed=seed)
            print(f"  Few-shot examples selected: {len(examples)} "
                  f"(vuln={sum(e[LABEL_COL]==1 for e in examples)}, "
                  f"secure={sum(e[LABEL_COL]==0 for e in examples)})")
            prompts = [
                build_few_shot_prompt(r[FUNC_COL], examples)
                for r in test_records
            ]

        # Inference
        t1 = time.time()
        raw_responses = run_inference(
            prompts, tokenizer, model,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
        )
        elapsed = time.time() - t1
        print(f"  Inference time: {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/sample)")

        # Parse responses
        y_pred   = []
        y_scores = []
        for resp in raw_responses:
            pred, score = parse_response(resp)
            y_pred.append(pred)
            y_scores.append(score)

        # Metrics
        metrics = compute_metrics(y_true, y_pred, y_scores, split_name=f"{k}-shot")
        metrics["inference_time_s"] = round(elapsed, 2)
        metrics["samples_per_sec"]  = round(len(prompts) / elapsed, 2)

        all_results[f"{k}_shot"] = metrics

        # Save per-sample predictions for error analysis
        preds_path = os.path.join(output_dir, f"predictions_{k}shot.jsonl")
        with open(preds_path, "w") as fh:
            for rec, resp, pred, score in zip(test_records, raw_responses, y_pred, y_scores):
                fh.write(json.dumps({
                    "func":       rec[FUNC_COL][:200] + "...",  # truncated for storage
                    "true_label": rec[LABEL_COL],
                    "pred_label": pred,
                    "confidence": score,
                    "raw_response": resp,
                    "correct": int(pred == rec[LABEL_COL]),
                }) + "\n")
        print(f"  Per-sample predictions → '{preds_path}'")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  EXP_0 — Summary")
    print(f"{'='*60}")
    header = f"  {'Shot':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'PR-AUC':>6}"
    print(header)
    print(f"  {'-'*50}")
    for k in shots:
        m = all_results[f"{k}_shot"]
        print(f"  {k:>5}k  {m['precision']:>9.4f}  {m['recall']:>6.4f}  "
              f"{m['f1']:>6.4f}  {m['pr_auc']:>6.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "exp0_results.json")
    final = {
        "experiment": "EXP_0",
        "model": model_id,
        "test_samples": len(test_records),
        "results": all_results,
    }
    with open(results_path, "w") as fh:
        json.dump(final, fh, indent=2)
    print(f"\n[save] Results → '{results_path}'")

    # Save CSV summary for easy comparison with later experiments
    csv_path = os.path.join(output_dir, "exp0_metrics.csv")
    with open(csv_path, "w") as fh:
        fh.write("experiment,model,shots,accuracy,precision,recall,f1,pr_auc\n")
        for k in shots:
            m = all_results[f"{k}_shot"]
            fh.write(
                f"EXP_0,{model_id},{k},"
                f"{m['accuracy']},{m['precision']},{m['recall']},"
                f"{m['f1']},{m['pr_auc']}\n"
            )
    print(f"[save] CSV summary → '{csv_path}'")
    print("\n[done] EXP_0 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP_0: Zero/Few-Shot Baseline")
    parser.add_argument("--test_path",      required=True,
                        help="Path to test.jsonl (from preprocess script).")
    parser.add_argument("--model_id",       default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model ID.")
    parser.add_argument("--train_path",     default=None,
                        help="Path to train.jsonl (required for k>0 few-shot).")
    parser.add_argument("--shots",          nargs="+", type=int, default=[0, 2, 4],
                        help="List of k values to evaluate (default: 0 2 4).")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="Max tokens to generate per sample (default: 16).")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Inference batch size (default: 8).")
    parser.add_argument("--device",         default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to run inference on (default: cuda).")
    parser.add_argument("--max_samples",    type=int, default=None,
                        help="Subsample test set for speed (default: use all).")
    parser.add_argument("--output_dir",     default="exp0_outputs",
                        help="Directory to save results (default: exp0_outputs/).")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    run_exp0(
        test_path      = args.test_path,
        model_id       = args.model_id,
        train_path     = args.train_path,
        shots          = args.shots,
        max_new_tokens = args.max_new_tokens,
        batch_size     = args.batch_size,
        device         = args.device,
        max_samples    = args.max_samples,
        output_dir     = args.output_dir,
        seed           = args.seed,
    )