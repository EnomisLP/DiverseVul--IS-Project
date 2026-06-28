import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ── Aesthetic config ──────────────────────────────────────────────────────────
PALETTE = {"vulnerable": "#E74C3C", "secure": "#2ECC71"}
FIG_DPI  = 150
OUT_DIR  = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load RDiverseVul JSON into a DataFrame."""
    print(f"[1/4] Loading dataset from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try common wrapper keys first
        for key in ("data", "samples", "entries", "records"):
            if key in data:
                df = pd.DataFrame(data[key])
                break
        else:
            # 👇 CHANGE THIS FALLBACK 👇
            # If the dict keys are the columns/features, load it directly
            df = pd.DataFrame.from_dict(data)
    else:
        raise ValueError("Unexpected JSON structure.")

    print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"    Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def inspect_schema(df: pd.DataFrame) -> None:
    """Print basic schema and null-value report."""
    print("\n[2/4] Schema inspection")
    print(df.dtypes.to_string())
    print("\nNull counts:")
    print(df.isnull().sum().to_string())
    print("\nSample row (first entry):")
    print(df.iloc[0].to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  CLASS IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_class_imbalance(df: pd.DataFrame, label_col: str = "target") -> None:
    """
    4.1 – Quantify vulnerable vs. non-vulnerable ratio.
    Saves: eda_outputs/class_imbalance.png
    """
    print("\n[3/4] 4.1 – Class Imbalance Analysis")

    counts = df[label_col].value_counts().sort_index()
    labels_map = {0: "Secure (0)", 1: "Vulnerable (1)"}
    counts.index = [labels_map.get(i, str(i)) for i in counts.index]

    total   = counts.sum()
    pct_vul = counts.get("Vulnerable (1)", 0) / total * 100

    print(f"    Total samples    : {total:,}")
    print(f"    Secure (0)       : {counts.get('Secure (0)', 0):,}  "
          f"({100 - pct_vul:.1f}%)")
    print(f"    Vulnerable (1)   : {counts.get('Vulnerable (1)', 0):,}  "
          f"({pct_vul:.1f}%)")
    imb_ratio = counts.get("Secure (0)", 1) / max(counts.get("Vulnerable (1)", 1), 1)
    print(f"    Imbalance ratio  : {imb_ratio:.1f}:1  →  "
          f"{'Focal Loss / weighted CE recommended' if imb_ratio > 5 else 'Mild imbalance'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Bar chart
    colors = [PALETTE["secure"], PALETTE["vulnerable"]]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.005,
                     f"{val:,}\n({val/total*100:.1f}%)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Class Distribution (absolute)", fontweight="bold")
    axes[0].set_ylabel("Number of samples")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Class Distribution (percentage)", fontweight="bold")

    plt.suptitle("4.1 – Class Imbalance Analysis – RDiverseVul",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "class_imbalance.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  TOKEN LENGTH DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_token_lengths(df: pd.DataFrame,
                           code_col: str  = "func",
                           label_col: str = "target",
                           model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
                           sample_size: int = 20_000) -> None:
    """
    4.2 – Token length distribution using the Qwen2.5-Coder tokenizer.
    Falls back to a simple whitespace estimator if transformers unavailable.
    Saves: eda_outputs/token_length_distribution.png
    """
    print("\n[4/4] 4.2 – Token Length Distribution")

    # ── Try loading the real tokenizer ────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        print(f"    Loading tokenizer: {model_name}  (this may download ~few MB)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        def tokenize(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        tokenizer_label = f"Qwen2.5-Coder tokenizer"
    except Exception as e:
        print(f"    WARNING: Could not load tokenizer ({e}). "
              f"Falling back to word-count proxy (÷ 0.75 chars/token).")
        def tokenize(text: str) -> int:
            return max(1, len(text) // 4)
        tokenizer_label = "Char-count proxy (≈ tokens)"

    # ── Sample for speed ──────────────────────────────────────────────────────
    sub = df.sample(min(sample_size, len(df)), random_state=42).copy()
    print(f"    Computing token lengths for {len(sub):,} samples …")
    tqdm.pandas(desc="    Tokenising")
    sub["n_tokens"] = sub[code_col].progress_apply(
        lambda x: tokenize(str(x)) if pd.notna(x) else 0
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    for label, name in [(0, "Secure"), (1, "Vulnerable")]:
        subset = sub[sub[label_col] == label]["n_tokens"]
        print(f"    {name:12s} — mean: {subset.mean():6.0f}  "
              f"median: {subset.median():6.0f}  "
              f"p95: {subset.quantile(0.95):6.0f}  "
              f"max: {subset.max():6.0f}")

    p95_all = sub["n_tokens"].quantile(0.95)
    p99_all = sub["n_tokens"].quantile(0.99)
    print(f"\n    p95 (all): {p95_all:.0f} tokens  →  "
          f"suggested max_seq_len ≥ {int(p95_all):,}")
    print(f"    p99 (all): {p99_all:.0f} tokens  →  "
          f"safe upper bound: {int(p99_all):,}")

    # Truncation recommendations
    for threshold in [512, 1024, 2048, 4096, 8192]:
        truncated_pct = (sub["n_tokens"] > threshold).mean() * 100
        print(f"      max_seq_len={threshold:5d}  →  "
              f"{truncated_pct:.1f}% samples truncated")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, name, color) in zip(
        axes, [(0, "Secure", PALETTE["secure"]), (1, "Vulnerable", PALETTE["vulnerable"])]
    ):
        data = sub[sub[label_col] == label]["n_tokens"]
        sns.histplot(data, bins=80, ax=ax, color=color, edgecolor="white",
                     linewidth=0.3, alpha=0.85)
        ax.axvline(data.median(), color="black", ls="--", lw=1.5,
                   label=f"Median: {data.median():.0f}")
        for thr, ls in [(512, ":"), (1024, "-."), (2048, "--"), (4096, "-")]:
            ax.axvline(thr, color="grey", ls=ls, lw=1, alpha=0.7, label=f"{thr}")
        ax.set_title(f"{name} samples", fontweight="bold")
        ax.set_xlabel(f"Token length ({tokenizer_label})")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, title="Thresholds", title_fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.suptitle("4.2 – Token Length Distribution – RDiverseVul",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "token_length_distribution.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  CWE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cwe_distribution(df: pd.DataFrame,
                              cwe_col: str   = "cwe",
                              label_col: str = "target",
                              top_n: int     = 25) -> None:
    """
    4.3 – CWE frequency distribution for vulnerable samples.
    Handles both single-value and list/JSON-encoded CWE columns.
    Saves: eda_outputs/cwe_distribution.png
    """
    print("\n[5/5] 4.3 – CWE Distribution Mapping")

    # ── Only vulnerable samples have CWE labels ───────────────────────────────
    vul = df[df[label_col] == 1].copy()

    if cwe_col not in vul.columns:
        # Try common alternative names
        for alt in ("CWE", "cwe_id", "cwe_ids", "weakness", "vuln_type"):
            if alt in vul.columns:
                cwe_col = alt
                break
        else:
            print(f"    WARNING: No CWE column found. "
                  f"Available columns: {list(df.columns)}")
            return

    # ── Parse CWE values (may be str, list, or JSON) ─────────────────────────
    cwe_counts: Counter = Counter()
    for val in vul[cwe_col].dropna():
        if isinstance(val, list):
            for v in val:
                cwe_counts[str(v).strip()] += 1
        elif isinstance(val, str):
            # Could be "[CWE-119, CWE-787]" or "CWE-119"
            val = val.strip("[]\"' ")
            for part in val.split(","):
                part = part.strip()
                if part:
                    cwe_counts[part] += 1
        else:
            cwe_counts[str(val)] += 1

    if not cwe_counts:
        print("    WARNING: CWE column is empty or unparseable.")
        return

    total_labeled = sum(cwe_counts.values())
    n_unique = len(cwe_counts)
    print(f"    Unique CWE types : {n_unique}")
    print(f"    Total CWE labels : {total_labeled:,}")
    print(f"\n    Top 10 CWEs:")
    for cwe, cnt in cwe_counts.most_common(10):
        print(f"      {cwe:20s}  {cnt:6,}  ({cnt/total_labeled*100:.1f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    top = cwe_counts.most_common(top_n)
    cwe_labels, cwe_vals = zip(*top)
    cwe_pcts = [v / total_labeled * 100 for v in cwe_vals]

    fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.38)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(cwe_labels)))
    bars = ax.barh(list(reversed(cwe_labels)), list(reversed(cwe_vals)),
                   color=list(reversed(colors)), edgecolor="white", linewidth=0.5)
    for bar, val, pct in zip(bars, reversed(cwe_vals), reversed(cwe_pcts)):
        ax.text(bar.get_width() + total_labeled * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)",
                va="center", fontsize=8)
    ax.set_xlabel("Number of vulnerable samples")
    ax.set_title(f"4.3 – Top {top_n} CWE Categories – RDiverseVul\n"
                 f"(n={n_unique} unique CWEs, {total_labeled:,} total labels)",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = OUT_DIR / "cwe_distribution.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")

    # ── Secondary: stacked bar of top CWEs across the full dataset ───────────
    # (shows which CWEs have mixed labelling / appear in secure functions too)
    top_cwes = [c for c, _ in cwe_counts.most_common(15)]
    rows = []
    for cwe in top_cwes:
        for label_val, label_name in [(0, "Secure"), (1, "Vulnerable")]:
            sub = df[df[label_col] == label_val]
            cnt = 0
            for val in sub[cwe_col].dropna():
                if isinstance(val, list):
                    if cwe in [str(v).strip() for v in val]:
                        cnt += 1
                elif cwe in str(val):
                    cnt += 1
            rows.append({"CWE": cwe, "Class": label_name, "Count": cnt})

    stacked_df = pd.DataFrame(rows)
    if stacked_df["Count"].sum() > 0:
        pivot = stacked_df.pivot(index="CWE", columns="Class", values="Count").fillna(0)
        pivot = pivot.loc[[c for c, _ in cwe_counts.most_common(15)]]
        fig2, ax2 = plt.subplots(figsize=(13, 6))
        pivot.plot(kind="bar", ax=ax2,
                   color=[PALETTE["secure"], PALETTE["vulnerable"]],
                   edgecolor="white", linewidth=0.5, width=0.75)
        ax2.set_title("4.3b – Top 15 CWEs by Class Label", fontweight="bold")
        ax2.set_xlabel("CWE category")
        ax2.set_ylabel("Sample count")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        out2 = OUT_DIR / "cwe_by_class.png"
        plt.savefig(out2, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"    Saved → {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(df: pd.DataFrame,
                 label_col: str = "target",
                 cwe_col: str   = "cwe") -> None:
    """Write a concise Markdown summary of EDA findings."""
    lines = ["# RDiverseVul – EDA Summary\n",
             f"**Total samples:** {len(df):,}\n"]

    counts = df[label_col].value_counts().sort_index()
    vul_n  = int(counts.get(1, 0))
    sec_n  = int(counts.get(0, 0))
    lines += [
        f"**Vulnerable (1):** {vul_n:,} ({vul_n/len(df)*100:.1f}%)\n",
        f"**Secure (0):**      {sec_n:,} ({sec_n/len(df)*100:.1f}%)\n",
        f"**Imbalance ratio:** {sec_n/max(vul_n,1):.1f}:1\n\n",
        "## Output files\n",
        "- `eda_outputs/class_imbalance.png`\n",
        "- `eda_outputs/token_length_distribution.png`\n",
        "- `eda_outputs/cwe_distribution.png`\n",
        "- `eda_outputs/cwe_by_class.png`\n",
    ]

    out = OUT_DIR / "eda_summary.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\n    Summary saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA for RDiverseVul – HEFT Vulnerability Detection Project"
    )
    parser.add_argument("--dataset",    default="RDiverseVul.json",
                        help="Path to RDiverseVul.json")
    parser.add_argument("--label_col",  default="target",
                        help="Column name for binary label (0/1)")
    parser.add_argument("--code_col",   default="func",
                        help="Column name for C/C++ function code")
    parser.add_argument("--cwe_col",    default="cwe",
                        help="Column name for CWE labels")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-Coder-1.5B",
                        help="HuggingFace model name for tokenizer")
    parser.add_argument("--sample_tok", type=int, default=20_000,
                        help="Samples used for token-length estimation")
    parser.add_argument("--top_cwe",    type=int, default=25,
                        help="Number of top CWEs to plot")
    args, unknown = parser.parse_known_args()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    df = load_dataset(args.dataset)
    inspect_schema(df)

    if df[args.label_col].dtype == 'O': # 'O' means Object/Dictionary/String
        df[args.label_col] = df[args.label_col].apply(
            lambda x: int(next(iter(x.values()))) if isinstance(x, dict) else int(x)
        )
    
    # Now the rest of your pipeline will run smoothly!
    analyze_class_imbalance(df, label_col=args.label_col)
    analyze_token_lengths(df,
                          code_col=args.code_col,
                          label_col=args.label_col,
                          model_name=args.model,
                          sample_size=args.sample_tok)
    analyze_cwe_distribution(df,
                              cwe_col=args.cwe_col,
                              label_col=args.label_col,
                              top_n=args.top_cwe)
    save_summary(df, label_col=args.label_col, cwe_col=args.cwe_col)
    print("\n✅  EDA complete. All outputs saved to ./eda_outputs/")


if __name__ == "__main__":
    main()