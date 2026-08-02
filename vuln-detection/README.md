# Intelligent System for Software Vulnerability Detection
## via NeoBERT, RDiverseVul, and HEFT
 
**Date:** June 2026  
**Status:** In development

---

## Overview

This project researches and implements an intelligent system for predictive
software vulnerability detection in C/C++ source code, framed as a binary
classification problem. It is structured as a two-track comparative study:

- **Track A — Standard baseline:** TF-IDF feature extraction fed into Logistic
  Regression, Random Forest, and a shallow MLP.
- **Track B — Advanced pipeline:** NeoBERT-250M fine-tuned via HEFT
  (Hierarchical Efficient Fine-Tuning), a two-phase PEFT paradigm combining
  LoRA (coarse) and ReFT (fine-grained).

---

## Dataset

**RDiverseVul** (Refined DiverseVul, February 2025) — not included in this
repository. Download it manually and place the file at:

```
data/raw/rdiversevul.json
```

---

## Repository Structure

```
vuln-detection/
│
├── configs/
│   ├── track_a.yaml          # TF-IDF and classifier hyperparameters
│   ├── track_b.yaml          # Tokenizer, training loop, backbone settings
│   └── heft.yaml             # LoRA and ReFT hyperparameters
│
├── data/
│   ├── raw/                  # RDiverseVul raw file — GITIGNORED
│   ├── processed/            # Tokenized tensors — GITIGNORED
│   └── splits/               # Fold indices and holdout split — GITIGNORED
|
── data_exploration/
│   ├──eda.py                 # Class imbalance, sequence lengths, CWE distribution
│
├── notebooks/
│   ├── 00_setup.ipynb        # Colab session init: clone, install, load data
│   ├── 01_track_a.ipynb      # EXP_0 (LR) and EXP_1 (RF + MLP)
│   ├── 02_track_b_probe.ipynb  # EXP_2: frozen backbone linear probe
│   ├── 03_track_b_lora.ipynb   # EXP_3: LoRA fine-tuning
│   ├── 04_track_b_heft.ipynb   # EXP_4 (HEFT) and EXP_5 (ModernBERT)
│   └── 05_results.ipynb      # Final comparison, PR curves, CWE breakdown
│
├── src/
│   ├── track_a/
│   │   ├── features.py       # TF-IDF vectorizer pipeline
│   │   ├── models.py         # LR, RF, MLP definitions
│   │   └── train.py          # CV loop for Track A
│   ├── track_b/
│   │   ├── dataset.py        # PyTorch Dataset, tokenization, dataloaders
│   │   ├── model.py          # NeoBERT + classification head, bug fixes
│   │   ├── lora.py           # LoRA config and PEFT model setup
│   │   ├── reft.py           # ReFT intervention config and pyreft setup
│   │   └── train.py          # CV loop for Track B
│   ├── evaluate.py           # Shared: Precision, Recall, F1, PR-AUC
│   └── utils.py              # Shared: seeding, logging, class weights
│
├── results/
│   ├── checkpoints/          # Saved adapter weights — GITIGNORED
│   ├── metrics/              # Per-fold JSON score files — COMMITTED
│   └── figures/              # PR curves, CWE plots — COMMITTED
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## Experiments

| ID | Track | Description |
|----|-------|-------------|
| EXP_0 | A | Logistic Regression (L2, balanced) on TF-IDF features |
| EXP_1 | A | Random Forest + MLP on TF-IDF features |
| EXP_2 | B | Frozen NeoBERT-250M — linear probe only |
| EXP_3 | B | LoRA on W_q / W_v — rank ∈ {8, 16} |
| EXP_4 | B | Full HEFT — LoRA frozen → ReFT on layers {12,16,20,24} |
| EXP_5 | B | ModernBERT-Base with identical HEFT pipeline |

All experiments use **stratified 5-fold cross-validation** on 80% of the data,
with a locked **20% holdout** evaluated only once at the end.

---

## Evaluation Metrics

Accuracy is omitted due to class imbalance. Primary metrics:

- **PR-AUC** — primary optimization target
- **F1-Score**
- **Precision**
- **Recall**

---

## Setup

### Local

```bash
git clone https://github.com/...
cd vuln-detection
pip install -e .
```

### Google Colab

Open `notebooks/00_setup.ipynb` and run all cells. It handles cloning,
installation, and data path configuration automatically.

---

## References

- ReFT: Adapting Large Language Models for Parameter-Efficient Log Anomaly Detection — Lim et al., PAKDD 2025
- FRLog: Log Anomaly Detection Based on Three-Stage Training with ReFT — Qiu et al., JAISCR 2026
- DiverseVul: A New Vulnerable Source Code Dataset — Chen et al., RAID 2023
- Evaluating LLaMA 3.2 for Software Vulnerability Detection — Gonçalves et al., EICC 2025
- HEFT: A Coarse-to-Fine Hierarchy for Enhancing LLM Reasoning — Hill, 2025