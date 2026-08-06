# Intelligent System for Software Vulnerability Detection
## via CodeBERT-v1-small, RDiverseVul, and HEFT
 
**Date:** June 2026  
**Status:** In development

---

## Overview

This project researches and implements an intelligent system for predictive
software vulnerability detection in C/C++ source code, framed as a binary
classification problem. It is structured as a two-track comparative study:

- **Track A — Standard baseline:** TF-IDF feature extraction fed into Logistic
  Regression, Random Forest, and a shallow MLP.
- **Track B — Advanced pipeline:** CodeBERT-v1-small fine-tuned via HEFT
  (Hierarchical Efficient Fine-Tuning), a two-phase PEFT paradigm combining
  LoRA (coarse) and ReFT (fine-grained).

---

## Dataset

**RDiverseVul** (Refined DiverseVul, February 2025) — not included in this
repository. Download it manually and place the file at:


## Experiments

| ID | Track | Description |
|----|-------|-------------|
| EXP_0 | A | Logistic Regression (L2, balanced) on TF-IDF features |
| EXP_1 | A | Random Forest on TF-IDF features |
| EXP_2 | A | MLP on TF-IDF features |
| EXP_3 | B | Frozen CodeBERT — linear probe only |
| EXP_4 | B | LoRA on W_q / W_v — rank ∈ {8, 16} |
| EXP_5 | B | Full HEFT — LoRA frozen → ReFT on layers {12,16,20,24} |

All experiments use **stratified 5-fold cross-validation** on 80% of the data,
with a locked **20% holdout** evaluated only once at the end.

---

## Evaluation Metrics

Accuracy is omitted due to class imbalance. Primary metrics:

- **PR-AUC** — primary optimization target
- **F1-Score**
- **Precision**
- **Recall**


## References

- ReFT: Adapting Large Language Models for Parameter-Efficient Log Anomaly Detection — Lim et al., PAKDD 2025
- FRLog: Log Anomaly Detection Based on Three-Stage Training with ReFT — Qiu et al., JAISCR 2026
- DiverseVul: A New Vulnerable Source Code Dataset — Chen et al., RAID 2023
- Evaluating LLaMA 3.2 for Software Vulnerability Detection — Gonçalves et al., EICC 2025
- HEFT: A Coarse-to-Fine Hierarchy for Enhancing LLM Reasoning — Hill, 2025