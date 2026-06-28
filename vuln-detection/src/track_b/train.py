"""
5-fold CV training loop for all Track B experiments.

Experiment map
--------------
EXP_2  — frozen backbone (probe): head only
EXP_3  — LoRA on W_q + W_v, rank ∈ cfg.lora.r  (one run per rank)
EXP_4  — load best EXP_3 LoRA, freeze it, apply ReFT, rank ∈ cfg.reft.r
EXP_5  — identical to EXP_4 with backbone swapped to ModernBERT-Base

Call order per experiment
-------------------------
1. build_model(cfg)                         — fresh backbone + head
2. apply_lora / freeze_backbone             — phase-specific setup
3. run_experiment(exp_id, cfg, heft_cfg, df, train_idx)
   └─ for each fold:
       a. build_fold_dataloaders
       b. _build_optimizer  (differential LRs)
       c. _build_scheduler
       d. _train_epoch / _eval_epoch
       e. evaluate_model → per-fold JSON
4. evaluate_on_holdout (called from 05_results.ipynb)

Adapter weights saved to results/checkpoints/{exp_id}_r{r}/fold{N}/
Per-fold JSON metrics to results/metrics/
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data import COL_TARGET, get_cv_folds, get_texts_and_labels
from src.evaluate import (
    EvaluationResult,
    aggregate_fold_metrics,
    evaluate_model,
    plot_all_folds_pr_curves,
)
from src.track_b.dataset import VulnDataset, build_dataloader, build_fold_dataloaders
from src.track_b.lora import apply_lora, freeze_lora, save_lora_adapter
from src.track_b.model import NeoBERTForVulnClassification, build_model
from src.track_b.reft import (
    _REFT_ATTR,
    apply_reft,
    get_reft_param_count,
    load_reft_adapter,
    reft_forward,
    save_reft_adapter,
)
from src.utils import (
    class_weights_as_tensor,
    generate_exp_id,
    get_logger,
    load_config,
    save_metrics,
    set_seed,
)

_log = get_logger(__name__)

_METRICS_DIR     = Path("results/metrics")
_FIGURES_DIR     = Path("results/figures")
_CHECKPOINTS_DIR = Path("results/checkpoints")

_VALID_EXPERIMENTS = {"EXP_2", "EXP_3", "EXP_4", "EXP_5"}

# ModernBERT-Base model name used for EXP_5.
_MODERNBERT_NAME = "answerdotai/ModernBERT-base"


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: NeoBERTForVulnClassification,
    cfg,
    heft_cfg=None,
    *,
    phase: str = "lora",
) -> AdamW:
    """Build AdamW with differential learning rates.

    Three parameter groups:
      - backbone (base weights, if trainable): lr from cfg.training.lr
      - LoRA / ReFT adapters:                 lr from heft_cfg (phase-specific)
      - classification head:                  lr from cfg.training.lr

    Args:
        model:    The model whose parameters are grouped.
        cfg:      track_b.yaml namespace (training.lr, training.weight_decay).
        heft_cfg: heft.yaml namespace. Required when phase in {"lora","reft"}.
        phase:    One of "probe" | "lora" | "reft".

    Returns:
        Configured ``AdamW`` optimiser.
    """
    base_lr    = cfg.training.lr
    wd         = cfg.training.weight_decay
    head_params = model.get_head_params()
    head_ids    = {id(p) for p in head_params}

    if phase == "probe":
        # Only head parameters are trainable.
        param_groups = [{"params": head_params, "lr": base_lr}]

    elif phase == "lora":
        adapter_lr = heft_cfg.lora.lr if hasattr(heft_cfg.lora, "lr") else base_lr
        backbone_params = [
            p for p in model.get_backbone_params() if id(p) not in head_ids
        ]
        param_groups = [
            {"params": backbone_params, "lr": adapter_lr},
            {"params": head_params,     "lr": base_lr},
        ]

    elif phase == "reft":
        reft_lr = heft_cfg.reft.lr
        reft_params = [
            p
            for p in getattr(model, _REFT_ATTR).parameters()
            if p.requires_grad
        ]
        param_groups = [
            {"params": reft_params,  "lr": reft_lr},
            {"params": head_params,  "lr": base_lr},
        ]

    else:
        raise ValueError(f"Unknown phase '{phase}'. Choose probe | lora | reft.")

    optimizer = AdamW(param_groups, weight_decay=wd)
    _log.info(
        "AdamW built — phase=%s  groups=%d  wd=%.4f",
        phase, len(param_groups), wd,
    )
    return optimizer


def _build_scheduler(
    optimizer: AdamW,
    *,
    num_training_steps: int,
    warmup_ratio: float = 0.06,
) -> LambdaLR:
    """Linear warmup then linear decay scheduler.

    Args:
        optimizer:           The optimiser to schedule.
        num_training_steps:  Total number of gradient update steps.
        warmup_ratio:        Fraction of steps used for linear warmup.

    Returns:
        A ``LambdaLR`` scheduler.
    """
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: NeoBERTForVulnClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    *,
    class_weights: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
    use_reft: bool,
) -> float:
    """Run one training epoch.

    Args:
        model:          The model being trained.
        loader:         Training DataLoader.
        optimizer:      AdamW optimiser.
        scheduler:      LR scheduler (stepped every batch).
        scaler:         GradScaler for mixed-precision training.
        class_weights:  (num_classes,) tensor on ``device``.
        device:         Target device.
        use_fp16:       Whether to use autocast.
        use_reft:       Whether to use the reft_forward path.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    n_batches  = 0

    amp_ctx = autocast() if use_fp16 and torch.cuda.is_available() else contextlib.nullcontext()

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            if use_reft:
                outputs = reft_forward(model, input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            loss   = loss_fn(logits, labels)

        if use_fp16 and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def _eval_epoch(
    model: NeoBERTForVulnClassification,
    loader: DataLoader,
    *,
    device: torch.device,
    use_fp16: bool,
    use_reft: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over one DataLoader split.

    Args:
        model:     The model in eval mode.
        loader:    Validation or holdout DataLoader.
        device:    Target device.
        use_fp16:  Whether to use autocast for inference.
        use_reft:  Whether to use the reft_forward path.

    Returns:
        Tuple of (y_true, y_scores) numpy arrays, where y_scores are the
        softmax probabilities for the positive class.
    """
    model.eval()
    all_labels: List[int]  = []
    all_scores: List[float] = []

    amp_ctx = autocast() if use_fp16 and torch.cuda.is_available() else contextlib.nullcontext()

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].cpu().numpy()

        with amp_ctx:
            if use_reft:
                outputs = reft_forward(model, input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask)

        logits = outputs["logits"].float()
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        all_labels.extend(labels.tolist())
        all_scores.extend(probs.tolist())

    return np.array(all_labels, dtype=np.int64), np.array(all_scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-experiment setup helpers
# ---------------------------------------------------------------------------

def _setup_exp2(model: NeoBERTForVulnClassification, cfg) -> Tuple[str, str]:
    """EXP_2: freeze backbone, train head only."""
    model.freeze_backbone()
    return "probe", f"EXP_2"


def _setup_exp3(
    model: NeoBERTForVulnClassification,
    cfg,
    heft_cfg,
    *,
    r: int,
) -> Tuple[str, str]:
    """EXP_3: apply LoRA (rank r) and train with backbone unfrozen."""
    model.unfreeze_backbone()
    apply_lora(model, heft_cfg, r=r)
    return "lora", f"EXP_3_r{r}"


def _setup_exp4(
    model: NeoBERTForVulnClassification,
    cfg,
    heft_cfg,
    *,
    lora_r: int,
    reft_r: int,
    lora_checkpoint_dir: Path,
) -> Tuple[str, str]:
    """EXP_4: load best EXP_3 LoRA checkpoint, freeze it, then apply ReFT."""
    from src.track_b.lora import load_lora_adapter

    apply_lora(model, heft_cfg, r=lora_r)
    load_lora_adapter(model, lora_checkpoint_dir)
    freeze_lora(model)
    apply_reft(model, heft_cfg, r=reft_r)
    return "reft", f"EXP_4_lorar{lora_r}_reftr{reft_r}"


def _setup_exp5(
    model: NeoBERTForVulnClassification,
    cfg,
    heft_cfg,
    *,
    lora_r: int,
    reft_r: int,
    lora_checkpoint_dir: Path,
) -> Tuple[str, str]:
    """EXP_5: identical to EXP_4, backbone already swapped to ModernBERT-Base."""
    phase, base_id = _setup_exp4(
        model, cfg, heft_cfg,
        lora_r=lora_r,
        reft_r=reft_r,
        lora_checkpoint_dir=lora_checkpoint_dir,
    )
    return phase, base_id.replace("EXP_4", "EXP_5")


# ---------------------------------------------------------------------------
# Adapter checkpoint helpers
# ---------------------------------------------------------------------------

def _save_adapter(
    model: NeoBERTForVulnClassification,
    *,
    exp_id: str,
    fold: int,
    phase: str,
    checkpoints_dir: Path,
) -> Path:
    """Save adapter weights for the current fold."""
    out_dir = checkpoints_dir / exp_id / f"fold{fold}"
    if phase == "reft":
        save_reft_adapter(model, out_dir)
    else:
        # probe or lora — head is always saved alongside.
        from src.track_b.lora import save_lora_adapter as _save_lora
        _save_lora(model, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Core CV loop
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id: str,
    cfg,
    heft_cfg,
    df,
    train_idx: np.ndarray,
    *,
    lora_r: int = 8,
    reft_r: int = 4,
    lora_checkpoint_dir: Optional[Path] = None,
    metrics_dir:     Path = _METRICS_DIR,
    figures_dir:     Path = _FIGURES_DIR,
    checkpoints_dir: Path = _CHECKPOINTS_DIR,
    threshold: float = 0.5,
    save_pr_curves: bool = True,
    overwrite: bool = False,
) -> Tuple[List[EvaluationResult], Dict]:
    """Run 5-fold CV for a single Track B experiment.

    Args:
        exp_id:               One of EXP_2 | EXP_3 | EXP_4 | EXP_5.
        cfg:                  Config namespace from ``configs/track_b.yaml``.
        heft_cfg:             Config namespace from ``configs/heft.yaml``.
        df:                   Full dataset DataFrame.
        train_idx:            Indices of the training partition (holdout excluded).
        lora_r:               LoRA rank used in EXP_3 / as frozen base in EXP_4-5.
        reft_r:               ReFT rank used in EXP_4 / EXP_5.
        lora_checkpoint_dir:  Directory of best EXP_3 LoRA fold checkpoint.
                              Required for EXP_4 / EXP_5.
        metrics_dir:          Where per-fold JSON files are written.
        figures_dir:          Where PR curve PNGs are written.
        checkpoints_dir:      Where adapter weights are written.
        threshold:            Decision threshold for binary metrics.
        save_pr_curves:       Whether to plot PR curves after CV.
        overwrite:            Overwrite existing metric files if True.

    Returns:
        Tuple of (fold_results, aggregate_metrics_dict).

    Raises:
        ValueError: If ``exp_id`` is not a valid experiment identifier.
    """
    base_id = exp_id.split("_")[0] + "_" + exp_id.split("_")[1] if "_" in exp_id else exp_id
    if base_id not in _VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{exp_id}'. Valid: {sorted(_VALID_EXPERIMENTS)}"
        )

    set_seed(cfg.cv.random_state)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.training.fp16 and torch.cuda.is_available()
    labels  = df[COL_TARGET].to_numpy(dtype=np.int64)

    fold_results:      List[EvaluationResult] = []
    fold_data_for_plot: List[Dict]             = []

    _log.info("=" * 60)
    _log.info("Starting %s  device=%s  fp16=%s", exp_id, device, use_fp16)
    _log.info("=" * 60)

    for fold, (fold_train_idx, fold_val_idx) in enumerate(
        get_cv_folds(
            train_idx, labels,
            n_splits=cfg.cv.n_splits,
            random_state=cfg.cv.random_state,
        )
    ):
        _log.info("── Fold %d / %d ──", fold + 1, cfg.cv.n_splits)

        # ── Build a fresh model for every fold ──────────────────────────
        fold_model = build_model(cfg, device=device)

        # ── Experiment-specific setup ────────────────────────────────────
        base_exp = exp_id[:5]   # "EXP_2" … "EXP_5"
        if base_exp == "EXP_2":
            phase, run_id = _setup_exp2(fold_model, cfg)
        elif base_exp == "EXP_3":
            phase, run_id = _setup_exp3(fold_model, cfg, heft_cfg, r=lora_r)
        elif base_exp == "EXP_4":
            if lora_checkpoint_dir is None:
                raise ValueError("lora_checkpoint_dir is required for EXP_4.")
            phase, run_id = _setup_exp4(
                fold_model, cfg, heft_cfg,
                lora_r=lora_r, reft_r=reft_r,
                lora_checkpoint_dir=lora_checkpoint_dir,
            )
        else:  # EXP_5
            if lora_checkpoint_dir is None:
                raise ValueError("lora_checkpoint_dir is required for EXP_5.")
            phase, run_id = _setup_exp5(
                fold_model, cfg, heft_cfg,
                lora_r=lora_r, reft_r=reft_r,
                lora_checkpoint_dir=lora_checkpoint_dir,
            )

        use_reft = hasattr(fold_model, _REFT_ATTR)

        # ── Data ─────────────────────────────────────────────────────────
        texts_train, y_train = get_texts_and_labels(df, fold_train_idx)
        texts_val,   y_val   = get_texts_and_labels(df, fold_val_idx)

        from src.track_b.dataset import load_tokenizer_from_config
        tokenizer = load_tokenizer_from_config(cfg)
        train_loader, val_loader = build_fold_dataloaders(
            texts_train, y_train.tolist(),
            texts_val,   y_val.tolist(),
            tokenizer, cfg,
        )

        # ── Class weights ─────────────────────────────────────────────────
        cw = class_weights_as_tensor(y_train.tolist(), num_classes=2, device=device)

        # ── Optimizer & scheduler ─────────────────────────────────────────
        n_epochs = (
            heft_cfg.reft.epochs if phase == "reft" else cfg.training.epochs
        )
        steps_per_epoch    = len(train_loader)
        num_training_steps = steps_per_epoch * n_epochs

        optimizer = _build_optimizer(fold_model, cfg, heft_cfg, phase=phase)
        scheduler = _build_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=cfg.training.warmup_ratio,
        )
        scaler = GradScaler(enabled=use_fp16 and torch.cuda.is_available())

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(n_epochs):
            train_loss = _train_epoch(
                fold_model, train_loader, optimizer, scheduler, scaler,
                class_weights=cw,
                device=device,
                use_fp16=use_fp16,
                use_reft=use_reft,
            )
            _log.info(
                "  Epoch %d/%d — train_loss=%.4f", epoch + 1, n_epochs, train_loss
            )

        # ── Evaluation ────────────────────────────────────────────────────
        y_true, y_scores = _eval_epoch(
            fold_model, val_loader,
            device=device, use_fp16=use_fp16, use_reft=use_reft,
        )

        result = evaluate_model(
            y_true, y_scores,
            fold=fold, exp_id=run_id,
            output_dir=metrics_dir,
            threshold=threshold,
            overwrite=overwrite,
        )
        fold_results.append(result)
        fold_data_for_plot.append(
            {"fold": fold, "y_true": y_true.tolist(), "y_scores": y_scores.tolist()}
        )

        # ── Save adapter weights ───────────────────────────────────────────
        _save_adapter(
            fold_model,
            exp_id=run_id, fold=fold, phase=phase,
            checkpoints_dir=checkpoints_dir,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = aggregate_fold_metrics(fold_results)
    save_metrics(
        agg,
        Path(metrics_dir) / f"{run_id}_aggregate.json",
        overwrite=overwrite,
    )
    _log.info(
        "%s CV complete — mean F1=%.4f  mean PR-AUC=%.4f",
        run_id,
        agg["mean"]["f1"],
        agg["mean"]["pr_auc"],
    )

    if save_pr_curves:
        try:
            plot_all_folds_pr_curves(
                fold_data_for_plot,
                exp_id=run_id,
                output_path=Path(figures_dir) / f"{run_id}_pr_curves.png",
            )
        except ImportError:
            _log.warning("matplotlib not available; skipping PR curve plot.")

    return fold_results, agg


# ---------------------------------------------------------------------------
# run_all convenience wrappers
# ---------------------------------------------------------------------------

def run_all_track_b(
    cfg,
    heft_cfg,
    df,
    train_idx: np.ndarray,
    *,
    lora_checkpoint_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Dict]:
    """Run EXP_2 through EXP_4 in sequence, iterating over configured ranks.

    EXP_5 (ModernBERT-Base) must be triggered separately via
    ``run_experiment("EXP_5", ...)`` with a modified config.

    Args:
        cfg:                 track_b.yaml namespace.
        heft_cfg:            heft.yaml namespace.
        df:                  Full dataset DataFrame.
        train_idx:           Training partition indices.
        lora_checkpoint_dir: Best EXP_3 checkpoint dir for EXP_4 input.
        **kwargs:            Forwarded to ``run_experiment``.

    Returns:
        Dict mapping run_id strings to their aggregate metric dicts.
    """
    results: Dict[str, Dict] = {}

    # EXP_2 — frozen probe (single run, no rank sweep).
    _, agg = run_experiment("EXP_2", cfg, heft_cfg, df, train_idx, **kwargs)
    results["EXP_2"] = agg

    # EXP_3 — LoRA rank sweep.
    for lora_r in heft_cfg.lora.r:
        _, agg = run_experiment(
            "EXP_3", cfg, heft_cfg, df, train_idx, lora_r=lora_r, **kwargs
        )
        results[f"EXP_3_r{lora_r}"] = agg

    # EXP_4 — HEFT rank sweep (requires EXP_3 checkpoint).
    if lora_checkpoint_dir is not None:
        for lora_r in heft_cfg.lora.r:
            for reft_r in heft_cfg.reft.r:
                _, agg = run_experiment(
                    "EXP_4", cfg, heft_cfg, df, train_idx,
                    lora_r=lora_r, reft_r=reft_r,
                    lora_checkpoint_dir=lora_checkpoint_dir,
                    **kwargs,
                )
                results[f"EXP_4_lorar{lora_r}_reftr{reft_r}"] = agg
    else:
        _log.warning("lora_checkpoint_dir not provided — skipping EXP_4.")

    return results


# ---------------------------------------------------------------------------
# Holdout evaluation (called from 05_results.ipynb only)
# ---------------------------------------------------------------------------

def evaluate_on_holdout(
    exp_id: str,
    cfg,
    heft_cfg,
    df,
    train_idx:   np.ndarray,
    holdout_idx: np.ndarray,
    *,
    lora_r:              int  = 8,
    reft_r:              int  = 4,
    lora_checkpoint_dir: Optional[Path] = None,
    adapter_checkpoint_dir: Optional[Path] = None,
    metrics_dir:  Path  = _METRICS_DIR,
    figures_dir:  Path  = _FIGURES_DIR,
    threshold:    float = 0.5,
    overwrite:    bool  = False,
) -> EvaluationResult:
    """Retrain on full train set and evaluate once on the locked holdout.

    This function must only be called after all CV hyperparameter decisions
    are finalised.  The holdout is never touched during CV.

    Args:
        exp_id:                  Experiment identifier.
        cfg:                     track_b.yaml namespace.
        heft_cfg:                heft.yaml namespace.
        df:                      Full dataset DataFrame.
        train_idx:               Training partition indices (all 80%).
        holdout_idx:             Holdout partition indices (locked 20%).
        lora_r:                  LoRA rank for EXP_3/4/5.
        reft_r:                  ReFT rank for EXP_4/5.
        lora_checkpoint_dir:     Best EXP_3 LoRA checkpoint for EXP_4/5 base.
        adapter_checkpoint_dir:  If provided, load pre-trained adapter instead
                                 of retraining from scratch.
        metrics_dir:             Where the holdout metric JSON is written.
        figures_dir:             Where the holdout PR curve PNG is written.
        threshold:               Decision threshold for binary metrics.
        overwrite:               Overwrite existing files if True.

    Returns:
        ``EvaluationResult`` for the holdout set.
    """
    set_seed(cfg.cv.random_state)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.training.fp16 and torch.cuda.is_available()
    labels   = df[COL_TARGET].to_numpy(dtype=np.int64)

    model = build_model(cfg, device=device)

    base_exp = exp_id[:5]
    if base_exp == "EXP_2":
        phase, run_id = _setup_exp2(model, cfg)
    elif base_exp == "EXP_3":
        phase, run_id = _setup_exp3(model, cfg, heft_cfg, r=lora_r)
    elif base_exp == "EXP_4":
        phase, run_id = _setup_exp4(
            model, cfg, heft_cfg,
            lora_r=lora_r, reft_r=reft_r,
            lora_checkpoint_dir=lora_checkpoint_dir,
        )
    else:  # EXP_5
        phase, run_id = _setup_exp5(
            model, cfg, heft_cfg,
            lora_r=lora_r, reft_r=reft_r,
            lora_checkpoint_dir=lora_checkpoint_dir,
        )

    use_reft = hasattr(model, _REFT_ATTR)

    from src.track_b.dataset import load_tokenizer_from_config
    tokenizer = load_tokenizer_from_config(cfg)

    if adapter_checkpoint_dir is not None:
        # Load pre-trained adapter; skip re-training.
        if use_reft:
            load_reft_adapter(model, adapter_checkpoint_dir)
        else:
            from src.track_b.lora import load_lora_adapter
            load_lora_adapter(model, adapter_checkpoint_dir)
        _log.info("Loaded adapter from %s — skipping retraining.", adapter_checkpoint_dir)
    else:
        # Retrain on full training set.
        texts_train, y_train = get_texts_and_labels(df, train_idx)
        texts_hold,  y_hold  = get_texts_and_labels(df, holdout_idx)

        max_seq = cfg.model.max_seq_length
        bs      = cfg.training.batch_size
        train_ds = VulnDataset(texts_train, y_train.tolist(), tokenizer, max_seq_length=max_seq)
        train_loader = build_dataloader(train_ds, tokenizer, batch_size=bs, shuffle=True)

        cw = class_weights_as_tensor(y_train.tolist(), num_classes=2, device=device)
        n_epochs = heft_cfg.reft.epochs if phase == "reft" else cfg.training.epochs
        num_training_steps = len(train_loader) * n_epochs

        optimizer = _build_optimizer(model, cfg, heft_cfg, phase=phase)
        scheduler = _build_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=cfg.training.warmup_ratio,
        )
        scaler = GradScaler(enabled=use_fp16 and torch.cuda.is_available())

        for epoch in range(n_epochs):
            loss = _train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                class_weights=cw, device=device,
                use_fp16=use_fp16, use_reft=use_reft,
            )
            _log.info("Holdout retrain epoch %d/%d — loss=%.4f", epoch + 1, n_epochs, loss)

    # Evaluate on holdout.
    texts_hold, y_hold = get_texts_and_labels(df, holdout_idx)
    hold_ds     = VulnDataset(
        texts_hold, y_hold.tolist(), tokenizer,
        max_seq_length=cfg.model.max_seq_length,
    )
    hold_loader = build_dataloader(
        hold_ds, tokenizer, batch_size=cfg.training.batch_size, shuffle=False
    )

    y_true, y_scores = _eval_epoch(
        model, hold_loader,
        device=device, use_fp16=use_fp16, use_reft=use_reft,
    )

    holdout_run_id = f"{run_id}_holdout"
    result = evaluate_model(
        y_true, y_scores,
        fold=-1, exp_id=holdout_run_id,
        output_dir=metrics_dir,
        threshold=threshold,
        overwrite=overwrite,
    )

    try:
        from src.evaluate import plot_pr_curve
        plot_pr_curve(
            y_true, y_scores,
            exp_id=holdout_run_id, fold=None,
            output_path=Path(figures_dir) / f"{holdout_run_id}_pr_curve.png",
        )
    except ImportError:
        pass

    return result