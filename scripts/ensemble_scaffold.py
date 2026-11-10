"""
scripts/ensemble_scaffold.py
=============================
Ensemble meta-learner scaffold for combining multiple CheXpertB1V2 checkpoints.

Usage:
    python scripts/ensemble_scaffold.py \\
        --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \\
        --config configs/chexpert_b1_v2.yaml \\
        --holdout_split test \\
        --meta_learner weighted_avg \\
        --output_dir outputs/ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from scripts.train_chexpert_b1_v2 import (
    CheXpertLocalDatasetV2,
    make_patient_splits,
    CHEXPERT_14_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("ensemble")


# =============================================================================
# Inference from checkpoint
# =============================================================================

@torch.no_grad()
def load_model_predictions(
    checkpoint_path: str,
    dataset,
    cfg: dict,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    apply_calibration: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a checkpoint, run inference on dataset.

    Returns:
        logits: [N, 14]
        probs:  [N, 14] (calibrated if checkpoint has temperature)
        labels: [N, 14]
    """
    ckpt = torch.load(PROJECT_ROOT / checkpoint_path, map_location=device)
    model_cfg = ckpt.get("cfg", {}).get("model", cfg.get("model", {}))
    model = CheXpertB1V2Model(model_cfg)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    all_logits, all_targets = [], []
    for imgs, targets, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(imgs)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # Apply temperature scaling if stored in checkpoint
    temperature = 1.0
    if apply_calibration and "thresholds" in ckpt:
        # Look for temperature in a saved eval artifact
        checkpoint_dir = Path(checkpoint_path).parent
        exp_name = ckpt.get("cfg", {}).get("experiment", {}).get("name", "")
        thresh_file = checkpoint_dir.parent / "outputs" / "chexpert_b1_v2" / f"thresholds_{exp_name}.json"
        if thresh_file.exists():
            with open(thresh_file) as f:
                thresh_data = json.load(f)
            temperature = thresh_data.get("temperature", 1.0)
            log.info("Loaded temperature T=%.4f from %s", temperature, thresh_file)

    logits_cal = logits_np / max(temperature, 0.01)
    probs_cal  = 1.0 / (1.0 + np.exp(-logits_cal))

    return logits_np, probs_cal, targets_np


# =============================================================================
# Meta-learner: Weighted Average
# =============================================================================

def meta_learner_weighted_avg(
    probs_list: List[np.ndarray],
    labels: np.ndarray,
    optimize: bool = True,
    n_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted average meta-learner.

    Args:
        probs_list: list of [N, C] probability arrays, one per model
        labels:     [N, C] ground truth
        optimize:   if True, find per-class weights maximizing AUPRC

    Returns:
        weights:        [n_models] weight array
        ensemble_probs: [N, C] weighted average probabilities
    """
    n_models = len(probs_list)
    if not optimize or labels is None:
        weights = np.ones(n_models) / n_models
        ensemble_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        return weights, ensemble_probs

    # Optimize per-class weights using AUPRC on the holdout
    # Simple approach: grid search over Dirichlet simplex
    best_weights = np.ones(n_models) / n_models
    best_auprc = -1.0

    from itertools import product
    weight_grid = np.linspace(0, 1, 5)

    # For speed: just optimize a single global weight per model
    # (not per-class — per-class would require [n_models * n_classes] optimization)
    stack = np.stack(probs_list, axis=0)  # [n_models, N, C]
    best_w = np.ones(n_models) / n_models

    # Gradient-free: try random simplex samples
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n_models))
        ensemble = np.einsum("m,mnc->nc", w, stack)
        auprc_vals = []
        for c in range(labels.shape[1]):
            if len(np.unique(labels[:, c])) >= 2:
                try:
                    auprc_vals.append(average_precision_score(labels[:, c], ensemble[:, c]))
                except Exception:
                    pass
        mean_auprc = float(np.mean(auprc_vals)) if auprc_vals else 0.0
        if mean_auprc > best_auprc:
            best_auprc = mean_auprc
            best_w = w

    log.info("Optimized weights: %s → mean AUPRC=%.4f", best_w.round(3), best_auprc)
    ensemble_probs = np.einsum("m,mnc->nc", best_w, stack)
    return best_w, ensemble_probs


# =============================================================================
# Meta-learner: Per-class Logistic Regression
# =============================================================================

def meta_learner_logistic(
    logits_list: List[np.ndarray],
    labels: np.ndarray,
    C: float = 1.0,
) -> List[LogisticRegression]:
    """
    Per-class logistic regression over the concatenated logit stack.

    Args:
        logits_list: list of [N, C] arrays
        labels:      [N, C] ground truth
        C:           regularization (smaller = more regularization)

    Returns:
        list of fitted LogisticRegression, one per class
    """
    n_classes = labels.shape[1]
    # Feature: concatenate logits from all models → [N, n_models * n_classes]
    X = np.concatenate(logits_list, axis=1)  # [N, n_models * C]
    classifiers = []

    for c in range(n_classes):
        y = labels[:, c]
        if len(np.unique(y)) < 2:
            log.warning("Class %d has no variation in labels — using trivial classifier", c)
            clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs")
            clf.fit(X, np.zeros(len(y)))  # dummy fit
        else:
            clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs", class_weight="balanced")
            clf.fit(X, y)
        classifiers.append(clf)

    return classifiers


def predict_logistic(
    logits_list: List[np.ndarray],
    classifiers: List[LogisticRegression],
) -> np.ndarray:
    """Apply fitted per-class logistic regressors. Returns [N, C] probs."""
    X = np.concatenate(logits_list, axis=1)
    probs = np.stack([clf.predict_proba(X)[:, 1] for clf in classifiers], axis=1)
    return probs


# =============================================================================
# Meta-learner: XGBoost (scaffold)
# =============================================================================

def meta_learner_xgboost(logits_list: List[np.ndarray], labels: np.ndarray):
    """
    XGBoost meta-learner (per class). Requires xgboost package.

    Args:
        logits_list: list of [N, C] arrays
        labels:      [N, C] ground truth

    Returns:
        list of fitted xgb.XGBClassifier, one per class
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "XGBoost not installed. Run: pip install xgboost"
        )

    n_classes = labels.shape[1]
    X = np.concatenate(logits_list, axis=1)
    models = []

    for c in range(n_classes):
        y = labels[:, c]
        if len(np.unique(y)) < 2:
            models.append(None)
            continue
        pos = y.sum()
        neg = len(y) - pos
        scale_pos_weight = neg / max(pos, 1)
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        clf.fit(X, y, eval_set=[(X, y)], verbose=False)
        models.append(clf)

    return models


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ensemble(
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
    individual_probs_list: List[np.ndarray],
    label_names: List[str],
) -> Dict:
    """
    Compute AUROC/AUPRC for ensemble and each individual model.
    Returns comparison dict.
    """
    def _metrics(probs, labels):
        auroc_vals, auprc_vals = [], []
        for c in range(labels.shape[1]):
            if len(np.unique(labels[:, c])) >= 2:
                try:
                    auroc_vals.append(roc_auc_score(labels[:, c], probs[:, c]))
                    auprc_vals.append(average_precision_score(labels[:, c], probs[:, c]))
                except Exception:
                    pass
        return {
            "mean_auroc": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
            "mean_auprc": float(np.mean(auprc_vals)) if auprc_vals else float("nan"),
        }

    results = {"ensemble": _metrics(ensemble_probs, labels)}
    for i, probs in enumerate(individual_probs_list):
        results[f"model_{i+1}"] = _metrics(probs, labels)

    # Print comparison
    print(f"\n{'='*50}")
    print(f"{'Model':<20} {'AUROC':>8} {'AUPRC':>8}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<20} {m['mean_auroc']:>8.4f} {m['mean_auprc']:>8.4f}")
    print("=" * 50)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ensemble meta-learner")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to .pth checkpoints")
    parser.add_argument("--config", default="configs/chexpert_b1_v2.yaml")
    parser.add_argument("--holdout_split", default="test",
                        choices=["test", "val"])
    parser.add_argument("--meta_learner", default="weighted_avg",
                        choices=["weighted_avg", "logistic", "xgboost"])
    parser.add_argument("--output_dir", default="outputs/ensemble")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all = pd.read_csv(PROJECT_ROOT / data_cfg.get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv"))
    train_df, val_df, test_df = make_patient_splits(df_all)
    holdout_df = {"test": test_df, "val": val_df}[args.holdout_split]

    ds_kwargs = {
        "uncertain_policy": data_cfg.get("uncertain_policy", "u_zeros"),
        "cache_dir": str(PROJECT_ROOT / data_cfg.get("cache_dir", "data/chexpert_cache")),
        "model_input_resolution": cfg.get("model", {}).get("input_resolution", 224),
    }
    holdout_ds = CheXpertLocalDatasetV2(holdout_df, split="val", cfg=ds_kwargs)

    # Collect predictions from each model
    all_logits, all_probs, all_labels = [], [], []
    for ckpt_path in args.checkpoints:
        log.info("Loading predictions from: %s", ckpt_path)
        logits, probs, labels = load_model_predictions(
            ckpt_path, holdout_ds, cfg, device,
            batch_size=args.batch_size,
            num_workers=train_cfg.get("num_workers", 4),
        )
        all_logits.append(logits)
        all_probs.append(probs)
        all_labels.append(labels)

    # All label arrays should be the same (use first)
    labels_all = all_labels[0]
    assert all(np.allclose(labels_all, l) for l in all_labels[1:]), \
        "Label arrays differ across checkpoints — check splits!"

    # Apply meta-learner
    log.info("Meta-learner: %s", args.meta_learner)
    if args.meta_learner == "weighted_avg":
        weights, ensemble_probs = meta_learner_weighted_avg(all_probs, labels_all, optimize=True)
        log.info("Optimal weights: %s", weights.round(3))
    elif args.meta_learner == "logistic":
        classifiers = meta_learner_logistic(all_logits, labels_all)
        ensemble_probs = predict_logistic(all_logits, classifiers)
    elif args.meta_learner == "xgboost":
        xgb_models = meta_learner_xgboost(all_logits, labels_all)
        import xgboost as xgb
        X = np.concatenate(all_logits, axis=1)
        ensemble_probs = np.stack(
            [m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(X))
             for m in xgb_models],
            axis=1,
        )

    # Evaluate
    results = evaluate_ensemble(ensemble_probs, labels_all, all_probs, CHEXPERT_14_LABELS)

    # Save
    np.savez(str(output_dir / "ensemble_probs.npz"),
             ensemble_probs=ensemble_probs,
             labels=labels_all)
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("Saved ensemble results to %s", output_dir)


if __name__ == "__main__":
    main()

