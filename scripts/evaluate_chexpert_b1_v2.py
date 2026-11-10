"""
scripts/evaluate_chexpert_b1_v2.py
===================================
Evaluation script for CheXpertB1V2Model checkpoints.

Computes per-class and macro:
  - AUROC, AUPRC
  - F1/precision/recall at threshold 0.5
  - Per-class threshold tuning (maximize F1 on val)
  - Temperature scaling calibration + ECE
  - Saves CSV results table, threshold JSON, confusion counts

Usage:
    python scripts/evaluate_chexpert_b1_v2.py \\
        --checkpoint checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth \\
        --config configs/chexpert_b1_v2.yaml \\
        --split val \\
        [--target_recall 0.80]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from scripts.train_chexpert_b1_v2 import (
    CheXpertLocalDatasetV2,
    make_patient_splits,
    CHEXPERT_14_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_b1v2")


# =============================================================================
# Calibration: Temperature Scaling
# =============================================================================

class TemperatureScaling(nn.Module):
    """Single temperature parameter that scales all logits before sigmoid."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits: np.ndarray, targets: np.ndarray, n_iter: int = 200, lr: float = 0.01):
        """Fit temperature on a validation set by minimizing NLL."""
        logits_t = torch.from_numpy(logits).float()
        targets_t = torch.from_numpy(targets).float()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)

        def _eval():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled, targets_t)
            loss.backward()
            return loss

        optimizer.step(_eval)
        log.info("Temperature scaling fitted: T=%.4f", self.temperature.item())


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) — flatten over all classes.
    ECE = sum_b (|bin_b| / N) * |acc_b - conf_b|
    """
    probs_flat = probs.ravel()
    targets_flat = targets.ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs_flat)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        conf = probs_flat[mask].mean()
        acc  = targets_flat[mask].mean()
        ece += (mask.sum() / n_total) * abs(acc - conf)

    return float(ece)


# =============================================================================
# Threshold Tuning
# =============================================================================

def tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    search_range: np.ndarray = None,
    target_recall: Optional[float] = None,
) -> Dict[str, float]:
    """
    Per-class threshold tuning.
    If target_recall is set: find threshold giving >= target_recall (take highest such threshold).
    Otherwise: maximize F1.
    Returns {label: threshold}.
    """
    if search_range is None:
        search_range = np.linspace(0.05, 0.95, 37)

    thresholds = {}
    for i, lbl in enumerate(labels):
        t = targets[:, i]
        p = probs[:, i]

        if len(np.unique(t)) < 2:
            thresholds[lbl] = 0.5
            continue

        best_thresh = 0.5
        best_score = -1.0

        for thresh in search_range:
            preds = (p >= thresh).astype(int)
            if target_recall is not None:
                rec = recall_score(t, preds, zero_division=0)
                if rec >= target_recall:
                    # Want highest threshold that achieves target recall
                    if thresh > best_score:
                        best_score = thresh
                        best_thresh = thresh
            else:
                f1 = f1_score(t, preds, zero_division=0)
                if f1 > best_score:
                    best_score = f1
                    best_thresh = thresh

        thresholds[lbl] = float(best_thresh)

    return thresholds


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (logits [N,14], targets [N,14], masks [N,14]).
    """
    model.eval()
    all_logits, all_targets, all_masks = [], [], []

    for imgs, targets, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, _ = model(imgs)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.numpy())
        all_masks.append(masks.numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_masks, axis=0),
    )


# =============================================================================
# Metrics computation
# =============================================================================

def compute_all_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    thresholds_at05: bool = True,
    tuned_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compute per-class and macro metrics."""
    results = {}
    valid_auroc, valid_auprc = [], []

    for i, lbl in enumerate(labels):
        t = targets[:, i]
        p = probs[:, i]

        row = {"label": lbl}

        # AUROC / AUPRC
        if len(np.unique(t)) < 2:
            row["auroc"] = float("nan")
            row["auprc"] = float("nan")
        else:
            try:
                row["auroc"] = float(roc_auc_score(t, p))
                row["auprc"] = float(average_precision_score(t, p))
                valid_auroc.append(row["auroc"])
                valid_auprc.append(row["auprc"])
            except Exception as e:
                log.warning("Metric error for %s: %s", lbl, e)
                row["auroc"] = float("nan")
                row["auprc"] = float("nan")

        # At threshold 0.5
        preds_05 = (p >= 0.5).astype(int)
        row["f1_05"]        = float(f1_score(t, preds_05, zero_division=0))
        row["precision_05"] = float(precision_score(t, preds_05, zero_division=0))
        row["recall_05"]    = float(recall_score(t, preds_05, zero_division=0))

        # At tuned threshold
        if tuned_thresholds is not None:
            thr = tuned_thresholds.get(lbl, 0.5)
            preds_tun = (p >= thr).astype(int)
            row["threshold_tuned"] = float(thr)
            row["f1_tuned"]        = float(f1_score(t, preds_tun, zero_division=0))
            row["precision_tuned"] = float(precision_score(t, preds_tun, zero_division=0))
            row["recall_tuned"]    = float(recall_score(t, preds_tun, zero_division=0))

        # Confusion counts (at 0.5)
        tp = int(((preds_05 == 1) & (t == 1)).sum())
        fp = int(((preds_05 == 1) & (t == 0)).sum())
        tn = int(((preds_05 == 0) & (t == 0)).sum())
        fn = int(((preds_05 == 0) & (t == 1)).sum())
        row.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "n_pos": int(t.sum()), "n_neg": int((t == 0).sum())})

        results[lbl] = row

    # Macro
    results["_macro"] = {
        "mean_auroc": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "mean_auprc": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
        "macro_f1_05": float(np.mean([v["f1_05"] for v in results.values() if "f1_05" in v])),
    }

    return results


# =============================================================================
# Pretty print
# =============================================================================

def print_results_table(results: Dict, labels: List[str], calibrated: bool = False):
    suf = "_tuned" if calibrated else "_05"
    hdr = f"{'Label':<35} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Prec':>6} {'Recall':>6} {'#Pos':>6}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for lbl in labels:
        r = results[lbl]
        auroc = r.get("auroc", float("nan"))
        auprc = r.get("auprc", float("nan"))
        f1 = r.get(f"f1{suf}", r.get("f1_05", float("nan")))
        pre = r.get(f"precision{suf}", r.get("precision_05", float("nan")))
        rec = r.get(f"recall{suf}", r.get("recall_05", float("nan")))
        n_pos = r.get("n_pos", 0)
        auroc_s = f"{auroc:.4f}" if not (isinstance(auroc, float) and np.isnan(auroc)) else "  N/A "
        auprc_s = f"{auprc:.4f}" if not (isinstance(auprc, float) and np.isnan(auprc)) else "  N/A "
        print(f"{lbl:<35} {auroc_s:>7} {auprc_s:>7} {f1:>6.4f} {pre:>6.4f} {rec:>6.4f} {n_pos:>6}")
    macro = results.get("_macro", {})
    print("-" * len(hdr))
    print(f"{'MACRO':<35} {macro.get('mean_auroc', 0):>7.4f} {macro.get('mean_auprc', 0):>7.4f} {macro.get('macro_f1_05', 0):>6.4f}")
    print("=" * len(hdr))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CheXpertB1V2Model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", default="configs/chexpert_b1_v2.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--target_recall", type=float, default=None)
    parser.add_argument("--no_calibration", action="store_true")
    parser.add_argument("--no_threshold_tuning", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg    = cfg.get("data", {})
    model_cfg   = cfg.get("model", {})
    eval_cfg    = cfg.get("eval", {})
    log_cfg     = cfg.get("logging", {})
    exp_cfg     = cfg.get("experiment", {})
    train_cfg   = cfg.get("training", {})

    experiment_name = exp_cfg.get("name", "chexpert_b1_v2_baseline")
    results_dir = Path(log_cfg.get("results_dir", "outputs/chexpert_b1_v2"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | Checkpoint: %s | Split: %s", device, args.checkpoint, args.split)

    # ── Load checkpoint + model ───────────────────────────────────────────────
    ckpt = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {}).get("model", {})
    # Merge: checkpoint model cfg takes precedence (preserves architecture)
    effective_model_cfg = {**model_cfg, **ckpt_cfg}

    model = CheXpertB1V2Model(effective_model_cfg)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys: %d", len(missing))
    if unexpected:
        log.warning("Unexpected keys: %d", len(unexpected))
    model = model.to(device)
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    index_csv = PROJECT_ROOT / data_cfg.get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv")
    df_all = pd.read_csv(index_csv)

    train_df, val_df, test_df = make_patient_splits(
        df_all,
        val_frac=data_cfg.get("val_frac", 0.10),
        test_frac=data_cfg.get("test_frac", 0.10),
        seed=data_cfg.get("split_seed", 42),
    )

    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    log.info("Evaluating on %s split: %d samples", args.split, len(split_df))

    ds_kwargs = {
        "uncertain_policy": data_cfg.get("uncertain_policy", "u_zeros"),
        "cache_dir": str(PROJECT_ROOT / data_cfg.get("cache_dir", "data/chexpert_cache")),
        "model_input_resolution": effective_model_cfg.get("input_resolution", 224),
    }
    eval_ds = CheXpertLocalDatasetV2(split_df, split="val", cfg=ds_kwargs)  # always val-mode transforms
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=eval_cfg.get("val_batch_size", 64),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    use_amp = train_cfg.get("use_amp", True)
    log.info("Running inference...")
    logits, targets, masks = run_inference(model, eval_loader, device, use_amp=use_amp)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # ── Calibration ───────────────────────────────────────────────────────────
    temp_scaler = None
    ece_before = compute_ece(probs, targets)
    log.info("ECE before calibration: %.4f", ece_before)

    if eval_cfg.get("calibration", True) and not args.no_calibration:
        log.info("Fitting temperature scaling on %s set...", args.split)
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(logits, targets)
        logits_cal = logits / temp_scaler.temperature.item()
        probs_cal = 1.0 / (1.0 + np.exp(-logits_cal))
        ece_after = compute_ece(probs_cal, targets)
        log.info("ECE after calibration: %.4f (T=%.4f)", ece_after, temp_scaler.temperature.item())
        # Use calibrated probs for downstream
        probs_eval = probs_cal
    else:
        probs_eval = probs
        ece_after = ece_before
        logits_cal = logits

    # ── Threshold tuning ──────────────────────────────────────────────────────
    tuned_thresholds = None
    if eval_cfg.get("threshold_tuning", True) and not args.no_threshold_tuning:
        log.info("Tuning per-class thresholds...")
        tuned_thresholds = tune_thresholds(
            probs_eval, targets, CHEXPERT_14_LABELS,
            target_recall=args.target_recall,
        )
        log.info("Tuned thresholds: %s",
                 {k: f"{v:.2f}" for k, v in tuned_thresholds.items()})

    # ── Compute metrics ───────────────────────────────────────────────────────
    results = compute_all_metrics(
        probs_eval, targets, CHEXPERT_14_LABELS,
        tuned_thresholds=tuned_thresholds,
    )

    print_results_table(results, CHEXPERT_14_LABELS,
                        calibrated=(tuned_thresholds is not None))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for lbl in CHEXPERT_14_LABELS:
        rows.append(results[lbl])
    df_results = pd.DataFrame(rows)
    csv_path = results_dir / f"metrics_{args.split}_{experiment_name}.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("Saved metrics CSV: %s", csv_path)

    # ── Save thresholds JSON ──────────────────────────────────────────────────
    thresh_data = {
        "thresholds_05": {lbl: 0.5 for lbl in CHEXPERT_14_LABELS},
        "thresholds_tuned": tuned_thresholds or {lbl: 0.5 for lbl in CHEXPERT_14_LABELS},
        "temperature": temp_scaler.temperature.item() if temp_scaler is not None else 1.0,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "split": args.split,
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
    }
    thresh_path = results_dir / f"thresholds_{experiment_name}.json"
    with open(thresh_path, "w") as f:
        json.dump(thresh_data, f, indent=2)
    log.info("Saved thresholds JSON: %s", thresh_path)

    # ── Save confusion counts ─────────────────────────────────────────────────
    confusion_data = {
        lbl: {k: results[lbl][k] for k in ["TP", "FP", "TN", "FN", "n_pos", "n_neg"]}
        for lbl in CHEXPERT_14_LABELS
    }
    conf_path = results_dir / f"confusion_{args.split}_{experiment_name}.json"
    with open(conf_path, "w") as f:
        json.dump(confusion_data, f, indent=2)
    log.info("Saved confusion JSON: %s", conf_path)

    # ── Save calibration summary ──────────────────────────────────────────────
    cal_data = {
        "ece_before": ece_before,
        "ece_after":  ece_after,
        "temperature": temp_scaler.temperature.item() if temp_scaler is not None else 1.0,
        "macro_auroc": results["_macro"]["mean_auroc"],
        "macro_auprc": results["_macro"]["mean_auprc"],
        "macro_f1":    results["_macro"]["macro_f1_05"],
    }
    cal_path = results_dir / f"calibration_{args.split}_{experiment_name}.json"
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)
    log.info("Saved calibration JSON: %s", cal_path)

    print(f"\nMacro AUROC: {results['_macro']['mean_auroc']:.4f}")
    print(f"Macro AUPRC: {results['_macro']['mean_auprc']:.4f}")
    print(f"ECE before/after calibration: {ece_before:.4f} → {ece_after:.4f}")


if __name__ == "__main__":
    main()

