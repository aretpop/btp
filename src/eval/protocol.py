"""
src/eval/protocol.py
====================
Unified evaluation and post-processing protocol for CheXpert V4 ("v4_gap").

Requirements:
- Evaluation label convention: blank -> negative (0.0), uncertain (-1.0) EXCLUDED (mask=0).
- Per class report:
    n_pos, n_neg, prevalence (p), AUROC, AUPRC, AUPRC lift = (AUPRC - p)/(1 - p),
    F1 at tuned threshold, all-positive baseline F1 = 2p/(1 + p),
    precision, recall, ECE before and after calibration.
- Patient-level bootstrap 95% CIs (1,000 iterations for final eval; 200 for per-epoch val).
- Macro averages and group metrics (head / medium / tail).
    tail: Pneumonia, Pleural Other, Fracture, Lung Lesion, Enlarged Cardiomediastinum, Consolidation
    medium: Atelectasis, Pneumothorax, Edema
    head: Cardiomegaly, Lung Opacity, Pleural Effusion, Support Devices, No Finding
- Calibration: per-class temperature scaling fitted on VAL logits only (LBFGS), saved to temperatures.json.
- Thresholds: per-class threshold tuned on VAL calibrated probabilities to maximize F1, saved to thresholds.json.
  Classes with < 20 val positives are flagged and use 0.5.
- Test evaluated once with frozen thresholds and temperatures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

from src.data.labels import CHEXPERT_14_LABELS

logger = logging.getLogger(__name__)

TAIL_CLASSES = [
    "Pneumonia", "Pleural Other", "Fracture",
    "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"
]
MEDIUM_CLASSES = ["Atelectasis", "Pneumothorax", "Edema"]
HEAD_CLASSES = [
    "Cardiomegaly", "Lung Opacity", "Pleural Effusion",
    "Support Devices", "No Finding"
]


# ---------------------------------------------------------------------------
# Calibration & Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Per-class temperature scaling optimized via L-BFGS."""

    def __init__(self, num_classes: int = 14):
        super().__init__()
        # Initialize temperature at 1.0 (log_temp = 0.0)
        self.log_temp = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp.to(logits.device)).clamp(min=1e-3, max=100.0)
        return logits / temp.unsqueeze(0)

    def get_temperatures(self) -> List[float]:
        with torch.no_grad():
            temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
            return temp.cpu().tolist()


def fit_temperature_scaling(
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    val_masks: torch.Tensor,
    num_classes: int = 14,
) -> Tuple[TemperatureScaler, Dict[str, float]]:
    """
    Fit per-class temperature parameter on validation logits using LBFGS.
    """
    device = val_logits.device
    scaler = TemperatureScaler(num_classes=num_classes).to(device)
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=50)

    val_logits = val_logits.detach()
    val_targets = val_targets.detach()
    val_masks = val_masks.detach()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(val_logits)
        loss = F.binary_cross_entropy_with_logits(scaled, val_targets, reduction="none")
        masked_loss = (loss * val_masks).sum() / val_masks.sum().clamp(min=1.0)
        masked_loss.backward()
        return masked_loss

    optimizer.step(closure)
    temperatures = scaler.get_temperatures()
    temp_dict = {CHEXPERT_14_LABELS[i]: round(temperatures[i], 4) for i in range(num_classes)}
    return scaler, temp_dict


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary predictions."""
    if len(probs) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs <= bin_upper) if i == n_bins - 1 else (probs >= bin_lower) & (probs < bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(targets[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / total) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Threshold Tuning on Validation
# ---------------------------------------------------------------------------

def tune_thresholds_on_val(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    val_masks: np.ndarray,
    min_positives: int = 20,
) -> Dict[str, float]:
    """
    Tune per-class binary classification threshold on validation set to maximize F1.
    If a class has < min_positives in val, default to 0.5.
    """
    C = val_probs.shape[1]
    thresholds = {}
    threshold_grid = np.linspace(0.02, 0.98, 49)

    for c in range(C):
        cls_name = CHEXPERT_14_LABELS[c]
        valid_idx = np.where(val_masks[:, c] > 0.5)[0]
        if len(valid_idx) == 0:
            thresholds[cls_name] = 0.5
            continue

        y_true = val_targets[valid_idx, c]
        y_prob = val_probs[valid_idx, c]
        n_pos = int(np.sum(y_true == 1.0))

        if n_pos < min_positives:
            logger.info(f"Class '{cls_name}' has {n_pos} < {min_positives} val positives. Using default threshold 0.5.")
            thresholds[cls_name] = 0.5
            continue

        best_th = 0.5
        best_f1 = -1.0
        for th in threshold_grid:
            y_pred = (y_prob >= th).astype(float)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th

        thresholds[cls_name] = round(float(best_th), 4)

    return thresholds


# ---------------------------------------------------------------------------
# Patient-Level Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_patient_metrics(
    patient_ids: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    thresholds: Dict[str, float],
    n_bootstraps: int = 200,
    seed: int = 42,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compute 95% patient-level bootstrap confidence intervals for AUROC, AUPRC, and F1.
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    rng = np.random.RandomState(seed)

    # Pre-index samples per patient for fast grouping
    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    C = probs.shape[1]
    boot_aurocs = {c: [] for c in range(C)}
    boot_auprcs = {c: [] for c in range(C)}
    boot_f1s = {c: [] for c in range(C)}

    for _ in range(n_bootstraps):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = []
        for pid in sampled_pids:
            sampled_indices.extend(patient_to_indices[pid])

        b_targets = targets[sampled_indices]
        b_probs = probs[sampled_indices]
        b_masks = masks[sampled_indices]

        for c in range(C):
            v_idx = np.where(b_masks[:, c] > 0.5)[0]
            if len(v_idx) == 0:
                continue
            y_t = b_targets[v_idx, c]
            y_p = b_probs[v_idx, c]
            if len(np.unique(y_t)) < 2:
                continue

            try:
                auc_roc = roc_auc_score(y_t, y_p)
                boot_aurocs[c].append(auc_roc)
            except Exception:
                pass

            try:
                p_prec, p_rec, _ = precision_recall_curve(y_t, y_p)
                auc_pr = auc(p_rec, p_prec)
                boot_auprcs[c].append(auc_pr)
            except Exception:
                pass

            th = thresholds.get(CHEXPERT_14_LABELS[c], 0.5)
            y_pred = (y_p >= th).astype(float)
            boot_f1s[c].append(f1_score(y_t, y_pred, zero_division=0))

    cis = {}
    for c in range(C):
        cls_name = CHEXPERT_14_LABELS[c]
        cis[cls_name] = {}
        for metric_name, val_list in [("auroc", boot_aurocs[c]), ("auprc", boot_auprcs[c]), ("f1", boot_f1s[c])]:
            if len(val_list) >= 10:
                low = float(np.percentile(val_list, 2.5))
                high = float(np.percentile(val_list, 97.5))
                cis[cls_name][metric_name] = (round(low, 4), round(high, 4))
            else:
                cis[cls_name][metric_name] = (None, None)

    return cis


# ---------------------------------------------------------------------------
# Comprehensive Evaluation Protocol
# ---------------------------------------------------------------------------

def evaluate_predictions(
    logits: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    scaler: Optional[TemperatureScaler] = None,
    n_bootstraps: int = 1000,
    min_positives_lift: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """
    Full evaluation protocol per Section 7 specification.
    Returns:
        metrics_df: per-class detailed metrics DataFrame
        summary: dict containing macro, group, and lift metrics
        calibrated_probs: calibrated probabilities array [N, C]
    """
    C = logits.shape[1]
    raw_probs = 1.0 / (1.0 + np.exp(-logits))

    if scaler is not None:
        with torch.no_grad():
            t_logits = torch.from_numpy(logits).float()
            scaled_t = scaler(t_logits)
            calibrated_probs = torch.sigmoid(scaled_t).cpu().numpy()
    else:
        calibrated_probs = raw_probs.copy()

    if thresholds is None:
        thresholds = {CHEXPERT_14_LABELS[c]: 0.5 for c in range(C)}

    # Compute bootstrap CIs
    cis = (
        bootstrap_patient_metrics(
            patient_ids=patient_ids,
            probs=calibrated_probs,
            targets=targets,
            masks=masks,
            thresholds=thresholds,
            n_bootstraps=n_bootstraps,
        )
        if n_bootstraps > 0
        else {}
    )

    records = []
    aurocs = []
    auprcs = []
    f1s = []
    lifts_eligible = []

    for c in range(C):
        cls_name = CHEXPERT_14_LABELS[c]
        v_idx = np.where(masks[:, c] > 0.5)[0]

        if len(v_idx) == 0:
            continue

        y_true = targets[v_idx, c]
        y_prob = calibrated_probs[v_idx, c]
        y_prob_raw = raw_probs[v_idx, c]

        n_pos = int(np.sum(y_true == 1.0))
        n_neg = int(np.sum(y_true == 0.0))
        n_total = n_pos + n_neg
        prev = n_pos / max(1, n_total)

        # Baseline all-positive F1: 2p / (1 + p)
        all_pos_f1 = (2.0 * prev) / max(1e-8, 1.0 + prev)

        th = thresholds.get(cls_name, 0.5)
        y_pred = (y_prob >= th).astype(float)

        # AUROC
        try:
            auc_roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        except Exception:
            auc_roc = None

        # AUPRC
        try:
            if len(np.unique(y_true)) > 1:
                p_prec, p_rec, _ = precision_recall_curve(y_true, y_prob)
                auc_pr = float(auc(p_rec, p_prec))
            else:
                auc_pr = None
        except Exception:
            auc_pr = None

        # AUPRC lift: (AUPRC - p) / (1 - p)
        if auc_pr is not None and prev < 1.0:
            auprc_lift = (auc_pr - prev) / max(1e-8, 1.0 - prev)
        else:
            auprc_lift = 0.0

        if n_pos >= min_positives_lift and auc_pr is not None:
            lifts_eligible.append(auprc_lift)

        cls_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cls_prec = float(precision_score(y_true, y_pred, zero_division=0))
        cls_rec = float(recall_score(y_true, y_pred, zero_division=0))

        ece_before = compute_ece(y_prob_raw, y_true)
        ece_after = compute_ece(y_prob, y_true)

        if auc_roc is not None:
            aurocs.append(auc_roc)
        if auc_pr is not None:
            auprcs.append(auc_pr)
        f1s.append(cls_f1)

        ci_info = cis.get(cls_name, {})
        roc_ci = ci_info.get("auroc", (None, None))
        pr_ci = ci_info.get("auprc", (None, None))
        f1_ci = ci_info.get("f1", (None, None))

        records.append({
            "class_name": cls_name,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": round(prev, 5),
            "threshold": round(th, 4),
            "auroc": round(auc_roc, 4) if auc_roc is not None else None,
            "auroc_ci_low": roc_ci[0],
            "auroc_ci_high": roc_ci[1],
            "auprc": round(auc_pr, 4) if auc_pr is not None else None,
            "auprc_ci_low": pr_ci[0],
            "auprc_ci_high": pr_ci[1],
            "auprc_lift": round(auprc_lift, 4),
            "f1": round(cls_f1, 4),
            "f1_ci_low": f1_ci[0],
            "f1_ci_high": f1_ci[1],
            "all_pos_f1": round(all_pos_f1, 4),
            "precision": round(cls_prec, 4),
            "recall": round(cls_rec, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
        })

    metrics_df = pd.DataFrame(records)

    # Group metrics
    def _group_mean(classes, col):
        sub = metrics_df[metrics_df["class_name"].isin(classes)]
        vals = sub[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) > 0 else 0.0

    macro_auroc = round(float(np.mean(aurocs)), 4) if len(aurocs) > 0 else 0.0
    macro_auprc = round(float(np.mean(auprcs)), 4) if len(auprcs) > 0 else 0.0
    macro_f1 = round(float(np.mean(f1s)), 4) if len(f1s) > 0 else 0.0
    macro_lift = round(float(np.mean(lifts_eligible)), 4) if len(lifts_eligible) > 0 else 0.0

    summary = {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": macro_f1,
        "macro_auprc_lift": macro_lift,
        "head_auroc": _group_mean(HEAD_CLASSES, "auroc"),
        "medium_auroc": _group_mean(MEDIUM_CLASSES, "auroc"),
        "tail_auroc": _group_mean(TAIL_CLASSES, "auroc"),
        "head_auprc": _group_mean(HEAD_CLASSES, "auprc"),
        "medium_auprc": _group_mean(MEDIUM_CLASSES, "auprc"),
        "tail_auprc": _group_mean(TAIL_CLASSES, "auprc"),
    }

    return metrics_df, summary, calibrated_probs


def save_markdown_summary(metrics_df: pd.DataFrame, summary: Dict[str, Any], out_path: Path):
    """Save an executive markdown report for test/val evaluation."""
    md = []
    md.append("# CheXpert V4 (v4_gap) Evaluation Report\n")
    md.append("## Macro & Tier Summary\n")
    md.append(f"- **Macro AUROC**: `{summary['macro_auroc']:.4f}`")
    md.append(f"- **Macro AUPRC**: `{summary['macro_auprc']:.4f}`")
    md.append(f"- **Macro AUPRC Lift**: `{summary['macro_auprc_lift']:.4f}`")
    md.append(f"- **Macro F1**: `{summary['macro_f1']:.4f}`\n")
    md.append(f"| Tier | AUROC | AUPRC |")
    md.append(f"| :--- | :---: | :---: |")
    md.append(f"| **Head** (Cardiomegaly, Effusion, Opacity, Devices, No Finding) | `{summary['head_auroc']:.4f}` | `{summary['head_auprc']:.4f}` |")
    md.append(f"| **Medium** (Atelectasis, Pneumothorax, Edema) | `{summary['medium_auroc']:.4f}` | `{summary['medium_auprc']:.4f}` |")
    md.append(f"| **Tail** (Pneumonia, Pleural Other, Fracture, Lesion, ECM, Consolidation) | `{summary['tail_auroc']:.4f}` | `{summary['tail_auprc']:.4f}` |\n")

    md.append("## Per-Class Detailed Performance\n")
    md.append("| Pathology | AUROC (95% CI) | AUPRC (95% CI) | AUPRC Lift | F1 (vs All-Pos) | ECE (Calib) | Pos / Neg Support |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in metrics_df.iterrows():
        roc_ci = f"[{r['auroc_ci_low']}, {r['auroc_ci_high']}]" if r['auroc_ci_low'] is not None else "-"
        pr_ci = f"[{r['auprc_ci_low']}, {r['auprc_ci_high']}]" if r['auprc_ci_low'] is not None else "-"
        f1_comp = f"{r['f1']:.4f} ({r['all_pos_f1']:.4f})"
        ece_comp = f"{r['ece_before']:.3f} -> {r['ece_after']:.3f}"
        md.append(f"| **{r['class_name']}** | {r['auroc']} {roc_ci} | {r['auprc']} {pr_ci} | {r['auprc_lift']:.4f} | {f1_comp} | {ece_comp} | {r['n_pos']:,} / {r['n_neg']:,} |")

    out_path.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Saved executive markdown report to {out_path}")

