"""
scripts/evaluate_chexpert_b1_v3.py
====================================
Standalone evaluation for CheXpertV3Model.

Features:
  - Excludes uncertain labels (raw == -1) from metric computation
  - Per-class AUROC / AUPRC / F1
  - Per-class threshold tuning on val → frozen before test
  - Head/medium/tail group metrics
  - Per-class calibration (ECE)
  - Bootstrap 95% CIs (1000 iterations, configurable)
  - Saves results to JSON + prints table

Usage:
    python scripts/evaluate_chexpert_b1_v3.py \
        --config configs/chexpert_b1_v3.yaml \
        --checkpoint checkpoints/chexpert_b1_v3/chexpert_b1_v3_uslfa_v3_stageB_best.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v3 import CheXpertV3Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("eval_v3")

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# Head (top 4 by freq), Medium (next 5), Tail (last 5) — from B1v2 training data prevalence
HEAD_CLASSES   = ["Support Devices", "No Finding", "Pleural Effusion", "Edema"]
MEDIUM_CLASSES = ["Cardiomegaly", "Lung Opacity", "Atelectasis", "Consolidation", "Pneumothorax"]
TAIL_CLASSES   = ["Enlarged Cardiomediastinum", "Lung Lesion", "Pneumonia", "Pleural Other", "Fracture"]


# ── Dataset (same as train script, raw labels) ────────────────────────────────

def _get_transforms(res: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CheXpertEvalDataset(Dataset):
    CACHE = PROJECT_ROOT / "data" / "chexpert_cache"

    def __init__(self, df: pd.DataFrame, res: int = 224, cache_dir: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.transform = _get_transforms(res)
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = self.cache_dir / f"{row['file_id']}{ext}"
            if p.exists():
                img_path = p
                break
        try:
            img = Image.open(img_path).convert("RGB") if img_path else Image.new("RGB", (224, 224), 0)
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        img = self.transform(img)
        raw = [float(row.get(l, float("nan"))) if not pd.isna(row.get(l, float("nan"))) else float("nan")
               for l in CHEXPERT_14_LABELS]
        return img, torch.tensor(raw, dtype=torch.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (probs [N,14], raw_labels [N,14])."""
    model.eval()
    all_probs, all_raw = [], []
    for imgs, raw in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, _ = model(imgs)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_raw.append(raw.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_raw, axis=0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_per_class_metrics(
    probs: np.ndarray,
    raw: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    exclude_uncertain: bool = True,
) -> Dict:
    C = len(CHEXPERT_14_LABELS)
    results = {}
    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        raw_col = raw[:, i]
        # Build observation mask
        if exclude_uncertain:
            obs = ~np.isnan(raw_col) & (raw_col != -1.0)
        else:
            obs = ~np.isnan(raw_col)
        t = (raw_col[obs] == 1.0).astype(float)
        p = probs[obs, i]
        thr = (thresholds or {}).get(lbl, 0.5)

        entry = {"n_obs": int(obs.sum()), "n_pos": int(t.sum()), "threshold": thr}
        if obs.sum() == 0 or len(np.unique(t)) < 2:
            entry.update({"auroc": None, "auprc": None, "f1": None})
        else:
            try:
                entry["auroc"] = float(roc_auc_score(t, p))
                entry["auprc"] = float(average_precision_score(t, p))
            except Exception:
                entry["auroc"] = None
                entry["auprc"] = None
            preds = (p >= thr).astype(int)
            entry["f1"] = float(f1_score(t.astype(int), preds, zero_division=0))
        results[lbl] = entry
    return results


def tune_thresholds(
    probs: np.ndarray,
    raw: np.ndarray,
    metric: str = "f1",
    exclude_uncertain: bool = True,
) -> Dict[str, float]:
    """Find optimal per-class threshold on val set by sweeping [0.1, 0.9]."""
    thresholds = {}
    thr_grid = np.arange(0.1, 0.91, 0.02)
    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        raw_col = raw[:, i]
        obs = ~np.isnan(raw_col) & (raw_col != -1.0) if exclude_uncertain else ~np.isnan(raw_col)
        t = (raw_col[obs] == 1.0).astype(float)
        p = probs[obs, i]
        if obs.sum() == 0 or len(np.unique(t)) < 2:
            thresholds[lbl] = 0.5
            continue
        best_thr, best_score = 0.5, -1.0
        for thr in thr_grid:
            preds = (p >= thr).astype(int)
            score = f1_score(t.astype(int), preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[lbl] = best_thr
        log.info("  Threshold tuned  %-35s  %.2f  (F1=%.4f)", lbl, best_thr, best_score)
    return thresholds


def bootstrap_cis(
    probs: np.ndarray,
    raw: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
    exclude_uncertain: bool = True,
) -> Dict[str, Dict]:
    """Bootstrap 95% CI on AUROC per class."""
    rng = np.random.default_rng(seed)
    N = probs.shape[0]
    per_class_auroc_boot = {lbl: [] for lbl in CHEXPERT_14_LABELS}

    for _ in range(n_iter):
        idx = rng.integers(0, N, size=N)
        p_boot = probs[idx]
        r_boot = raw[idx]
        for i, lbl in enumerate(CHEXPERT_14_LABELS):
            rc = r_boot[:, i]
            obs = ~np.isnan(rc) & (rc != -1.0) if exclude_uncertain else ~np.isnan(rc)
            t = (rc[obs] == 1.0).astype(float)
            p = p_boot[obs, i]
            if obs.sum() < 10 or len(np.unique(t)) < 2:
                continue
            try:
                per_class_auroc_boot[lbl].append(float(roc_auc_score(t, p)))
            except Exception:
                pass

    cis = {}
    for lbl in CHEXPERT_14_LABELS:
        vals = per_class_auroc_boot[lbl]
        if len(vals) < 100:
            cis[lbl] = {"mean": None, "ci_low": None, "ci_high": None}
        else:
            arr = np.array(vals)
            cis[lbl] = {
                "mean":    float(np.mean(arr)),
                "ci_low":  float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }
    return cis


def compute_ece(probs: np.ndarray, raw: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Expected Calibration Error per class (macro average)."""
    eces = {}
    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        rc = raw[:, i]
        obs = ~np.isnan(rc) & (rc != -1.0)
        t = (rc[obs] == 1.0).astype(float)
        p = probs[obs, i]
        if obs.sum() < 10:
            eces[lbl] = None
            continue
        # Bin calibration
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (p >= bin_edges[b]) & (p < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            acc = t[mask].mean()
            conf = p[mask].mean()
            ece += abs(acc - conf) * mask.sum() / len(p)
        eces[lbl] = float(ece)
    return eces


def group_summary(per_class: Dict, group: List[str], metric: str) -> Optional[float]:
    vals = [per_class[l][metric] for l in group if l in per_class and per_class[l][metric] is not None]
    return float(np.mean(vals)) if vals else None


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(
    cfg: dict,
    checkpoint: str,
    split: str = "val",
    n_bootstrap: int = 1000,
    tune_thr: bool = True,
    out_json: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = cfg.get("input_resolution", 224)
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/chexpert_cache")
    index_csv = cfg.get("data", {}).get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv")
    val_frac   = cfg.get("data", {}).get("val_frac",   0.10)
    test_frac  = cfg.get("data", {}).get("test_frac",  0.10)
    split_seed = cfg.get("data", {}).get("split_seed", 42)

    # Load data
    df = pd.read_csv(index_csv)

    import re as _re
    def pid(fn):
        m = _re.search(r"(patient\d+)", str(fn))
        return m.group(1) if m else str(fn)

    if "local_filename" in df.columns:
        pid_col = df["local_filename"].apply(pid)
    elif "patient_id" in df.columns:
        pid_col = df["patient_id"]
    else:
        pid_col = df["file_id"].apply(lambda x: x.split(".")[0])

    pids = pid_col.unique()
    rng = np.random.default_rng(split_seed)
    rng.shuffle(pids)
    n = len(pids)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    test_pids = set(pids[:n_test])
    val_pids  = set(pids[n_test:n_test + n_val])
    train_pids = set(pids[n_test + n_val:])

    if split == "test":
        eval_df = df[pid_col.isin(test_pids)].reset_index(drop=True)
        val_df  = df[pid_col.isin(val_pids)].reset_index(drop=True)
    elif split == "val":
        eval_df = df[pid_col.isin(val_pids)].reset_index(drop=True)
        val_df  = eval_df
    else:
        eval_df = df[pid_col.isin(train_pids)].reset_index(drop=True)
        val_df  = df[pid_col.isin(val_pids)].reset_index(drop=True)

    log.info("Evaluating on %s split: %d rows", split, len(eval_df))

    # Build model
    model_cfg = {k: v for k, v in cfg.items()
                 if k not in ("data", "sampler", "loss", "training", "eval", "logging", "experiment")}
    model = CheXpertV3Model.load_checkpoint_compat(checkpoint, model_cfg).to(device)

    # DataLoader
    batch_size = cfg.get("eval", {}).get("val_batch_size", 64)
    eval_ds = CheXpertEvalDataset(eval_df, res=res, cache_dir=cache_dir)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Inference
    log.info("Running inference...")
    probs, raw = run_inference(model, eval_loader, device)

    # Threshold tuning on val (if eval != val, run val separately)
    thresholds = None
    if tune_thr:
        log.info("Tuning thresholds on val split...")
        if split != "val":
            val_ds = CheXpertEvalDataset(val_df, res=res, cache_dir=cache_dir)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            val_probs, val_raw = run_inference(model, val_loader, device)
        else:
            val_probs, val_raw = probs, raw
        thresholds = tune_thresholds(val_probs, val_raw)

    # Per-class metrics
    per_class = compute_per_class_metrics(probs, raw, thresholds=thresholds)

    # Bootstrap CIs
    log.info("Computing bootstrap CIs (%d iterations)...", n_bootstrap)
    cis = bootstrap_cis(probs, raw, n_iter=n_bootstrap)

    # ECE
    eces = compute_ece(probs, raw)

    # Macro averages
    valid_auroc = [per_class[l]["auroc"] for l in CHEXPERT_14_LABELS if per_class[l]["auroc"] is not None]
    valid_auprc = [per_class[l]["auprc"] for l in CHEXPERT_14_LABELS if per_class[l]["auprc"] is not None]
    macro_auroc = float(np.mean(valid_auroc)) if valid_auroc else 0.0
    macro_auprc = float(np.mean(valid_auprc)) if valid_auprc else 0.0

    # Group summaries
    head_auroc   = group_summary(per_class, HEAD_CLASSES,   "auroc")
    medium_auroc = group_summary(per_class, MEDIUM_CLASSES, "auroc")
    tail_auroc   = group_summary(per_class, TAIL_CLASSES,   "auroc")
    head_auprc   = group_summary(per_class, HEAD_CLASSES,   "auprc")
    medium_auprc = group_summary(per_class, MEDIUM_CLASSES, "auprc")
    tail_auprc   = group_summary(per_class, TAIL_CLASSES,   "auprc")

    macro_ece = float(np.mean([v for v in eces.values() if v is not None])) if eces else None

    # ── Print table ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("EVALUATION RESULTS — %s split | checkpoint: %s", split.upper(), Path(checkpoint).name)
    log.info("=" * 80)
    log.info("%-35s  %6s  %6s  %6s  %6s  CI_low  CI_hi", "Class", "AUROC", "AUPRC", "F1", "ECE")
    log.info("-" * 80)
    for lbl in CHEXPERT_14_LABELS:
        e = per_class[lbl]
        ci = cis[lbl]
        auroc = f"{e['auroc']:.4f}" if e["auroc"] is not None else "  N/A "
        auprc = f"{e['auprc']:.4f}" if e["auprc"] is not None else "  N/A "
        f1v   = f"{e['f1']:.4f}"   if e["f1"]    is not None else "  N/A "
        ece   = f"{eces[lbl]:.4f}" if eces[lbl]  is not None else "  N/A "
        ci_lo = f"{ci['ci_low']:.4f}"  if ci["ci_low"]  is not None else " N/A  "
        ci_hi = f"{ci['ci_high']:.4f}" if ci["ci_high"] is not None else " N/A  "
        log.info("  %-33s  %s  %s  %s  %s  %s  %s", lbl, auroc, auprc, f1v, ece, ci_lo, ci_hi)
    log.info("-" * 80)
    log.info("  %-33s  %.4f  %.4f  %6s  %.4f", "MACRO (all 14)", macro_auroc, macro_auprc, "", macro_ece or 0.0)
    log.info("")
    log.info("  Group AUROC — Head: %.4f | Medium: %.4f | Tail: %.4f",
             head_auroc or 0.0, medium_auroc or 0.0, tail_auroc or 0.0)
    log.info("  Group AUPRC — Head: %.4f | Medium: %.4f | Tail: %.4f",
             head_auprc or 0.0, medium_auprc or 0.0, tail_auprc or 0.0)
    log.info("=" * 80)

    # CheXpert Competition 5-class (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)
    core5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    core5_auroc = group_summary(per_class, core5, "auroc")
    core5_auprc = group_summary(per_class, core5, "auprc")
    log.info("  CheXpert Core-5 AUROC: %.4f | AUPRC: %.4f", core5_auroc or 0.0, core5_auprc or 0.0)
    log.info("=" * 80)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "split":        split,
        "checkpoint":   str(checkpoint),
        "macro_auroc":  macro_auroc,
        "macro_auprc":  macro_auprc,
        "macro_ece":    macro_ece,
        "core5_auroc":  core5_auroc,
        "core5_auprc":  core5_auprc,
        "head_auroc":   head_auroc,
        "medium_auroc": medium_auroc,
        "tail_auroc":   tail_auroc,
        "head_auprc":   head_auprc,
        "medium_auprc": medium_auprc,
        "tail_auprc":   tail_auprc,
        "per_class":    per_class,
        "bootstrap_cis": cis,
        "ece_per_class": eces,
        "thresholds":   thresholds,
    }

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Results saved → %s", out_json)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="val", choices=["train", "val", "test"])
    parser.add_argument("--bootstrap",  type=int, default=1000)
    parser.add_argument("--no_tune_thr", action="store_true")
    parser.add_argument("--out_json",   default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_json = args.out_json or f"outputs/chexpert_b1_v3/eval_{args.split}_{Path(args.checkpoint).stem}.json"
    evaluate(cfg, args.checkpoint, split=args.split,
             n_bootstrap=args.bootstrap, tune_thr=not args.no_tune_thr, out_json=out_json)


if __name__ == "__main__":
    main()

