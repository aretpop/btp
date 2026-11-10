"""
scripts/train_chexpert_b1_v2.py
================================
Two-stage training script for CheXpertB1V2Model.

Stage A: Freeze backbone, train head only (5 epochs, lr=1e-3).
Stage B: Partial backbone unfreeze, cosine schedule with warmup (20 epochs).

Usage:
    python scripts/train_chexpert_b1_v2.py --config configs/chexpert_b1_v2.yaml
    python scripts/train_chexpert_b1_v2.py --config configs/chexpert_b1_v2.yaml \\
        --override model.fuse_block5=false loss.pos_weight_mode=cohort_sqrt
    python scripts/train_chexpert_b1_v2.py --config configs/chexpert_b1_v2.yaml --smoke_test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model

# ── MLflow (optional) ─────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    warnings.warn("mlflow not installed — experiment tracking disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────
CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
NUM_CLASSES = len(CHEXPERT_14_LABELS)

# Pool disease assignments (from B0 training)
POOL1_DISEASES = [  # rare / focal
    "Pneumonia", "Pleural Other", "Fracture",
    "Lung Lesion", "Consolidation", "Enlarged Cardiomediastinum",
]
POOL2_DISEASES = [  # common / diffuse
    "Cardiomegaly", "Edema", "Pleural Effusion",
    "Pneumothorax", "Atelectasis", "Lung Opacity",
]

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FMT = "%(asctime)s  %(levelname)-7s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_LOG_DATEFMT,
)
log = logging.getLogger("train_b1v2")


def _add_file_handler(log_path: str) -> None:
    """Re-configure the root logger to tee output to a log file on disk.

    Uses force=True (Python ≥3.8) to replace the existing StreamHandler
    with [StreamHandler(stderr) + FileHandler(log_path)], so every
    log.info() call from this point forward writes to both destinations.
    """
    import os
    log_path = os.path.abspath(log_path)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
    # force=True clears existing handlers on the root logger and re-adds ours
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        datefmt=_LOG_DATEFMT,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    log.info("Logging to file: %s", log_path)


# =============================================================================
# Config helpers (plain yaml + argparse, no omegaconf required)
# =============================================================================

def _deep_set(d: dict, key_path: str, value):
    """Set a nested dict value using dot-separated key path."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to coerce the value to the right type
    orig = d.get(keys[-1])
    if orig is not None:
        try:
            if isinstance(orig, bool):
                value = value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
            elif isinstance(orig, int):
                value = int(value)
            elif isinstance(orig, float):
                value = float(value)
            elif isinstance(orig, list):
                value = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
    d[keys[-1]] = value


def load_config(config_path: str, overrides: List[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        k, v = override.split("=", 1)
        _deep_set(cfg, k.strip(), v.strip())
        log.info("Override: %s = %s", k.strip(), v.strip())
    return cfg


def cfg_get(cfg: dict, *keys, default=None):
    """Safely get nested dict value."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# =============================================================================
# Data utilities  (self-contained — no external src.datasets.chexpert_local_v2)
# =============================================================================

import re as _re_module

def _extract_patient_id_from_local_filename(local_filename: str) -> str:
    """Extract patientXXXXX from local_filename (e.g. 'patient04528/study1/view1_frontal.png')."""
    m = _re_module.search(r'(patient\d+)', str(local_filename))
    return m.group(1) if m else str(local_filename)


def make_patient_splits(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified splits (no patient leakage).
    Extracts patient ID from 'local_filename' via regex (patient\\d+).
    Falls back to 'file_id' column if 'local_filename' is not present.
    """
    rng = np.random.default_rng(seed)

    # Derive patient ID: prefer local_filename regex, fall back to file_id
    if "local_filename" in df.columns:
        pid_series = df["local_filename"].apply(_extract_patient_id_from_local_filename)
    elif "patient_id" in df.columns:
        pid_series = df["patient_id"]
    else:
        pid_series = df["file_id"].apply(lambda x: x.split(".")[0])

    unique_pids = pid_series.unique()
    rng.shuffle(unique_pids)

    n = len(unique_pids)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    test_pids  = set(unique_pids[:n_test])
    val_pids   = set(unique_pids[n_test: n_test + n_val])
    train_pids = set(unique_pids[n_test + n_val:])

    train_df = df[pid_series.isin(train_pids)].reset_index(drop=True)
    val_df   = df[pid_series.isin(val_pids)].reset_index(drop=True)
    test_df  = df[pid_series.isin(test_pids)].reset_index(drop=True)

    log.info(
        "Patient splits — unique patients: total=%d, train=%d, val=%d, test=%d | "
        "rows: train=%d, val=%d, test=%d",
        n, len(train_pids), len(val_pids), len(test_pids),
        len(train_df), len(val_df), len(test_df),
    )

    return train_df, val_df, test_df


def assert_no_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Runtime assertion: zero patient overlap across splits."""
    def get_pids(df):
        if "local_filename" in df.columns:
            return set(df["local_filename"].apply(_extract_patient_id_from_local_filename).unique())
        elif "patient_id" in df.columns:
            return set(df["patient_id"].unique())
        return set(df["file_id"].apply(lambda x: x.split(".")[0]).unique())

    train_pids = get_pids(train_df)
    val_pids = get_pids(val_df)
    test_pids = get_pids(test_df)

    tv = train_pids & val_pids
    tt = train_pids & test_pids
    vt = val_pids & test_pids
    assert len(tv) == 0, f"Patient leakage train∩val: {len(tv)} patients"
    assert len(tt) == 0, f"Patient leakage train∩test: {len(tt)} patients"
    assert len(vt) == 0, f"Patient leakage val∩test: {len(vt)} patients"
    log.info(
        "✓ Zero patient leakage confirmed — train=%d, val=%d, test=%d unique patients",
        len(train_pids), len(val_pids), len(test_pids),
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

def _get_transforms(input_resolution: int, split: str):
    from torchvision import transforms
    if split == "train":
        return transforms.Compose([
            transforms.Resize((input_resolution, input_resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class CheXpertLocalDatasetV2(Dataset):
    """
    Multi-label CheXpert dataset from materialized local cache.
    Returns (img_tensor, labels [14], mask [14]).
    mask[i] = 1 if label is observed (not uncertain/nan), 0 if unobserved.
    """

    CACHE_DIR = PROJECT_ROOT / "data" / "chexpert_cache"

    def __init__(self, df: pd.DataFrame, split: str = "train", cfg: dict = None):
        self.df = df.reset_index(drop=True)
        self.split = split
        cfg = cfg or {}
        self.uncertain_policy = cfg.get("uncertain_policy", "u_zeros")
        input_resolution = cfg.get("model_input_resolution", 224)
        cache_dir = cfg.get("cache_dir", str(self.CACHE_DIR))
        self.cache_dir = Path(cache_dir)
        self.transform = _get_transforms(input_resolution, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        fid = row["file_id"]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = self.cache_dir / f"{fid}{ext}"
            if p.exists():
                img_path = p
                break

        try:
            img = Image.open(img_path).convert("RGB") if img_path else Image.new("RGB", (224, 224), 0)
        except Exception:
            img = Image.new("RGB", (224, 224), 0)

        img = self.transform(img)

        # Build labels and mask
        labels = []
        mask = []
        for lbl in CHEXPERT_14_LABELS:
            raw = row.get(lbl, float("nan"))
            if pd.isna(raw):
                # Missing → treat as negative + masked out
                labels.append(0.0)
                mask.append(0.0)
            elif raw == -1.0:  # uncertain
                if self.uncertain_policy == "u_ones":
                    labels.append(1.0)
                    mask.append(1.0)
                elif self.uncertain_policy == "ignore":
                    labels.append(0.0)
                    mask.append(0.0)  # masked out
                elif self.uncertain_policy == "smooth":
                    labels.append(0.70)  # Stanford CheXpert competition U-smooth soft target
                    mask.append(1.0)
                else:  # u_zeros (default)
                    labels.append(0.0)
                    mask.append(1.0)
            else:
                labels.append(float(raw))
                mask.append(1.0)

        return (
            img,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


# ── Sampler ───────────────────────────────────────────────────────────────────

class TieredSamplerV2(Sampler):
    """
    Three-pool tiered batch sampler.
    pool_counts = [n_rare, n_common, n_normal] summing to batch_size.

    intra_pool_mode:
      - "sqrt_freq": sample from class buckets proportional to sqrt(class_freq)
      - "uniform": round-robin across buckets
      - "random": sample uniformly from each pool

    Returns batched indices (batch_sampler style).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pool_counts: List[int] = None,
        labels: List[str] = None,
        batches_per_epoch: int = 800,
        batch_size: int = 32,
        intra_pool_mode: str = "sqrt_freq",
        seed: int = 42,
    ):
        self.batches_per_epoch = batches_per_epoch
        self.intra_pool_mode = intra_pool_mode
        self.rng = np.random.default_rng(seed)
        labels = labels or CHEXPERT_14_LABELS

        pool_counts = pool_counts or [14, 12, 6]
        self.pool_counts = pool_counts

        # Build disease buckets for Pool 1 (rare) and Pool 2 (common)
        p1_diseases = [l for l in POOL1_DISEASES if l in labels]
        p2_diseases = [l for l in POOL2_DISEASES if l in labels]

        self.p1_buckets = [df[df[d] == 1.0].index.values.copy() for d in p1_diseases if d in df.columns]
        self.p2_buckets = [df[df[d] == 1.0].index.values.copy() for d in p2_diseases if d in df.columns]

        # Pool 3: "No Finding" (clean normals)
        if "No Finding" in df.columns:
            nf_mask = df["No Finding"] == 1.0
            self.p3_indices = df[nf_mask].index.values.copy()
        else:
            self.p3_indices = df.index.values.copy()

        # Shuffle buckets initially
        for b in self.p1_buckets + self.p2_buckets:
            self.rng.shuffle(b)
        self.rng.shuffle(self.p3_indices)

        # Compute sqrt_freq weights if needed
        if intra_pool_mode == "sqrt_freq":
            self.p1_weights = np.array([math.sqrt(len(b)) for b in self.p1_buckets], dtype=float)
            self.p1_weights /= self.p1_weights.sum()
            self.p2_weights = np.array([math.sqrt(len(b)) for b in self.p2_buckets], dtype=float)
            self.p2_weights /= self.p2_weights.sum()

        # Pointers for round-robin
        self.p1_ptrs = [0] * len(self.p1_buckets)
        self.p2_ptrs = [0] * len(self.p2_buckets)
        self.p3_ptr = 0

    def _sample_pool(self, buckets, ptrs, count, weights=None):
        samples = []
        if not buckets:
            return list(self.rng.integers(0, max(1, len(buckets)), size=count))
        for _ in range(count):
            if weights is not None:
                b_idx = self.rng.choice(len(buckets), p=weights)
            else:
                b_idx = int(self.rng.integers(0, len(buckets)))
            ptr = ptrs[b_idx]
            if ptr >= len(buckets[b_idx]):
                self.rng.shuffle(buckets[b_idx])
                ptrs[b_idx] = 0
                ptr = 0
            samples.append(int(buckets[b_idx][ptr]))
            ptrs[b_idx] += 1
        return samples

    def _sample_p3(self, count):
        samples = []
        for _ in range(count):
            if self.p3_ptr >= len(self.p3_indices):
                self.rng.shuffle(self.p3_indices)
                self.p3_ptr = 0
            samples.append(int(self.p3_indices[self.p3_ptr]))
            self.p3_ptr += 1
        return samples

    def __iter__(self):
        n1, n2, n3 = self.pool_counts
        weights1 = getattr(self, "p1_weights", None) if self.intra_pool_mode == "sqrt_freq" else None
        weights2 = getattr(self, "p2_weights", None) if self.intra_pool_mode == "sqrt_freq" else None

        for _ in range(self.batches_per_epoch):
            b1 = self._sample_pool(self.p1_buckets, self.p1_ptrs, n1, weights1)
            b2 = self._sample_pool(self.p2_buckets, self.p2_ptrs, n2, weights2)
            b3 = self._sample_p3(n3)
            batch = b1 + b2 + b3
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch

    def compute_realized_ratios(self, n_batches: int = 100) -> Dict[str, float]:
        """Sample n_batches and compute fraction from each pool."""
        n1, n2, n3 = self.pool_counts
        total = sum(self.pool_counts)
        return {
            "pool1_rare_frac": n1 / total,
            "pool2_common_frac": n2 / total,
            "pool3_normal_frac": n3 / total,
        }


def compute_sampled_stream_pos_rates(
    dataset: Dataset,
    sampler: TieredSamplerV2,
    labels: List[str],
    n_batches: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample n_batches from the sampler and compute empirical positive rates.
    Returns (empirical_rates [14], natural_rates [14]).
    """
    empirical_counts = np.zeros(len(labels))
    natural_counts = np.zeros(len(labels))
    sampled_n = 0
    natural_n = len(dataset)

    # Natural rates from full dataset
    df = dataset.df
    for i, lbl in enumerate(labels):
        if lbl in df.columns:
            natural_counts[i] = (df[lbl] == 1.0).sum()

    # Empirical rates from sampler stream
    batch_iter = iter(sampler)
    for b_idx in range(min(n_batches, len(sampler))):
        try:
            indices = next(batch_iter)
        except StopIteration:
            break
        for idx in indices:
            if idx < len(dataset):
                row = dataset.df.iloc[idx]
                for i, lbl in enumerate(labels):
                    val = row.get(lbl, 0.0)
                    if not pd.isna(val) and val == 1.0:
                        empirical_counts[i] += 1
        sampled_n += len(indices)

    empirical_rates = empirical_counts / max(sampled_n, 1)
    natural_rates = natural_counts / max(natural_n, 1)
    return empirical_rates, natural_rates


# ── Loss ──────────────────────────────────────────────────────────────────────

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE loss with per-sample label masking.
    Only observed (mask==1) positions contribute to the loss.
    """
    if pos_weight is not None:
        pw = pos_weight.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Apply mask
    loss = loss * mask
    denom = mask.sum()
    if denom < 1.0:
        return loss.sum() * 0.0  # all masked — return 0
    return loss.sum() / denom


# =============================================================================
# EMA helper
# =============================================================================

class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v.float() * (1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into model (use for validation)."""
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})

    def state_dict(self):
        return self.shadow


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict:
    """
    Full validation loop. Returns dict with per-class and macro metrics.
    Skips AUROC/AUPRC for classes with no positives in val set (logs warning).
    """
    model.eval()
    all_logits, all_targets, all_masks = [], [], []

    for imgs, targets, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, _ = model(imgs)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        all_masks.append(masks.cpu())

    logits_all = torch.cat(all_logits, dim=0).numpy()   # [N, 14]
    targets_all = torch.cat(all_targets, dim=0).numpy()  # [N, 14]
    masks_all = torch.cat(all_masks, dim=0).numpy()      # [N, 14]
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))        # sigmoid

    per_class_auroc = {}
    per_class_auprc = {}
    per_class_f1 = {}
    valid_auroc, valid_auprc = [], []

    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        # Only use samples where this label is observed
        obs_mask = masks_all[:, i] > 0.5
        t_obs = targets_all[obs_mask, i]
        p_obs = probs_all[obs_mask, i]

        if obs_mask.sum() == 0 or len(np.unique(t_obs)) < 2:
            log.warning("Val: class '%s' has insufficient positives/negatives — skipping AUROC/AUPRC.", lbl)
            per_class_auroc[lbl] = float("nan")
            per_class_auprc[lbl] = float("nan")
        else:
            try:
                auroc = roc_auc_score(t_obs, p_obs)
                auprc = average_precision_score(t_obs, p_obs)
                per_class_auroc[lbl] = auroc
                per_class_auprc[lbl] = auprc
                valid_auroc.append(auroc)
                valid_auprc.append(auprc)
            except Exception as e:
                log.warning("Error computing metrics for '%s': %s", lbl, e)
                per_class_auroc[lbl] = float("nan")
                per_class_auprc[lbl] = float("nan")

        preds_bin = (p_obs >= 0.5).astype(int)
        per_class_f1[lbl] = f1_score(t_obs, preds_bin, zero_division=0) if len(t_obs) > 0 else 0.0

    # Mean over ALL 14 classes: NaN → 0.0 (class had no positives in val subset)
    # This gives a consistent macro average regardless of val subset size
    all_auroc = [v if not math.isnan(v) else 0.0 for v in per_class_auroc.values()]
    all_auprc = [v if not math.isnan(v) else 0.0 for v in per_class_auprc.values()]
    mean_auroc = float(np.mean(all_auroc))
    mean_auprc = float(np.mean(all_auprc))
    # Also keep the "valid only" means for reference (excludes classes with no positives)
    mean_auroc_valid = float(np.mean(valid_auroc)) if valid_auroc else 0.0
    mean_auprc_valid = float(np.mean(valid_auprc)) if valid_auprc else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    return {
        "per_class_auroc": per_class_auroc,
        "per_class_auprc": per_class_auprc,
        "per_class_f1": per_class_f1,
        "mean_auroc": mean_auroc,           # macro over all 14 (NaN→0)
        "mean_auprc": mean_auprc,           # macro over all 14 (NaN→0)
        "mean_auroc_valid": mean_auroc_valid,  # macro over classes with ≥1 positive
        "mean_auprc_valid": mean_auprc_valid,  # macro over classes with ≥1 positive
        "macro_f1": macro_f1,
        "n_valid_classes": len(valid_auprc),   # how many of 14 had positives
    }


# =============================================================================
# Scheduler helpers
# =============================================================================

def get_lr_scale_warmup(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup: returns LR scale factor in [0, 1] during warmup epochs."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(warmup_epochs)


def cosine_lr_scale(epoch: int, total_epochs: int, eta_min_ratio: float = 1e-7) -> float:
    """Cosine annealing scale from 1.0 → eta_min_ratio over total_epochs."""
    return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * epoch / total_epochs))


# =============================================================================
# Training loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    cfg: dict,
    epoch: int,
    stage: str = "A",
    ema: Optional[ModelEMA] = None,
) -> float:
    model.train()
    # For Stage A, backbone must remain in eval mode (frozen BN)
    if stage == "A":
        model.backbone.eval()

    attn_reg_weight = cfg_get(cfg, "model", "attn_regularizer_weight", default=0.0)
    use_amp = cfg_get(cfg, "training", "use_amp", default=True) and device.type == "cuda"

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        imgs, targets, masks = batch
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, attn_reg = model(imgs)
            loss = masked_bce_loss(logits, targets, masks, pos_weight) + attn_reg

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            log.info(
                "  Stage %s Ep%02d [%d/%d] loss=%.4f",
                stage, epoch, batch_idx + 1, len(loader), loss.item(),
            )

    return total_loss / max(n_batches, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CheXpertB1V2Model")
    parser.add_argument("--config", default="configs/chexpert_b1_v2.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="key=value",
        help="Override config values, e.g. model.fuse_block5=false",
    )
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test: 100 train / 50 val, 1 epoch each stage")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config, args.override)

    # Pull frequently used sub-configs
    data_cfg = cfg.get("data", {})
    sampler_cfg = cfg.get("sampler", {})
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("eval", {})
    log_cfg = cfg.get("logging", {})
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})

    experiment_name = exp_cfg.get("name", "chexpert_b1_v2_baseline")
    checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints/chexpert_b1_v2"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(log_cfg.get("results_dir", "outputs/chexpert_b1_v2"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (always write a .log alongside the checkpoints) ──────────
    log_suffix = "_smoke" if args.smoke_test else ""
    _add_file_handler(str(checkpoint_dir / f"{experiment_name}{log_suffix}.log"))

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Load data ─────────────────────────────────────────────────────────────
    index_csv = Path(data_cfg.get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv"))
    df_all = pd.read_csv(PROJECT_ROOT / index_csv)
    log.info("Loaded index: %d rows", len(df_all))

    # ── Patient-level splits ──────────────────────────────────────────────────
    train_df, val_df, test_df = make_patient_splits(
        df_all,
        val_frac=data_cfg.get("val_frac", 0.10),
        test_frac=data_cfg.get("test_frac", 0.10),
        seed=data_cfg.get("split_seed", 42),
    )
    assert_no_patient_leakage(train_df, val_df, test_df)

    log.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Smoke test truncation ─────────────────────────────────────────────────
    if args.smoke_test:
        log.info("[SMOKE TEST] Truncating to 100 train / 50 val samples, 1 epoch each stage.")
        train_df = train_df.head(100).reset_index(drop=True)
        val_df = val_df.head(50).reset_index(drop=True)
        stage_a_epochs = 1
        stage_b_epochs = 1
        batches_per_epoch = 4
        pos_weight_n_batches = 4
        num_workers = 0
    else:
        stage_a_epochs = train_cfg.get("stage_a_epochs", 5)
        stage_b_epochs = train_cfg.get("stage_b_epochs", 20)
        batches_per_epoch = sampler_cfg.get("batches_per_epoch", 800)
        pos_weight_n_batches = loss_cfg.get("pos_weight_n_batches", 200)
        num_workers = train_cfg.get("num_workers", 4)

    # ── Build datasets ────────────────────────────────────────────────────────
    ds_kwargs = {
        "uncertain_policy": data_cfg.get("uncertain_policy", "u_zeros"),
        "cache_dir": str(PROJECT_ROOT / data_cfg.get("cache_dir", "data/chexpert_cache")),
        "model_input_resolution": model_cfg.get("input_resolution", 224),
    }
    train_ds = CheXpertLocalDatasetV2(train_df, split="train", cfg=ds_kwargs)
    val_ds = CheXpertLocalDatasetV2(val_df, split="val", cfg=ds_kwargs)

    # ── Sampler setup ─────────────────────────────────────────────────────────
    sampler_mode = sampler_cfg.get("mode", "tiered")
    batch_size = sampler_cfg.get("batch_size", 32)
    pin_memory = train_cfg.get("pin_memory", True) and device.type == "cuda"

    if sampler_mode == "tiered":
        batch_sampler = TieredSamplerV2(
            df=train_df,
            pool_counts=sampler_cfg.get("pool_counts", [14, 12, 6]),
            labels=CHEXPERT_14_LABELS,
            batches_per_epoch=batches_per_epoch,
            batch_size=batch_size,
            intra_pool_mode=sampler_cfg.get("intra_pool_mode", "sqrt_freq"),
            seed=sampler_cfg.get("seed", 42),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        realized_ratios = batch_sampler.compute_realized_ratios(n_batches=100)
        log.info("Sampler realized pool ratios: %s", realized_ratios)
    else:
        # Random sampler: wrap in BatchSampler
        rand_sampler = RandomSampler(train_ds)
        from torch.utils.data import BatchSampler
        batch_sampler = BatchSampler(rand_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        log.info("Using RandomSampler (mode=random)")

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_cfg.get("val_batch_size", 64),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── pos_weight computation ────────────────────────────────────────────────
    pos_weight_mode = loss_cfg.get("pos_weight_mode", "sampled_stream")
    pos_weight_tensor: Optional[torch.Tensor] = None

    if pos_weight_mode == "cohort_sqrt":
        cohort_weights = loss_cfg.get("cohort_sqrt_weights", [1.0] * NUM_CLASSES)
        pos_weight_tensor = torch.tensor(cohort_weights, dtype=torch.float32)
        log.info("pos_weight [cohort_sqrt]: %s", [f"{w:.2f}" for w in pos_weight_tensor.tolist()])

    elif pos_weight_mode in ("sampled_stream", "sampled_stream_sqrt"):
        log.info("Computing pos_weight via sampled stream (%d batches)...", pos_weight_n_batches)
        if sampler_mode == "tiered":
            sampler_for_posw = TieredSamplerV2(
                df=train_df,
                pool_counts=sampler_cfg.get("pool_counts", [14, 12, 6]),
                labels=CHEXPERT_14_LABELS,
                batches_per_epoch=pos_weight_n_batches,
                batch_size=batch_size,
                intra_pool_mode=sampler_cfg.get("intra_pool_mode", "sqrt_freq"),
                seed=sampler_cfg.get("seed", 0),
            )
        else:
            sampler_for_posw = batch_sampler  # type: ignore

        empirical_rates, natural_rates = compute_sampled_stream_pos_rates(
            dataset=train_ds,
            sampler=sampler_for_posw,  # type: ignore
            labels=CHEXPERT_14_LABELS,
            n_batches=pos_weight_n_batches,
        )

        log.info("Sampled-stream positive rates (empirical vs natural):")
        for i, lbl in enumerate(CHEXPERT_14_LABELS):
            log.info("  %-35s empirical=%.3f  natural=%.3f", lbl, empirical_rates[i], natural_rates[i])

        eps = 1e-6
        if pos_weight_mode == "sampled_stream_sqrt":
            pw = np.sqrt((1 - empirical_rates + eps) / (empirical_rates + eps))
        else:
            pw = (1 - empirical_rates + eps) / (empirical_rates + eps)

        pos_weight_tensor = torch.tensor(pw, dtype=torch.float32)
        log.info("pos_weight [%s]: %s", pos_weight_mode, [f"{w:.2f}" for w in pos_weight_tensor.tolist()])

    else:  # none
        log.info("pos_weight: none")

    pos_weight_device = pos_weight_tensor.to(device) if pos_weight_tensor is not None else None

    # ── Build model ───────────────────────────────────────────────────────────
    model = CheXpertB1V2Model(model_cfg)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: CheXpertB1V2Model | Total params: %.1fM", total_params / 1e6)

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema = train_cfg.get("use_ema", False)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    ema: Optional[ModelEMA] = ModelEMA(model, decay=ema_decay) if use_ema else None

    # ── AMP scaler ────────────────────────────────────────────────────────────
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run = None
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(log_cfg.get("mlflow_experiment", "chexpert_b1_v2"))
            mlflow_run = mlflow.start_run(run_name=experiment_name)
            # Log all config params (flatten nested dict)
            def _flat_params(d, prefix=""):
                out = {}
                for k, v in d.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        out.update(_flat_params(v, full_key))
                    else:
                        out[full_key] = str(v)
                return out
            mlflow.log_params(_flat_params(cfg))
            mlflow.log_param("smoke_test", args.smoke_test)
            log.info("MLflow run started: %s", mlflow_run.info.run_id)
        except Exception as e:
            log.warning("MLflow setup failed: %s", e)
            mlflow_run = None

    # ── Helper: get primary metric ────────────────────────────────────────────
    early_stop_metric = train_cfg.get("early_stop_metric", "mean_auprc")

    def _primary_metric(metrics: Dict) -> float:
        if early_stop_metric == "mean_auroc":
            return metrics.get("mean_auroc", 0.0)
        elif early_stop_metric == "loss":
            return -metrics.get("val_loss", float("inf"))
        else:
            return metrics.get("mean_auprc", 0.0)

    def _metric_improved(current: float, best: float) -> bool:
        return current > best

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE A: Frozen backbone, train head only
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STAGE A: Head warmup (%d epochs, lr=%.0e, backbone frozen)", stage_a_epochs, train_cfg.get("stage_a_lr", 1e-3))
    log.info("=" * 70)

    model.freeze_backbone()

    optimizer_a = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("stage_a_lr", 1e-3),
        weight_decay=train_cfg.get("stage_a_weight_decay", 1e-2),
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, T_max=stage_a_epochs, eta_min=1e-6
    )

    best_a_metric = -float("inf")
    best_a_path = checkpoint_dir / f"stageA_{experiment_name}.pth"
    patience_a = train_cfg.get("stage_a_patience", 3)
    no_improve_a = 0

    for epoch in range(1, stage_a_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer_a, scaler,
            device, pos_weight_device, cfg, epoch, stage="A", ema=ema,
        )
        scheduler_a.step()

        # Validate using EMA weights if enabled
        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.apply_to(eval_model)
        else:
            eval_model = model
        metrics = evaluate(eval_model, val_loader, device, use_amp=use_amp)
        metric_val = _primary_metric(metrics)

        log.info(
            "StageA Ep%02d | train_loss=%.4f | mean_auroc=%.4f (valid=%d/14) | mean_auprc=%.4f | F1=%.4f | %.0fs",
            epoch, train_loss, metrics["mean_auroc"], metrics["n_valid_classes"],
            metrics["mean_auprc"], metrics["macro_f1"], time.time() - t0,
        )
        for lbl in CHEXPERT_14_LABELS:
            auroc = metrics["per_class_auroc"].get(lbl, float("nan"))
            auprc = metrics["per_class_auprc"].get(lbl, float("nan"))
            tag = " [no pos]" if math.isnan(auroc) else ""
            log.info("  %-35s AUROC=%.4f  AUPRC=%.4f%s", lbl, auroc if not math.isnan(auroc) else -1.0, auprc if not math.isnan(auprc) else -1.0, tag)

        if _MLFLOW_AVAILABLE and mlflow_run:
            try:
                mlflow.log_metric("stageA_train_loss", train_loss, step=epoch)
                mlflow.log_metric("stageA_mean_auroc", metrics["mean_auroc"], step=epoch)
                mlflow.log_metric("stageA_mean_auprc", metrics["mean_auprc"], step=epoch)
                mlflow.log_metric("stageA_mean_auroc_valid", metrics["mean_auroc_valid"], step=epoch)
                mlflow.log_metric("stageA_mean_auprc_valid", metrics["mean_auprc_valid"], step=epoch)
                mlflow.log_metric("stageA_n_valid_classes", metrics["n_valid_classes"], step=epoch)
                for lbl, v in metrics["per_class_auroc"].items():
                    mlflow.log_metric(f"stageA_auroc_{lbl.replace(' ', '_')}", v if not math.isnan(v) else -1.0, step=epoch)
                for lbl, v in metrics["per_class_auprc"].items():
                    mlflow.log_metric(f"stageA_auprc_{lbl.replace(' ', '_')}", v if not math.isnan(v) else -1.0, step=epoch)
            except Exception:
                pass

        if _metric_improved(metric_val, best_a_metric):
            best_a_metric = metric_val
            no_improve_a = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_a.state_dict(),
                "mean_auprc": metrics["mean_auprc"],
                "mean_auroc": metrics["mean_auroc"],
                "per_class_auroc": metrics["per_class_auroc"],
                "per_class_auprc": metrics["per_class_auprc"],
                "cfg": cfg,
                "arch_version": "b1v2",
                "labels": CHEXPERT_14_LABELS,
                "pos_weight": pos_weight_tensor.tolist() if pos_weight_tensor is not None else None,
                "thresholds": {},
            }
            if ema is not None:
                ckpt["ema_state_dict"] = {k: v.cpu() for k, v in ema.state_dict().items()}
            torch.save(ckpt, best_a_path)
            log.info("  → Best Stage A checkpoint saved (metric=%.4f): %s", metric_val, best_a_path)
        else:
            no_improve_a += 1
            if no_improve_a >= patience_a:
                log.info("Stage A early stop at epoch %d (patience=%d)", epoch, patience_a)
                break

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE B: Partial unfreeze + cosine schedule with linear warmup
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info(
        "STAGE B: Backbone fine-tune (%d epochs, backbone_lr=%.0e, head_lr=%.0e, mode=%s)",
        stage_b_epochs,
        train_cfg.get("stage_b_backbone_lr", 1e-5),
        train_cfg.get("stage_b_head_lr", 1e-4),
        train_cfg.get("stage_b_unfreeze_mode", "last3"),
    )
    log.info("=" * 70)

    # Load best Stage A checkpoint
    log.info("Loading best Stage A checkpoint: %s", best_a_path)
    ckpt_a = torch.load(best_a_path, map_location=device)
    model.load_state_dict(ckpt_a["model_state_dict"])

    param_groups = model.get_param_groups(
        backbone_lr=train_cfg.get("stage_b_backbone_lr", 1e-5),
        head_lr=train_cfg.get("stage_b_head_lr", 1e-4),
        unfreeze_mode=train_cfg.get("stage_b_unfreeze_mode", "last3"),
    )

    optimizer_b = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("stage_b_weight_decay", 1e-2),
    )

    # Base LRs for warmup scaling
    base_lrs = [pg["lr"] for pg in optimizer_b.param_groups]
    warmup_epochs = train_cfg.get("warmup_epochs", 1)

    def _apply_lr_scale(optimizer, base_lrs, scale):
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * scale

    best_b_metric = -float("inf")
    best_b_path = checkpoint_dir / f"stageB_{experiment_name}.pth"
    patience_b = train_cfg.get("stage_b_patience", 5)
    no_improve_b = 0

    # Reinitialize EMA from current model weights for Stage B
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)

    for epoch in range(1, stage_b_epochs + 1):
        t0 = time.time()

        # LR schedule: warmup then cosine
        if epoch <= warmup_epochs:
            lr_scale = get_lr_scale_warmup(epoch - 1, warmup_epochs)
        else:
            lr_scale = cosine_lr_scale(epoch - warmup_epochs, stage_b_epochs - warmup_epochs, eta_min_ratio=1e-7)
        _apply_lr_scale(optimizer_b, base_lrs, lr_scale)

        train_loss = train_one_epoch(
            model, train_loader, optimizer_b, scaler,
            device, pos_weight_device, cfg, epoch, stage="B", ema=ema,
        )

        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.apply_to(eval_model)
        else:
            eval_model = model
        metrics = evaluate(eval_model, val_loader, device, use_amp=use_amp)
        metric_val = _primary_metric(metrics)

        log.info(
            "StageB Ep%02d | train_loss=%.4f | mean_auroc=%.4f (valid=%d/14) | mean_auprc=%.4f | F1=%.4f | lr_scale=%.4f | %.0fs",
            epoch, train_loss, metrics["mean_auroc"], metrics["n_valid_classes"],
            metrics["mean_auprc"], metrics["macro_f1"], lr_scale, time.time() - t0,
        )
        for lbl in CHEXPERT_14_LABELS:
            auroc = metrics["per_class_auroc"].get(lbl, float("nan"))
            auprc = metrics["per_class_auprc"].get(lbl, float("nan"))
            tag = " [no pos]" if math.isnan(auroc) else ""
            log.info("  %-35s AUROC=%.4f  AUPRC=%.4f%s", lbl, auroc if not math.isnan(auroc) else -1.0, auprc if not math.isnan(auprc) else -1.0, tag)

        if _MLFLOW_AVAILABLE and mlflow_run:
            try:
                mlflow.log_metric("stageB_train_loss", train_loss, step=epoch)
                mlflow.log_metric("stageB_mean_auroc", metrics["mean_auroc"], step=epoch)
                mlflow.log_metric("stageB_mean_auprc", metrics["mean_auprc"], step=epoch)
                mlflow.log_metric("stageB_mean_auroc_valid", metrics["mean_auroc_valid"], step=epoch)
                mlflow.log_metric("stageB_mean_auprc_valid", metrics["mean_auprc_valid"], step=epoch)
                mlflow.log_metric("stageB_n_valid_classes", metrics["n_valid_classes"], step=epoch)
                mlflow.log_metric("stageB_lr_scale", lr_scale, step=epoch)
                for lbl, v in metrics["per_class_auroc"].items():
                    mlflow.log_metric(f"stageB_auroc_{lbl.replace(' ', '_')}", v if not math.isnan(v) else -1.0, step=epoch)
                for lbl, v in metrics["per_class_auprc"].items():
                    mlflow.log_metric(f"stageB_auprc_{lbl.replace(' ', '_')}", v if not math.isnan(v) else -1.0, step=epoch)
            except Exception:
                pass

        if _metric_improved(metric_val, best_b_metric):
            best_b_metric = metric_val
            no_improve_b = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_b.state_dict(),
                "mean_auprc": metrics["mean_auprc"],
                "mean_auroc": metrics["mean_auroc"],
                "per_class_auroc": metrics["per_class_auroc"],
                "per_class_auprc": metrics["per_class_auprc"],
                "cfg": cfg,
                "arch_version": "b1v2",
                "labels": CHEXPERT_14_LABELS,
                "pos_weight": pos_weight_tensor.tolist() if pos_weight_tensor is not None else None,
                "thresholds": {},
            }
            if ema is not None:
                ckpt["ema_state_dict"] = {k: v.cpu() for k, v in ema.state_dict().items()}
            torch.save(ckpt, best_b_path)
            log.info("  → Best Stage B checkpoint saved (metric=%.4f): %s", metric_val, best_b_path)
        else:
            no_improve_b += 1
            if no_improve_b >= patience_b:
                log.info("Stage B early stop at epoch %d (patience=%d)", epoch, patience_b)
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training complete!")
    log.info("Best Stage A %s = %.4f | Checkpoint: %s", early_stop_metric, best_a_metric, best_a_path)
    log.info("Best Stage B %s = %.4f | Checkpoint: %s", early_stop_metric, best_b_metric, best_b_path)
    log.info("=" * 70)

    if _MLFLOW_AVAILABLE and mlflow_run:
        try:
            mlflow.log_metric("best_stageA_metric", best_a_metric)
            mlflow.log_metric("best_stageB_metric", best_b_metric)
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
