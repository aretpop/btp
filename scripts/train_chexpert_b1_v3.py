"""
scripts/train_chexpert_b1_v3.py
================================
Two-stage training script for CheXpertV3Model.

Supports:
  - recipe: v2_legacy | uslfa_v3
  - loss.type: bce_wpos | uslt (double-correction fix: no pos_weight with uslt)
  - sampler: random | tiered  (warning logged when tiered + uslt)
  - LabelAdaptiveHead or linear14 head
  - focus_normalize, cooc_masked diversity loss
  - Per-class alpha weight printing at startup

Usage:
    python scripts/train_chexpert_b1_v3.py --config configs/chexpert_b1_v3.yaml
    python scripts/train_chexpert_b1_v3.py --config configs/chexpert_b1_v3.yaml --smoke_test
    python scripts/train_chexpert_b1_v3.py --config configs/ablations/ablation_b_no_double_correction.yaml
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

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v3 import CheXpertV3Model
from src.models.uslt_loss import (
    USLTLoss,
    compute_alpha_weights,
    apply_label_policy,
    compute_label_cooccurrence,
)

# ── MLflow (optional) ─────────────────────────────────────────────────────────
try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False
    warnings.warn("mlflow not installed — experiment tracking disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────
CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
NUM_CLASSES = len(CHEXPERT_14_LABELS)

POOL1_DISEASES = ["Pneumonia", "Pleural Other", "Fracture",
                  "Lung Lesion", "Consolidation", "Enlarged Cardiomediastinum"]
POOL2_DISEASES = ["Cardiomegaly", "Edema", "Pleural Effusion",
                  "Pneumothorax", "Atelectasis", "Lung Opacity"]

# ── Logging ───────────────────────────────────────────────────────────────────
_FMT = "%(asctime)s  %(levelname)-7s  %(message)s"
_DFMT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_FMT, datefmt=_DFMT)
log = logging.getLogger("train_b1v3")


def _add_file_handler(log_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DFMT))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DFMT))
    logging.basicConfig(level=logging.INFO, format=_FMT, datefmt=_DFMT, handlers=[stream_handler, file_handler], force=True)
    log.info("Logging tee to file: %s", log_path)


# =============================================================================
# Config helpers
# =============================================================================

def _deep_set(d: dict, key_path: str, value):
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
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


def load_config(path: str, overrides: List[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        k, v = ov.split("=", 1)
        _deep_set(cfg, k.strip(), v.strip())
        log.info("Override: %s = %s", k.strip(), v.strip())
    return cfg


def cfg_get(cfg: dict, *keys, default=None):
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# =============================================================================
# Data
# =============================================================================

import re as _re

def _patient_id(local_filename: str) -> str:
    m = _re.search(r"(patient\d+)", str(local_filename))
    return m.group(1) if m else str(local_filename)


def make_patient_splits(
    df: pd.DataFrame, val_frac: float = 0.10, test_frac: float = 0.10, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if "local_filename" in df.columns:
        pid_series = df["local_filename"].apply(_patient_id)
    elif "patient_id" in df.columns:
        pid_series = df["patient_id"]
    else:
        pid_series = df["file_id"].apply(lambda x: x.split(".")[0])

    unique_pids = pid_series.unique()
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    test_pids  = set(unique_pids[:n_test])
    val_pids   = set(unique_pids[n_test:n_test + n_val])
    train_pids = set(unique_pids[n_test + n_val:])

    train_df = df[pid_series.isin(train_pids)].reset_index(drop=True)
    val_df   = df[pid_series.isin(val_pids)].reset_index(drop=True)
    test_df  = df[pid_series.isin(test_pids)].reset_index(drop=True)

    log.info(
        "Patient splits — total=%d, train=%d, val=%d, test=%d | "
        "rows: train=%d, val=%d, test=%d",
        n, len(train_pids), len(val_pids), len(test_pids),
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _get_transforms(res: int, split: str):
    from torchvision import transforms
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CheXpertDatasetV3(Dataset):
    """
    Multi-label CheXpert dataset.
    Returns (img, raw_labels [14]) where raw_labels ∈ {1.0, 0.0, -1.0, NaN}.
    Label policy is applied in collate_fn / loss to allow per-step flexibility.
    """
    CACHE_DIR = PROJECT_ROOT / "data" / "chexpert_cache"

    def __init__(self, df: pd.DataFrame, split: str = "train", cfg: dict = None):
        self.df = df.reset_index(drop=True)
        self.split = split
        cfg = cfg or {}
        res = cfg.get("model_input_resolution", 224)
        self.cache_dir = Path(cfg.get("cache_dir", str(self.CACHE_DIR)))
        self.transform = _get_transforms(res, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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

        # Return raw labels as floats (NaN preserved via float("nan"))
        raw = []
        for lbl in CHEXPERT_14_LABELS:
            v = row.get(lbl, float("nan"))
            raw.append(float(v) if not pd.isna(v) else float("nan"))

        return img, torch.tensor(raw, dtype=torch.float32)


def build_label_tensors(
    raw_batch: torch.Tensor,
    policy: str = "u_soft",
    uncertain_target: float = 0.5,
    per_class_policy: Optional[Dict[int, str]] = None,
    nan_policy: str = "ignore",
    nan_target: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert raw labels batch → (targets, uncertain, mask).
    Calls apply_label_policy from uslt_loss.
    """
    return apply_label_policy(
        raw_batch, policy=policy,
        uncertain_target=uncertain_target,
        per_class_policy=per_class_policy,
        nan_policy=nan_policy,
        nan_target=nan_target,
    )


# =============================================================================
# Sampler (tiered — same as v2 for legacy compat)
# =============================================================================

class TieredSamplerV3(Sampler):
    def __init__(
        self,
        df: pd.DataFrame,
        pool_counts: List[int] = None,
        batches_per_epoch: int = 800,
        intra_pool_mode: str = "sqrt_freq",
        seed: int = 42,
    ):
        self.batches_per_epoch = batches_per_epoch
        self.intra_pool_mode = intra_pool_mode
        self.rng = np.random.default_rng(seed)
        pool_counts = pool_counts or [14, 12, 6]
        self.pool_counts = pool_counts

        p1_diseases = [l for l in POOL1_DISEASES if l in df.columns]
        p2_diseases = [l for l in POOL2_DISEASES if l in df.columns]
        self.p1_buckets = [df[df[d] == 1.0].index.values.copy() for d in p1_diseases]
        self.p2_buckets = [df[df[d] == 1.0].index.values.copy() for d in p2_diseases]
        nf = df["No Finding"] == 1.0 if "No Finding" in df.columns else pd.Series(False, index=df.index)
        self.p3_indices = df[nf].index.values.copy()
        if len(self.p3_indices) == 0:
            self.p3_indices = df.index.values.copy()

        for b in self.p1_buckets + self.p2_buckets:
            self.rng.shuffle(b)
        self.rng.shuffle(self.p3_indices)

        if intra_pool_mode == "sqrt_freq":
            self.p1_w = self._mk_weights(self.p1_buckets)
            self.p2_w = self._mk_weights(self.p2_buckets)
        else:
            self.p1_w = self.p2_w = None

        self.p1_ptrs = [0] * len(self.p1_buckets)
        self.p2_ptrs = [0] * len(self.p2_buckets)
        self.p3_ptr = 0

    def _mk_weights(self, buckets):
        w = np.array([math.sqrt(max(len(b), 1)) for b in buckets], dtype=float)
        return w / w.sum() if w.sum() > 0 else w

    def _sample_pool(self, buckets, ptrs, count, weights=None):
        samples = []
        if not buckets:
            return list(np.zeros(count, dtype=int))
        for _ in range(count):
            b_idx = int(self.rng.choice(len(buckets), p=weights)) if weights is not None else int(self.rng.integers(0, len(buckets)))
            ptr = ptrs[b_idx]
            if ptr >= len(buckets[b_idx]):
                self.rng.shuffle(buckets[b_idx]); ptrs[b_idx] = 0; ptr = 0
            samples.append(int(buckets[b_idx][ptr])); ptrs[b_idx] += 1
        return samples

    def _sample_p3(self, count):
        samples = []
        for _ in range(count):
            if self.p3_ptr >= len(self.p3_indices):
                self.rng.shuffle(self.p3_indices); self.p3_ptr = 0
            samples.append(int(self.p3_indices[self.p3_ptr])); self.p3_ptr += 1
        return samples

    def __iter__(self):
        n1, n2, n3 = self.pool_counts
        for _ in range(self.batches_per_epoch):
            b = (self._sample_pool(self.p1_buckets, self.p1_ptrs, n1, self.p1_w) +
                 self._sample_pool(self.p2_buckets, self.p2_ptrs, n2, self.p2_w) +
                 self._sample_p3(n3))
            self.rng.shuffle(b); yield b

    def __len__(self):
        return self.batches_per_epoch


# =============================================================================
# Loss
# =============================================================================

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight.to(logits.device), reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum()
    return loss.sum() / denom if denom >= 1.0 else loss.sum() * 0.0


def build_pos_weight(
    df: pd.DataFrame,
    labels: List[str],
    mode: str = "sampled_stream",
    cohort_sqrt_weights: Optional[List[float]] = None,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    if mode == "cohort_sqrt" and cohort_sqrt_weights:
        return torch.tensor(cohort_sqrt_weights, dtype=torch.float32)
    # sampled_stream: compute from train df
    pw = []
    n_total = len(df)
    for lbl in labels:
        n_pos = (df[lbl] == 1.0).sum() if lbl in df.columns else 0
        n_neg = n_total - n_pos
        ratio = math.sqrt(max(n_neg, 1) / max(n_pos, 1))
        pw.append(ratio)
    return torch.tensor(pw, dtype=torch.float32)


# =============================================================================
# EMA
# =============================================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v.float() * (1.0 - self.decay)

    def apply_to(self, model: nn.Module):
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
    uncertain_policy: str = "u_soft",
    uncertain_target: float = 0.5,
    exclude_uncertain: bool = True,
) -> Dict:
    model.eval()
    all_logits, all_raw = [], []

    for imgs, raw_labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, _ = model(imgs)
        all_logits.append(logits.float().cpu())
        all_raw.append(raw_labels.cpu())

    logits_all = torch.cat(all_logits, dim=0).numpy()
    raw_all    = torch.cat(all_raw, dim=0).numpy()
    probs_all  = 1.0 / (1.0 + np.exp(-logits_all))

    per_class_auroc, per_class_auprc, per_class_f1 = {}, {}, {}
    valid_auroc, valid_auprc = [], []

    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        raw_col = raw_all[:, i]
        if exclude_uncertain:
            # Exclude uncertain (raw==-1) from metric computation
            obs_mask = ~np.isnan(raw_col) & (raw_col != -1.0)
        else:
            obs_mask = ~np.isnan(raw_col)

        if obs_mask.sum() == 0:
            per_class_auroc[lbl] = float("nan")
            per_class_auprc[lbl] = float("nan")
            per_class_f1[lbl] = 0.0
            continue

        t_obs = (raw_col[obs_mask] == 1.0).astype(float)
        p_obs = probs_all[obs_mask, i]

        if len(np.unique(t_obs)) < 2:
            log.warning("Val: '%s' has insufficient classes (all 0 or all 1) — skipping.", lbl)
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
        per_class_f1[lbl] = f1_score(t_obs.astype(int), preds_bin, zero_division=0)

    all_auroc = [v if not math.isnan(v) else 0.0 for v in per_class_auroc.values()]
    all_auprc = [v if not math.isnan(v) else 0.0 for v in per_class_auprc.values()]

    return {
        "per_class_auroc":    per_class_auroc,
        "per_class_auprc":    per_class_auprc,
        "per_class_f1":       per_class_f1,
        "mean_auroc":         float(np.mean(all_auroc)),
        "mean_auprc":         float(np.mean(all_auprc)),
        "mean_auroc_valid":   float(np.mean(valid_auroc)) if valid_auroc else 0.0,
        "mean_auprc_valid":   float(np.mean(valid_auprc)) if valid_auprc else 0.0,
        "macro_f1":           float(np.mean(list(per_class_f1.values()))),
        "n_valid_classes":    len(valid_auprc),
    }


# =============================================================================
# Scheduler helpers
# =============================================================================

def warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(warmup_epochs)


def cosine_scale(epoch: int, total: int, eta_min: float = 1e-7) -> float:
    return eta_min + 0.5 * (1.0 - eta_min) * (1 + math.cos(math.pi * epoch / max(total, 1)))


# =============================================================================
# Training loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    stage: str = "A",
    # Loss setup
    loss_type: str = "uslt",
    uslt_criterion: Optional[USLTLoss] = None,
    pos_weight: Optional[torch.Tensor] = None,
    uncertain_policy: str = "u_soft",
    uncertain_target: float = 0.5,
    per_class_policy: Optional[Dict] = None,
    nan_policy: str = "ignore",
    nan_target: float = 0.5,
    ema: Optional[ModelEMA] = None,
) -> float:
    model.train()
    if stage == "A":
        model.backbone.eval()   # keep BN frozen in Stage A

    use_amp = cfg_get(cfg, "training", "use_amp", default=True) and device.type == "cuda"
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        imgs, raw_labels = batch
        imgs = imgs.to(device, non_blocking=True)
        raw_labels = raw_labels.to(device, non_blocking=True)

        # Build targets/mask/uncertain from raw labels
        targets, uncertain, mask = build_label_tensors(
            raw_labels, policy=uncertain_policy,
            uncertain_target=uncertain_target,
            per_class_policy=per_class_policy,
            nan_policy=nan_policy,
            nan_target=nan_target,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, attn_reg = model(imgs)

            if loss_type == "uslt" and uslt_criterion is not None:
                bce = uslt_criterion(logits, targets, mask, uncertain)
            else:
                # Legacy bce_wpos: pos_weight applied only here (not in uslt path)
                bce = masked_bce_loss(logits, targets, mask, pos_weight if loss_type == "bce_wpos" else None)

            loss = bce + attn_reg

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

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            log.info(
                "  Stage %s Ep%02d [%d/%d] loss=%.4f (bce=%.4f reg=%.4f)",
                stage, epoch + 1, batch_idx + 1, len(loader), loss.item(), bce.item(), attn_reg.item(),
            )

    return total_loss / max(n_batches, 1)


# =============================================================================
# Main training function
# =============================================================================

def train(cfg: dict, smoke_test: bool = False):
    recipe     = cfg.get("recipe", "uslfa_v3")
    loss_type  = cfg_get(cfg, "loss", "type", default="uslt")
    sampler_mode = cfg_get(cfg, "sampler", "mode", default="random")
    seed       = cfg_get(cfg, "training", "seed", default=42)

    # ── Double-correction warning ──────────────────────────────────────────────
    if loss_type == "uslt" and sampler_mode == "tiered":
        log.warning(
            "DOUBLE-CORRECTION WARNING: loss.type=uslt includes class-balanced "
            "weights (alpha_c) AND sampler.mode=tiered oversamples rare classes. "
            "This stacks two corrections. Consider sampler.mode=random with uslt."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | recipe: %s | loss: %s | sampler: %s", device, recipe, loss_type, sampler_mode)

    # ── Data ──────────────────────────────────────────────────────────────────
    index_csv = cfg_get(cfg, "data", "index_csv", default="data/chexpert_subset/chexpert_materialized_index.csv")
    cache_dir = cfg_get(cfg, "data", "cache_dir", default="data/chexpert_cache")
    val_frac  = cfg_get(cfg, "data", "val_frac",  default=0.10)
    test_frac = cfg_get(cfg, "data", "test_frac", default=0.10)
    split_seed = cfg_get(cfg, "data", "split_seed", default=42)
    uncertain_policy  = cfg_get(cfg, "data", "uncertain_policy",  default="u_soft")
    uncertain_target  = cfg_get(cfg, "data", "uncertain_target",  default=0.5)
    nan_policy        = cfg_get(cfg, "data", "nan_policy",        default="ignore")
    nan_target        = cfg_get(cfg, "data", "nan_target",        default=0.5)
    per_class_policy_raw = cfg_get(cfg, "data", "per_class_policy", default=None)
    per_class_policy = {int(k): v for k, v in per_class_policy_raw.items()} if per_class_policy_raw else None

    df = pd.read_csv(index_csv)
    if smoke_test:
        df = df.head(200)
        log.info("SMOKE TEST: using first 200 rows")

    train_df, val_df, test_df = make_patient_splits(df, val_frac, test_frac, split_seed)

    res = cfg_get(cfg, "input_resolution", default=224)
    ds_cfg = {"model_input_resolution": res, "cache_dir": cache_dir}

    train_ds = CheXpertDatasetV3(train_df, "train", ds_cfg)
    val_ds   = CheXpertDatasetV3(val_df,   "val",   ds_cfg)

    # ── USLT class weights ─────────────────────────────────────────────────────
    uslt_criterion = None
    if loss_type == "uslt":
        beta       = cfg_get(cfg, "loss", "beta",       default=0.999)
        alpha_mode = cfg_get(cfg, "loss", "alpha_mode", default="effective_num")
        lambda_u   = cfg_get(cfg, "loss", "lambda_u",   default=0.5)
        eps_u      = cfg_get(cfg, "loss", "eps",        default=0.1)

        alpha_np, alpha_info = compute_alpha_weights(
            train_df, CHEXPERT_14_LABELS, beta=beta, alpha_mode=alpha_mode,
        )
        log.info("alpha_c min=%.4f max=%.4f ratio=%.4f", alpha_info["min_alpha"], alpha_info["max_alpha"], alpha_info["min_max_ratio"])
        uslt_criterion = USLTLoss(
            alpha=torch.tensor(alpha_np, dtype=torch.float32),
            lambda_u=lambda_u,
            eps=eps_u,
        ).to(device)

    # ── Co-occurrence mask ─────────────────────────────────────────────────────
    model_cfg = {k: v for k, v in cfg.items() if k not in ("data", "sampler", "loss", "training", "eval", "logging", "experiment")}
    model = CheXpertV3Model(model_cfg).to(device)

    div_mode = cfg.get("div_mode", "cooc_masked")
    if div_mode == "cooc_masked":
        log.info("Computing label co-occurrence from training split...")
        jaccard = compute_label_cooccurrence(train_df, CHEXPERT_14_LABELS)
        model.set_cooc_mask(jaccard)

    # ── pos_weight (legacy bce_wpos only) ─────────────────────────────────────
    pos_weight = None
    if loss_type == "bce_wpos":
        pw_mode = cfg_get(cfg, "loss", "pos_weight_mode", default="sampled_stream")
        cohort_w = cfg_get(cfg, "loss", "cohort_sqrt_weights", default=None)
        pos_weight = build_pos_weight(train_df, CHEXPERT_14_LABELS, pw_mode, cohort_w)
        if pos_weight is not None:
            log.info("pos_weight: %s", [f"{w:.2f}" for w in pos_weight.tolist()])
    elif loss_type == "uslt":
        log.info("loss.type=uslt → pos_weight NOT applied (double-correction fix)")

    # ── Sampler + DataLoader ───────────────────────────────────────────────────
    batch_size = cfg_get(cfg, "sampler", "batch_size", default=32)
    n_workers  = cfg_get(cfg, "training", "num_workers", default=4)
    pin_mem    = cfg_get(cfg, "training", "pin_memory", default=True)
    bpe        = cfg_get(cfg, "sampler", "batches_per_epoch", default=800)

    if smoke_test:
        bpe = 2

    if sampler_mode == "tiered":
        tiered_pool = cfg_get(cfg, "sampler", "pool_counts", default=[14, 12, 6])
        ipm = cfg_get(cfg, "sampler", "intra_pool_mode", default="sqrt_freq")
        s_seed = cfg_get(cfg, "sampler", "seed", default=42)
        sampler = TieredSamplerV3(train_df, pool_counts=tiered_pool, batches_per_epoch=bpe, intra_pool_mode=ipm, seed=s_seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=n_workers, pin_memory=pin_mem)
        # Log realized ratios and warn about double correction
        realized = {f"pool_{i+1}": tiered_pool[i] / sum(tiered_pool) for i in range(len(tiered_pool))}
        log.info("Tiered sampler realized pool fracs: %s", realized)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=pin_mem, drop_last=True,
        )

    val_batch = cfg_get(cfg, "eval", "val_batch_size", default=64)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=n_workers, pin_memory=pin_mem)

    # ── Checkpoint & Log dirs ─────────────────────────────────────────────────
    ckpt_dir = Path(cfg_get(cfg, "logging", "checkpoint_dir", default="checkpoints/chexpert_b1_v3"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg_get(cfg, "logging", "results_dir", default="outputs/chexpert_b1_v3"))
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = cfg_get(cfg, "experiment", "name", default="chexpert_b1_v3")
    _add_file_handler(str(results_dir / f"{exp_name}.log"))

    use_amp  = cfg_get(cfg, "training", "use_amp",   default=True) and device.type == "cuda"
    use_ema  = cfg_get(cfg, "training", "use_ema",   default=False)
    ema_dec  = cfg_get(cfg, "training", "ema_decay", default=0.999)
    es_metric = cfg_get(cfg, "training", "early_stop_metric", default="mean_auprc")
    exclude_unc = cfg_get(cfg, "eval", "exclude_uncertain_from_eval", default=True)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    ema = ModelEMA(model, ema_dec) if use_ema else None

    # ── STAGE A ───────────────────────────────────────────────────────────────
    stage_a_epochs  = 2 if smoke_test else cfg_get(cfg, "training", "stage_a_epochs", default=5)
    stage_a_lr      = cfg_get(cfg, "training", "stage_a_lr", default=1e-3)
    stage_a_wd      = cfg_get(cfg, "training", "stage_a_weight_decay", default=1e-2)
    stage_a_patience = cfg_get(cfg, "training", "stage_a_patience", default=3)
    warmup_epochs   = 0 if smoke_test else cfg_get(cfg, "training", "warmup_epochs", default=1)

    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt_a = torch.optim.AdamW(head_params, lr=stage_a_lr, weight_decay=stage_a_wd)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=stage_a_epochs, eta_min=1e-6)

    best_metric = -1.0
    best_ckpt = None
    patience_ctr = 0

    log.info("=" * 60)
    log.info("STAGE A: %d epochs, lr=%.1e, wd=%.1e", stage_a_epochs, stage_a_lr, stage_a_wd)

    for epoch in range(stage_a_epochs):
        if smoke_test and epoch >= 2:
            break
        train_loss = train_one_epoch(
            model, train_loader, opt_a, scaler, device, cfg, epoch, stage="A",
            loss_type=loss_type, uslt_criterion=uslt_criterion, pos_weight=pos_weight,
            uncertain_policy=uncertain_policy, uncertain_target=uncertain_target,
            per_class_policy=per_class_policy,
            nan_policy=nan_policy, nan_target=nan_target, ema=ema,
        )
        sched_a.step()

        metrics = evaluate(
            model, val_loader, device, use_amp,
            uncertain_policy=uncertain_policy, uncertain_target=uncertain_target,
            exclude_uncertain=exclude_unc,
        )
        m_val = metrics.get(es_metric, 0.0)

        log.info(
            "StageA ep=%d  loss=%.4f  auroc=%.4f  auprc=%.4f  (%s=%.4f)",
            epoch + 1, train_loss, metrics["mean_auroc"], metrics["mean_auprc"], es_metric, m_val,
        )

        if m_val > best_metric:
            best_metric = m_val
            patience_ctr = 0
            best_ckpt = ckpt_dir / f"{exp_name}_v3_stageA_best.pth"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metric": m_val, "cfg": cfg}, best_ckpt)
            log.info("  ✓ Saved StageA best → %s (%.4f)", best_ckpt.name, m_val)
        else:
            patience_ctr += 1
            if patience_ctr >= stage_a_patience:
                log.info("  Early stop at Stage A epoch %d", epoch + 1)
                break

    # ── STAGE B ───────────────────────────────────────────────────────────────
    if smoke_test:
        log.info("Smoke test: skipping Stage B")
    else:
        stage_b_epochs   = cfg_get(cfg, "training", "stage_b_epochs", default=20)
        stage_b_bblr     = cfg_get(cfg, "training", "stage_b_backbone_lr", default=1e-5)
        stage_b_hlr      = cfg_get(cfg, "training", "stage_b_head_lr", default=1e-4)
        stage_b_wd       = cfg_get(cfg, "training", "stage_b_weight_decay", default=1e-2)
        stage_b_patience = cfg_get(cfg, "training", "stage_b_patience", default=5)
        unfreeze_mode    = cfg_get(cfg, "training", "stage_b_unfreeze_mode", default="last4")

        param_groups = model.unfreeze_for_stage_b(mode=unfreeze_mode, backbone_lr=stage_b_bblr, head_lr=stage_b_hlr)
        opt_b = torch.optim.AdamW(param_groups, weight_decay=stage_b_wd)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=stage_b_epochs, eta_min=1e-7)

        patience_ctr = 0
        log.info("=" * 60)
        log.info("STAGE B: %d epochs, bblr=%.1e, hlr=%.1e", stage_b_epochs, stage_b_bblr, stage_b_hlr)

        for epoch in range(stage_b_epochs):
            train_loss = train_one_epoch(
                model, train_loader, opt_b, scaler, device, cfg, epoch, stage="B",
                loss_type=loss_type, uslt_criterion=uslt_criterion, pos_weight=pos_weight,
                uncertain_policy=uncertain_policy, uncertain_target=uncertain_target,
                per_class_policy=per_class_policy,
                nan_policy=nan_policy, nan_target=nan_target, ema=ema,
            )
            sched_b.step()

            eval_model = model
            if ema is not None:
                ema_snap = CheXpertV3Model(model_cfg).to(device)
                ema.apply_to(ema_snap)
                eval_model = ema_snap

            metrics = evaluate(
                eval_model, val_loader, device, use_amp,
                uncertain_policy=uncertain_policy, uncertain_target=uncertain_target,
                exclude_uncertain=exclude_unc,
            )
            m_val = metrics.get(es_metric, 0.0)
            log.info(
                "StageB ep=%d  loss=%.4f  auroc=%.4f  auprc=%.4f  (%s=%.4f)",
                epoch + 1, train_loss, metrics["mean_auroc"], metrics["mean_auprc"], es_metric, m_val,
            )
            # Per-class summary
            for lbl in CHEXPERT_14_LABELS:
                auprc = metrics["per_class_auprc"].get(lbl, float("nan"))
                auroc = metrics["per_class_auroc"].get(lbl, float("nan"))
                log.info("    %-35s  AUROC=%.4f  AUPRC=%.4f", lbl, auroc if not math.isnan(auroc) else -1, auprc if not math.isnan(auprc) else -1)

            if m_val > best_metric:
                best_metric = m_val
                patience_ctr = 0
                best_ckpt = ckpt_dir / f"{exp_name}_v3_stageB_best.pth"
                torch.save({"model_state_dict": eval_model.state_dict(), "epoch": epoch, "metric": m_val, "cfg": cfg}, best_ckpt)
                log.info("  ✓ Saved StageB best → %s (%.4f)", best_ckpt.name, m_val)
            else:
                patience_ctr += 1
                if patience_ctr >= stage_b_patience:
                    log.info("  Early stop at Stage B epoch %d", epoch + 1)
                    break

    log.info("Training complete. Best %s = %.4f | ckpt: %s", es_metric, best_metric, best_ckpt)
    return best_metric, best_ckpt


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chexpert_b1_v3.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                        help="key=value overrides, e.g. loss.lambda_u=0.75")
    parser.add_argument("--smoke_test", action="store_true",
                        help="2-step smoke test on 200 samples")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

