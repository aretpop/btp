"""
src/datasets/chexpert_local_v2.py

CheXpertLocalDatasetV2  — reads pre-cached local JPEGs.
TieredSamplerV2         — config-driven 3-pool tiered sampler.
make_patient_splits     — patient-level GroupShuffleSplit.
masked_bce_loss         — BCE with per-element mask.
compute_sampled_stream_pos_rates — empirical positive rate audit.
"""

from __future__ import annotations

import logging
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEXPERT_14_LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

SQRT_POS_WEIGHT: List[float] = [
    2.24, 3.63, 2.06, 1.02, 4.22, 1.80, 3.58,
    8.26, 2.28, 2.72, 1.28, 6.51, 4.24, 0.95,
]

# Default normalization stats (ImageNet)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_patient_id(local_filename: str) -> str:
    """Extract 'patientXXXXX' from local_filename."""
    m = re.search(r'(patient\d+)', str(local_filename))
    if m:
        return m.group(1)
    return "unknown"


def _build_transforms(input_resolution: int, split: str) -> T.Compose:
    """Build torchvision transform pipeline matching Phase 3."""
    if split == "train":
        return T.Compose([
            T.Resize((input_resolution, input_resolution)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((input_resolution, input_resolution)),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])


# ---------------------------------------------------------------------------
# Patient-level splits
# ---------------------------------------------------------------------------

def make_patient_splits(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level train/val/test split (no patient leakage).

    Uses manual grouping to avoid extra sklearn dependency issues;
    compatible with sklearn GroupShuffleSplit logic.

    Args:
        df: full dataframe (must have 'local_filename' column)
        val_frac:  fraction of patients for validation
        test_frac: fraction of patients for test
        seed:      random seed

    Returns:
        (train_df, val_df, test_df)
    """
    df = df.copy()
    df["_patient_id"] = df["local_filename"].apply(_extract_patient_id)

    all_patients = np.array(sorted(df["_patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_patients)

    n_total = len(all_patients)
    n_val  = max(1, int(round(n_total * val_frac)))
    n_test = max(1, int(round(n_total * test_frac)))
    n_train = n_total - n_val - n_test

    train_patients = set(all_patients[:n_train])
    val_patients   = set(all_patients[n_train:n_train + n_val])
    test_patients  = set(all_patients[n_train + n_val:])

    # Safety assertions
    assert len(train_patients & val_patients)  == 0, "PATIENT LEAKAGE: train ∩ val"
    assert len(train_patients & test_patients) == 0, "PATIENT LEAKAGE: train ∩ test"
    assert len(val_patients   & test_patients) == 0, "PATIENT LEAKAGE: val ∩ test"

    train_df = df[df["_patient_id"].isin(train_patients)].drop(columns=["_patient_id"]).reset_index(drop=True)
    val_df   = df[df["_patient_id"].isin(val_patients)].drop(columns=["_patient_id"]).reset_index(drop=True)
    test_df  = df[df["_patient_id"].isin(test_patients)].drop(columns=["_patient_id"]).reset_index(drop=True)

    logger.info(
        "Patient splits — patients: train=%d, val=%d, test=%d | "
        "rows: train=%d, val=%d, test=%d",
        len(train_patients), len(val_patients), len(test_patients),
        len(train_df), len(val_df), len(test_df),
    )
    print(
        f"[make_patient_splits] patients: train={len(train_patients)}, "
        f"val={len(val_patients)}, test={len(test_patients)} | "
        f"rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CheXpertLocalDatasetV2(Dataset):
    """
    Multi-label chest X-ray dataset reading pre-cached local JPEGs.

    Args:
        df:               DataFrame (from make_patient_splits or full CSV)
        image_cache_dir:  directory containing *.jpg files
        split:            "train" | "val" | "test"
        input_resolution: resize target (224 or 320)
        uncertain_policy: how to handle -1 labels:
                          "u_zeros" (default): -1 → 0
                          "u_ones":             -1 → 1
                          "ignore":             -1 → -1.0, mask=0
                          "smooth":             -1 → U[0.55, 0.85]
        labels:           list of label column names (default CHEXPERT_14_LABELS)

    Returns (per __getitem__):
        image_tensor: [3, H, W] float
        label_tensor: [num_classes] float
        mask_tensor:  [num_classes] float  (0 where uncertain+ignore, else 1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_cache_dir: str,
        split: str = "train",
        input_resolution: int = 224,
        uncertain_policy: str = "u_zeros",
        labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_cache_dir = Path(image_cache_dir)
        self.split = split
        self.input_resolution = input_resolution
        self.uncertain_policy = uncertain_policy
        self.labels = labels if labels is not None else CHEXPERT_14_LABELS
        self.num_classes = len(self.labels)
        self.transform = _build_transforms(input_resolution, split)

        # Pre-cache label arrays for speed
        self._label_matrix = self._preprocess_labels()

    def _preprocess_labels(self) -> np.ndarray:
        """
        Process label columns according to uncertain_policy.
        Returns float32 array of shape [N, C].
        NaN → 0.0 always.
        """
        # Grab label columns, fill NaN with 0
        raw = self.df[self.labels].fillna(0.0).values.astype(np.float32)

        if self.uncertain_policy == "u_zeros":
            raw = np.where(raw == -1.0, 0.0, raw)
        elif self.uncertain_policy == "u_ones":
            raw = np.where(raw == -1.0, 1.0, raw)
        elif self.uncertain_policy == "ignore":
            pass  # keep -1; mask will be computed per-sample
        elif self.uncertain_policy == "smooth":
            # Random uniform in [0.55, 0.85] for each uncertain entry
            uncertain_mask = raw == -1.0
            smooth_vals = np.random.uniform(0.55, 0.85, raw.shape).astype(np.float32)
            raw = np.where(uncertain_mask, smooth_vals, raw)
        else:
            raise ValueError(f"Unknown uncertain_policy: {self.uncertain_policy}")

        return raw

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # --- Image ---
        fname = Path(str(row["local_filename"])).name
        img_path = self.image_cache_dir / fname
        if not img_path.exists():
            # Try exact path as given
            img_path = Path(str(row["local_filename"]))

        img = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(img)

        # --- Labels ---
        raw_labels = self._label_matrix[idx].copy()

        if self.uncertain_policy == "ignore":
            # Mask: 0 where original label was -1 (uncertain)
            orig = self.df[self.labels].fillna(0.0).iloc[idx].values.astype(np.float32)
            mask = (orig != -1.0).astype(np.float32)
            # For the label itself, set to 0 (won't matter since masked out)
            raw_labels = np.where(raw_labels == -1.0, 0.0, raw_labels)
        else:
            mask = np.ones(self.num_classes, dtype=np.float32)

        label_tensor = torch.from_numpy(raw_labels)
        mask_tensor  = torch.from_numpy(mask)

        return image_tensor, label_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Tiered Sampler V2
# ---------------------------------------------------------------------------

class TieredSamplerV2(Sampler):
    """
    Config-driven 3-pool tiered sampler for imbalanced multi-label datasets.

    Pools:
        Pool 1 (Rare/Focal):   rarest positive label prevalence < threshold_1
        Pool 2 (Common):       rarest prevalence in [threshold_1, threshold_2)
        Pool 3 (No Finding / catch-all): No Finding samples or no positives

    Args:
        df:                DataFrame with label columns
        pool_counts:       [n_pool1, n_pool2, n_pool3] images per batch
        labels:            label column names (14 CheXpert labels)
        batches_per_epoch: number of batches that constitute one epoch
        batch_size:        total batch size (should equal sum(pool_counts))
        intra_pool_mode:   "sqrt_freq" | "uniform" | "random"
        seed:              random seed
        sampler_mode:      "tiered" (default) | "random" (plain baseline)
        threshold_1:       pool 1/2 boundary prevalence (default 0.05)
        threshold_2:       pool 2/3 boundary prevalence (default 0.25)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pool_counts: List[int],
        labels: List[str],
        batches_per_epoch: int = 800,
        batch_size: int = 32,
        intra_pool_mode: str = "sqrt_freq",
        seed: int = 42,
        sampler_mode: str = "tiered",
        threshold_1: float = 0.05,
        threshold_2: float = 0.25,
    ):
        self.df = df.reset_index(drop=True)
        self.pool_counts = pool_counts
        self.labels = labels
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.intra_pool_mode = intra_pool_mode
        self.seed = seed
        self.sampler_mode = sampler_mode
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Build pools
        if sampler_mode == "tiered":
            self._prevalence = self._compute_prevalence()
            self._pools, self._pool_weights = self._assign_pools()
            self._log_pool_stats()
        else:
            # Plain random baseline
            self._all_indices = list(range(len(self.df)))

    def _compute_prevalence(self) -> Dict[str, float]:
        """Natural prevalence for each label (fraction of rows with label == 1.0)."""
        n = len(self.df)
        return {
            label: float((self.df[label] == 1.0).sum()) / n
            for label in self.labels
        }

    def _assign_pools(self) -> Tuple[List[List[int]], List[np.ndarray]]:
        """
        Assign each row to Pool 1, 2, or 3 based on rarest positive label prevalence.
        Returns (pools, pool_weights) where pool_weights[k] is per-sample weight array.
        """
        pools: List[List[int]] = [[], [], []]  # pool1, pool2, pool3

        no_finding_col = "No Finding" if "No Finding" in self.labels else None

        for idx, row in self.df.iterrows():
            # Determine if this is a "No Finding" sample
            if no_finding_col and row.get(no_finding_col, 0.0) == 1.0:
                pools[2].append(idx)
                continue

            # Find rarest positive label
            pos_labels = [
                lbl for lbl in self.labels
                if lbl != no_finding_col and row.get(lbl, 0.0) == 1.0
            ]

            if not pos_labels:
                # No positives → Pool 3
                pools[2].append(idx)
                continue

            # Rarest = minimum prevalence
            rarest_prev = min(self._prevalence[lbl] for lbl in pos_labels)

            if rarest_prev < self.threshold_1:
                pools[0].append(idx)  # Pool 1: Rare
            elif rarest_prev < self.threshold_2:
                pools[1].append(idx)  # Pool 2: Common
            else:
                pools[2].append(idx)  # Pool 3

        # Compute per-sample weights for intra-pool sampling
        pool_weights = []
        for pool_idx, pool in enumerate(pools):
            if not pool:
                pool_weights.append(np.array([]))
                continue

            if self.intra_pool_mode == "sqrt_freq" and pool_idx < 2:
                # Count positive occurrences per label within pool
                pool_df = self.df.iloc[pool]
                label_counts = {
                    lbl: float((pool_df[lbl] == 1.0).sum()) + 1.0  # +1 smoothing
                    for lbl in self.labels
                }
                # Weight each sample by 1/sqrt(count of its rarest positive)
                weights = []
                for idx in pool:
                    row = self.df.iloc[idx]
                    pos_labels = [lbl for lbl in self.labels if row.get(lbl, 0.0) == 1.0]
                    if pos_labels:
                        rarest_lbl = min(pos_labels, key=lambda l: label_counts[l])
                        w = 1.0 / math.sqrt(label_counts[rarest_lbl])
                    else:
                        w = 1.0
                    weights.append(w)
                pool_weights.append(np.array(weights, dtype=np.float64))
            elif self.intra_pool_mode == "uniform":
                pool_weights.append(np.ones(len(pool), dtype=np.float64))
            else:  # "random"
                pool_weights.append(np.ones(len(pool), dtype=np.float64))

        return pools, pool_weights

    def _log_pool_stats(self):
        """Log pool sizes to stdout and logger."""
        msg = (
            f"[TieredSamplerV2] Pool sizes — "
            f"Pool1(Rare)={len(self._pools[0])}, "
            f"Pool2(Common)={len(self._pools[1])}, "
            f"Pool3(NoFinding+Other)={len(self._pools[2])} | "
            f"intra_pool_mode={self.intra_pool_mode}"
        )
        print(msg)
        logger.info(msg)

    def _sample_from_pool(self, pool_idx: int, n: int) -> List[int]:
        """Sample n indices from a pool using configured weighting."""
        pool = self._pools[pool_idx]
        if not pool or n == 0:
            return []

        weights = self._pool_weights[pool_idx]
        if len(pool) == 0:
            return []

        # Normalize weights
        w = weights / weights.sum()
        chosen = self._np_rng.choice(pool, size=n, replace=True, p=w)
        return chosen.tolist()

    def _build_one_batch(self) -> List[int]:
        """Build one batch by drawing from each pool."""
        batch = []
        for pool_idx, n in enumerate(self.pool_counts):
            batch.extend(self._sample_from_pool(pool_idx, n))
        self._rng.shuffle(batch)
        return batch

    def __iter__(self):
        if self.sampler_mode == "random":
            indices = self._all_indices.copy()
            random.shuffle(indices)
            # Yield in chunks of batch_size to match expected length
            for start in range(0, self.batches_per_epoch * self.batch_size, self.batch_size):
                chunk = [indices[i % len(indices)]
                         for i in range(start, start + self.batch_size)]
                yield from chunk
            return

        # Tiered mode
        for _ in range(self.batches_per_epoch):
            yield from self._build_one_batch()

    def __len__(self) -> int:
        return self.batches_per_epoch * self.batch_size

    def compute_realized_ratios(self, n_batches: int = 100) -> Dict[str, float]:
        """
        Iterate n_batches from sampler, return actual per-pool fractions.
        Returns dict: {"pool1_frac": ..., "pool2_frac": ..., "pool3_frac": ...}
        """
        if self.sampler_mode == "random":
            return {"pool1_frac": 0.0, "pool2_frac": 0.0, "pool3_frac": 1.0}

        pool_counts_seen = [0, 0, 0]
        total = 0

        # Build pool membership lookup
        pool_membership: Dict[int, int] = {}
        for pi, pool in enumerate(self._pools):
            for idx in pool:
                pool_membership[idx] = pi

        for _ in range(n_batches):
            batch = self._build_one_batch()
            for idx in batch:
                pi = pool_membership.get(idx, 2)
                pool_counts_seen[pi] += 1
                total += 1

        if total == 0:
            return {}

        return {
            "pool1_frac": pool_counts_seen[0] / total,
            "pool2_frac": pool_counts_seen[1] / total,
            "pool3_frac": pool_counts_seen[2] / total,
        }


# Need math for sqrt_freq weighting
import math


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked binary cross-entropy with logits.

    Args:
        logits:     [B, C] float
        targets:    [B, C] float (0 or 1)
        mask:       [B, C] float (0 = ignore, 1 = include)
        pos_weight: [C] tensor or None

    Returns:
        scalar loss (mean over non-masked elements)
    """
    # Element-wise BCE
    per_elem = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )  # [B, C]

    # Apply mask
    masked = per_elem * mask

    # Average over non-masked elements
    n_active = mask.sum().clamp(min=1.0)
    return masked.sum() / n_active


# ---------------------------------------------------------------------------
# Empirical positive-rate auditing
# ---------------------------------------------------------------------------

def compute_sampled_stream_pos_rates(
    dataset: CheXpertLocalDatasetV2,
    sampler: TieredSamplerV2,
    labels: List[str],
    n_batches: int = 200,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Iterate n_batches from sampler, accumulate per-class positive counts.

    Returns:
        empirical_rates: {label: empirical positive rate in sampled stream}
        natural_rates:   {label: natural positive rate in full dataset}
    """
    pos_counts = {lbl: 0 for lbl in labels}
    total_samples = 0

    sampler_iter = iter(sampler)
    batch_size = sampler.batch_size

    for _ in range(n_batches):
        batch_indices = []
        for __ in range(batch_size):
            try:
                idx = next(sampler_iter)
                batch_indices.append(idx)
            except StopIteration:
                break

        for idx in batch_indices:
            row = dataset.df.iloc[idx]
            for lbl in labels:
                val = row.get(lbl, 0.0)
                if val is None:
                    val = 0.0
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                if val == 1.0:
                    pos_counts[lbl] += 1
            total_samples += 1

    empirical_rates = {
        lbl: pos_counts[lbl] / max(1, total_samples) for lbl in labels
    }

    # Natural rates
    n = len(dataset.df)
    natural_rates = {
        lbl: float((dataset.df[lbl] == 1.0).sum()) / n for lbl in labels
    }

    # Print comparison
    print(f"\n{'Label':<35} {'Natural':>10} {'Empirical':>10} {'Uplift':>8}")
    print("-" * 65)
    for lbl in labels:
        nat  = natural_rates[lbl]
        emp  = empirical_rates[lbl]
        uplift = (emp / nat) if nat > 1e-6 else float("inf")
        print(f"{lbl:<35} {nat:>10.4f} {emp:>10.4f} {uplift:>8.2f}x")
    print()

    return empirical_rates, natural_rates
