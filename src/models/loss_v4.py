"""
src/models/loss_v4.py
=====================
USLT Loss and class-balance alpha computation for CheXpert V4.

USLT Loss:
    bce = BCEWithLogits(logits, y)
    s = torch.where(u > 0, torch.clamp(1 - lam_u * u, min=eps), torch.ones_like(u))
    s = torch.where(blank > 0, torch.full_like(s, blank_weight), s)
    w = mask * alpha[None, :] * s
    L = sum(w * bce) / clamp(sum(mask), min=1)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import CHEXPERT_14_LABELS

logger = logging.getLogger(__name__)


def compute_alpha_c(
    train_df: pd.DataFrame,
    labels: List[str] = CHEXPERT_14_LABELS,
    alpha_mode: str = "effective_number",
    beta: float = 0.999,
    min_count: int = 1,
    reports_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Compute per-class alpha_c weights from the TRAIN split only using CERTAIN positives (raw == 1.0).

    Args:
        train_df: training split DataFrame
        labels: 14 class names
        alpha_mode: 'effective_number' | 'sqrt_inv_freq' | 'none'
        beta: effective-number parameter (default 0.999)
        min_count: floor to prevent division by zero
        reports_dir: directory to save reports/alpha_c.csv

    Returns:
        alpha_tensor: torch.Tensor [C]
        alpha_df: pd.DataFrame with class, n_certain_pos, alpha_hat, alpha_c
    """
    C = len(labels)
    n_c = np.zeros(C, dtype=np.float64)

    for i, lbl in enumerate(labels):
        if lbl in train_df.columns:
            n_c[i] = float((train_df[lbl] == 1.0).sum())
        else:
            n_c[i] = 0.0

    n_c_safe = np.maximum(n_c, min_count)

    if alpha_mode == "none":
        alpha_hat = np.ones(C, dtype=np.float64)
        alpha = np.ones(C, dtype=np.float64)
    elif alpha_mode == "sqrt_inv_freq":
        total = n_c_safe.sum()
        alpha_hat = 1.0 / np.sqrt(n_c_safe / total)
        alpha = C * alpha_hat / alpha_hat.sum()
    elif alpha_mode == "effective_number":
        if beta >= 1.0:
            alpha_hat = np.ones(C, dtype=np.float64)
        else:
            alpha_hat = (1.0 - beta) / (1.0 - np.power(beta, n_c_safe))
        alpha = C * alpha_hat / alpha_hat.sum()
    else:
        raise ValueError(f"Unknown alpha_mode '{alpha_mode}'. Supported: effective_number, sqrt_inv_freq, none")

    min_max_ratio = float(alpha.max() / max(1e-8, alpha.min())) if alpha.min() > 0 else 1.0

    records = []
    for i, lbl in enumerate(labels):
        records.append({
            "class_idx": i,
            "class_name": lbl,
            "n_certain_pos": int(n_c[i]),
            "alpha_hat": float(alpha_hat[i]),
            "alpha_c": float(alpha[i]),
        })
    alpha_df = pd.DataFrame(records)

    # Print and log alpha_c table at startup
    table_lines = [
        "=" * 70,
        f"USLT Class Weights (alpha_c) — mode={alpha_mode}, beta={beta:.4f}",
        f"{'Class':28s} | {'n_certain_pos':14s} | {'alpha_hat':10s} | {'alpha_c':8s}",
        "-" * 70,
    ]
    for _, row in alpha_df.iterrows():
        table_lines.append(f"{row['class_name']:28s} | {row['n_certain_pos']:14,d} | {row['alpha_hat']:10.4f} | {row['alpha_c']:8.4f}")
    table_lines.extend([
        "-" * 70,
        f"Max / Min alpha ratio: {min_max_ratio:.4f}",
    ])
    if min_max_ratio < 1.5 and alpha_mode != "none":
        table_lines.append("[WARNING] alpha_c max/min ratio is < 1.5! Class weights are nearly uniform.")
    table_lines.append("=" * 70)

    for line in table_lines:
        print(line)
        logger.info(line)

    # Save to reports/alpha_c.csv
    if reports_dir is not None:
        r_dir = Path(reports_dir)
        r_dir.mkdir(parents=True, exist_ok=True)
        csv_path = r_dir / "alpha_c.csv"
        alpha_df.to_csv(csv_path, index=False)
        logger.info(f"Saved alpha_c weights table to {csv_path}")

    alpha_tensor = torch.from_numpy(alpha).float()
    return alpha_tensor, alpha_df


def uslt_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    blank: torch.Tensor,
    mask: torch.Tensor,
    alpha: torch.Tensor,
    lam_u: Union[float, torch.Tensor] = 0.5,
    eps: float = 0.1,
    blank_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE with soft targets and uncertainty suppression per Section 5 specification.

    Args:
        logits:       [B, C] unnormalized network outputs
        y:            [B, C] soft target values in [0, 1]
        u:            [B, C] binary indicator for uncertain (raw == -1)
        blank:        [B, C] binary indicator for missing/unmentioned (raw was NaN)
        mask:         [B, C] binary mask (1 = include, 0 = ignore)
        alpha:        [C] class balance weights
        lam_u:        uncertainty suppression strength (float or [C] tensor)
        eps:          minimum suppression factor floor
        blank_weight: loss weight factor for blank labels (default 1.0)
    """
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

    # Uncertainty suppression: s = max(eps, 1 - lam_u * u) when u > 0, else 1
    if isinstance(lam_u, torch.Tensor):
        lam_u_t = lam_u.to(logits.device).unsqueeze(0)
    else:
        lam_u_t = float(lam_u)

    s = torch.where(u > 0, torch.clamp(1.0 - lam_u_t * u, min=eps), torch.ones_like(u))
    # Apply blank weight if blank > 0
    s = torch.where(blank > 0, torch.full_like(s, float(blank_weight)), s)

    # Broadcast alpha: [C] -> [1, C]
    alpha_broadcast = alpha.to(logits.device).unsqueeze(0)
    w = mask * alpha_broadcast * s

    return (w * bce).sum() / mask.sum().clamp(min=1.0)


class USLTLossV4(nn.Module):
    """Module wrapper for uslt_loss."""

    def __init__(
        self,
        alpha: torch.Tensor,
        lam_u: Union[float, torch.Tensor] = 0.5,
        eps: float = 0.1,
        blank_weight: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        if isinstance(lam_u, torch.Tensor):
            self.register_buffer("lam_u", lam_u.float())
        else:
            self.lam_u = float(lam_u)
        self.eps = float(eps)
        self.blank_weight = float(blank_weight)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor,
        blank: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return uslt_loss(
            logits=logits,
            y=y,
            u=u,
            blank=blank,
            mask=mask,
            alpha=self.alpha,
            lam_u=self.lam_u,
            eps=self.eps,
            blank_weight=self.blank_weight,
        )

