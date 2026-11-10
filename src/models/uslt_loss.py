"""
src/models/uslt_loss.py

Uncertainty-Suppressed Long-Tailed (USLT) loss for multi-label
chest X-ray classification.

Loss formula (per prompt Section 4):
  s_c   = max(eps, 1 - lambda * u_c)        uncertainty suppression
  w_c   = m_c * alpha_c * s_c               per-sample per-class weight
  L_bce = sum(w * BCE(logit, y)) / sum(m)   masked, class-balanced BCE

Class weights alpha_c computed from CERTAIN positive counts in training split:
  alpha_hat_c = (1 - beta) / (1 - beta^n_c)
  alpha_c = C * alpha_hat_c / sum(alpha_hat)

Also supports `alpha_mode: sqrt_inv_freq` as alternative.

IMPORTANT: When loss_type='uslt', do NOT also apply pos_weight in BCE.
See double-correction warning in compute_alpha_weights().
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_alpha_weights(
    df: pd.DataFrame,
    labels: List[str],
    beta: float = 0.999,
    alpha_mode: str = "effective_num",   # "effective_num" | "sqrt_inv_freq"
    min_count: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute per-class alpha weights from CERTAIN positives (raw==1) only.

    Args:
        df: training DataFrame with label columns (values: 1.0, 0.0, -1.0, NaN).
        labels: list of class name strings (length C).
        beta: effective-number parameter (default 0.999).
        alpha_mode: 'effective_num' (default) or 'sqrt_inv_freq'.
        min_count: floor to prevent division explosion (default 1).

    Returns:
        alpha (np.ndarray [C]): normalized class weights.
        info  (dict): counts, raw alpha_hat, min/max ratio.
    """
    C = len(labels)
    n_c = np.zeros(C, dtype=float)

    for i, lbl in enumerate(labels):
        if lbl in df.columns:
            # Only count CERTAIN positives (raw == 1.0, not uncertain -1.0)
            n_c[i] = float((df[lbl] == 1.0).sum())
        else:
            n_c[i] = 0.0

    # Floor at min_count to avoid log(0) or zero-weight classes
    n_c_safe = np.maximum(n_c, min_count)

    if alpha_mode == "sqrt_inv_freq":
        total = n_c_safe.sum()
        alpha_hat = 1.0 / np.sqrt(n_c_safe / total)
    else:
        # Effective Number of Samples: (1 - beta) / (1 - beta^n_c)
        if beta >= 1.0:
            # Degenerate case: uniform weights
            alpha_hat = np.ones(C, dtype=float)
        else:
            alpha_hat = (1.0 - beta) / (1.0 - np.power(beta, n_c_safe))

    # Normalize: sum = C (so average weight == 1)
    alpha = C * alpha_hat / alpha_hat.sum()

    min_max_ratio = float(alpha.min() / alpha.max()) if alpha.max() > 0 else 1.0

    info = {
        "n_c": n_c.tolist(),
        "alpha_hat": alpha_hat.tolist(),
        "alpha": alpha.tolist(),
        "min_alpha": float(alpha.min()),
        "max_alpha": float(alpha.max()),
        "min_max_ratio": min_max_ratio,
        "beta": beta,
        "alpha_mode": alpha_mode,
    }

    # Log alpha table
    logger.info("=" * 60)
    logger.info("USLT Class Weights (alpha_c) — beta=%.4f, mode=%s", beta, alpha_mode)
    logger.info("%-35s  %8s  %8s", "Class", "n_cert+", "alpha_c")
    logger.info("-" * 60)
    for i, lbl in enumerate(labels):
        logger.info("  %-33s  %8.0f  %8.4f", lbl, n_c[i], alpha[i])
    logger.info("-" * 60)
    logger.info("  min/max alpha ratio: %.4f", min_max_ratio)
    if min_max_ratio > 0.9:
        logger.warning(
            "USLT: alpha weights are nearly uniform (min/max=%.4f). "
            "With beta=%.3f all n_c>>1 → effective_num ≈ 1/(1-beta)=const. "
            "Consider beta=0.99 (flag: loss.beta=0.99) or alpha_mode=sqrt_inv_freq.",
            min_max_ratio, beta,
        )
    logger.info("=" * 60)

    return alpha, info


# ---------------------------------------------------------------------------
# USLT Loss
# ---------------------------------------------------------------------------

class USLTLoss(torch.nn.Module):
    """
    Uncertainty-Suppressed Long-Tailed BCE loss.

    Args:
        alpha (Tensor [C]): class balance weights from compute_alpha_weights().
        lambda_u (float): uncertainty suppression strength in [0, 1].
        eps (float): minimum suppression factor (default 0.1).
        reduction (str): 'mean' or 'sum' over valid positions.

    Forward inputs:
        logits  [B, C]
        targets [B, C]  – soft labels in [0, 1]
        mask    [B, C]  – 1 = observed, 0 = missing
        uncertain [B, C] – 1 = raw label was -1 (uncertain), 0 otherwise

    Returns:
        loss scalar.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        lambda_u: float = 0.5,
        eps: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.lambda_u = float(lambda_u)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        uncertain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:   [B, C]
            targets:  [B, C]  soft targets in [0,1]
            mask:     [B, C]  1=valid, 0=ignore
            uncertain:[B, C]  1=uncertain label (raw==-1)
        """
        # Per-sample per-class BCE (no reduction)
        per_elem = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [B, C]

        # Uncertainty suppression: s = max(eps, 1 - lambda * u)
        s = torch.clamp(1.0 - self.lambda_u * uncertain, min=self.eps)  # [B, C]

        # Class weight broadcast: alpha [C] → [1, C]
        alpha = self.alpha.unsqueeze(0)   # [1, C]

        # Combined weight per element
        w = mask * alpha * s   # [B, C]

        denom = mask.sum()
        if denom < 1.0:
            return logits.sum() * 0.0  # all masked

        loss = (w * per_elem).sum() / denom
        return loss


# ---------------------------------------------------------------------------
# Label policy helpers
# ---------------------------------------------------------------------------

def apply_label_policy(
    raw_labels: torch.Tensor,
    policy: str = "u_zeros",
    uncertain_target: float = 0.5,
    per_class_policy: Optional[Dict[int, str]] = None,
    nan_policy: str = "ignore",
    nan_target: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert raw label tensor to (targets, uncertain_flags, mask).

    raw_labels: float tensor [B, C] with values in {1.0, 0.0, -1.0, NaN}.
    policy: global uncertain label handling:
        'u_zeros'  – uncertain → target=0.0, mask=1
        'u_ones'   – uncertain → target=1.0, mask=1
        'u_ignore' – uncertain → target=0.0, mask=0 (excluded from loss)
        'u_soft'   – uncertain → target=uncertain_target, mask=1

    nan_policy: missing/unmentioned label handling:
        'ignore' – NaN → target=0.0, mask=0 (excluded from loss)
        'soft'   – NaN → target=nan_target (e.g. 0.5), mask=1, uncertain=1

    per_class_policy: dict {class_idx: policy_string} to override per class.

    Returns:
        targets   [B, C] float in [0, 1]
        uncertain [B, C] float (1 = originally uncertain or soft NaN)
        mask      [B, C] float (1 = contributes to loss)
    """
    B, C = raw_labels.shape
    targets = torch.zeros_like(raw_labels)
    mask = torch.ones_like(raw_labels)
    uncertain = torch.zeros_like(raw_labels)

    is_nan = torch.isnan(raw_labels)
    is_uncertain = (raw_labels == -1.0)
    is_positive = (raw_labels == 1.0)
    is_negative = (raw_labels == 0.0)

    # Positives and negatives are always straightforward
    targets[is_positive] = 1.0
    targets[is_negative] = 0.0

    if nan_policy == "soft":
        targets[is_nan] = float(nan_target)
        mask[is_nan] = 1.0
        uncertain[is_nan] = 1.0  # Treat NaN as uncertain for USLT suppression
    else:  # "ignore" (default)
        mask[is_nan] = 0.0          # NaN → excluded
        targets[is_nan] = 0.0

    # Mark uncertain
    uncertain[is_uncertain] = 1.0

    def _apply_single_policy(pol: str, cls_mask: torch.Tensor):
        # cls_mask: bool [B, C] marking positions to update
        uncertain_pos = is_uncertain & cls_mask
        if pol == "u_ones":
            targets[uncertain_pos] = 1.0
            mask[uncertain_pos] = 1.0
        elif pol == "u_ignore":
            targets[uncertain_pos] = 0.0
            mask[uncertain_pos] = 0.0
        elif pol == "u_soft":
            targets[uncertain_pos] = float(uncertain_target)
            mask[uncertain_pos] = 1.0
        else:  # u_zeros (default)
            targets[uncertain_pos] = 0.0
            mask[uncertain_pos] = 1.0

    # Apply global policy to all classes
    all_mask = torch.ones_like(raw_labels, dtype=torch.bool)
    _apply_single_policy(policy, all_mask)

    # Apply per-class overrides
    if per_class_policy:
        for cls_idx, pol in per_class_policy.items():
            cls_col_mask = torch.zeros_like(raw_labels, dtype=torch.bool)
            cls_col_mask[:, cls_idx] = True
            _apply_single_policy(pol, cls_col_mask)

    return targets, uncertain, mask


# ---------------------------------------------------------------------------
# Co-occurrence matrix for div_mode=cooc_masked
# ---------------------------------------------------------------------------

def compute_label_cooccurrence(
    df: pd.DataFrame,
    labels: List[str],
    certain_only: bool = True,
) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity between labels from the training split.

    Args:
        df: training DataFrame (raw labels: 1.0=pos, 0.0=neg, -1.0=uncertain, NaN=missing)
        labels: list of label names (length C)
        certain_only: if True, only use certain positives (raw==1.0)

    Returns:
        jaccard [C, C] float array.
    """
    C = len(labels)
    pos = np.zeros((len(df), C), dtype=bool)
    for i, lbl in enumerate(labels):
        if lbl in df.columns:
            vals = df[lbl].values
            if certain_only:
                pos[:, i] = (vals == 1.0)
            else:
                pos[:, i] = (vals == 1.0) | (vals == -1.0)

    jaccard = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(C):
            if i == j:
                jaccard[i, j] = 1.0
                continue
            inter = (pos[:, i] & pos[:, j]).sum()
            union = (pos[:, i] | pos[:, j]).sum()
            jaccard[i, j] = inter / max(union, 1)

    return jaccard

