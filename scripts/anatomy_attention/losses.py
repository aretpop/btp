"""
losses.py
=========
Loss functions for the anatomy-conditioned architecture.

L_total = L_classification + λ_anatomy * L_anatomy + λ_diversity * L_diversity

L_classification: Asymmetric Loss (already used in Phase 1).
L_anatomy:        Weak regularisation encouraging attention to stay within
                  plausible anatomical regions (optional, λ default=0).
L_diversity:      Penalises pairwise cosine similarity between the 14
                  disease attention maps (optional, λ default=0).

Both auxiliary losses are OFF by default (λ=0) so the experiment starts
from a clean baseline. They are enabled through the config dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) — identical to Phase 1 implementation.
    Kept here so anatomy_attention/ is self-contained.
    """
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                w = torch.pow(1 - pt, gamma)
            loss = loss * w

        return -loss.mean()


def anatomy_regularisation_loss(
    disease_atts: torch.Tensor,
    anatomy_masks: Dict[str, torch.Tensor],
    plausible_region: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weakly encourages disease attention to remain within plausible anatomical
    regions. The penalty is the average attention mass OUTSIDE the combined
    lung+heart region.

    This is intentionally WEAK:
      - It does not force attention inside the organs.
      - It only penalises extreme anatomical implausibility.
      - λ_anatomy should start at 0 and be tested with {0.0001, 0.001, 0.01}.

    Args:
        disease_atts:    [B, 14, H, W]   — sigmoid attention maps
        anatomy_masks:   Dict with "left_lung", "right_lung", "heart"
                         each [B, 1, H_orig, W_orig] — resized here to H×W.
        plausible_region: Optional pre-computed [B, 1, H, W] combined mask.

    Returns:
        Scalar loss (mean attention outside plausible region).
    """
    if plausible_region is None:
        B, _, H, W = disease_atts.shape
        left  = F.interpolate(anatomy_masks["left_lung"],  (H, W), mode="bilinear", align_corners=False)
        right = F.interpolate(anatomy_masks["right_lung"], (H, W), mode="bilinear", align_corners=False)
        heart = F.interpolate(anatomy_masks["heart"],      (H, W), mode="bilinear", align_corners=False)
        # Union of lungs + heart, clamped to [0, 1]
        plausible_region = (left + right + heart).clamp(0, 1)  # [B, 1, H, W]

    outside = 1.0 - plausible_region                     # [B, 1, H, W]
    # Average attention mass outside region across all 14 disease maps
    attention_outside = (disease_atts * outside).mean()  # scalar
    return attention_outside


def attention_diversity_loss(disease_atts: torch.Tensor) -> torch.Tensor:
    """
    Penalises pairwise cosine similarity between the 14 disease attention maps.

    Goal: if all 14 maps are identical (collapsed), this loss is ~1.0.
    If maps are orthogonal, this loss is ~0.0.

    WARNING: This must NOT dominate classification. Use λ_diversity ∈ {0, 0.0001, 0.001}.
    If AUROC drops, set λ_diversity back to 0.

    Args:
        disease_atts: [B, 14, H, W]

    Returns:
        Scalar mean pairwise cosine similarity (lower = more diverse).
    """
    B, D, H, W = disease_atts.shape

    # Flatten spatial dims: [B, 14, H*W]
    atts_flat = disease_atts.view(B, D, -1)

    # L2-normalise along spatial dim
    atts_norm = F.normalize(atts_flat, p=2, dim=2)  # [B, 14, H*W]

    # Pairwise cosine similarity matrix: [B, 14, 14]
    sim_matrix = torch.bmm(atts_norm, atts_norm.transpose(1, 2))

    # Mask diagonal (self-similarity)
    mask = ~torch.eye(D, dtype=torch.bool, device=disease_atts.device)
    off_diag = sim_matrix[:, mask]  # [B, 14*13]

    return off_diag.mean()  # scalar ∈ [-1, 1], want to minimise


class CombinedLoss(nn.Module):
    """
    Combines classification + optional anatomy + optional diversity losses.

    Args:
        lambda_anatomy:   weight for L_anatomy   (default 0.0 = off)
        lambda_diversity: weight for L_diversity  (default 0.0 = off)
        asl_gamma_neg:    ASL parameter
        asl_gamma_pos:    ASL parameter
        asl_clip:         ASL clipping threshold
    """

    def __init__(
        self,
        lambda_anatomy: float = 0.0,
        lambda_diversity: float = 0.0,
        asl_gamma_neg: float = 4,
        asl_gamma_pos: float = 0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_anatomy   = lambda_anatomy
        self.lambda_diversity = lambda_diversity
        self.asl = AsymmetricLoss(asl_gamma_neg, asl_gamma_pos, asl_clip)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        disease_atts: Optional[torch.Tensor] = None,
        anatomy_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with individual loss components and total.
        """
        L_cls = self.asl(logits, targets)
        losses = {"classification": L_cls, "total": L_cls.clone()}

        if self.lambda_anatomy > 0 and disease_atts is not None and anatomy_masks is not None:
            L_anat = anatomy_regularisation_loss(disease_atts, anatomy_masks)
            losses["anatomy"] = L_anat
            losses["total"] = losses["total"] + self.lambda_anatomy * L_anat

        if self.lambda_diversity > 0 and disease_atts is not None:
            L_div = attention_diversity_loss(disease_atts)
            losses["diversity"] = L_div
            losses["total"] = losses["total"] + self.lambda_diversity * L_div

        return losses
