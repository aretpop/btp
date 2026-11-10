"""
anatomy_encoder.py
==================
AnatomyEncoder: converts 3-channel anatomical mask stack into
feature-map-aligned anatomy features.

Input format expected:
    anatomy_masks["left_lung"]  : [B, 1, H_orig, W_orig]  — binary or soft [0,1]
    anatomy_masks["right_lung"] : [B, 1, H_orig, W_orig]
    anatomy_masks["heart"]      : [B, 1, H_orig, W_orig]

These are stacked into [B, 3, H_orig, W_orig] and resized to the
backbone feature-map resolution (7×7 for 224×224 input).

IMPORTANT — Interpolation choice:
    We use bilinear interpolation (align_corners=False) because:
    1. We intentionally want SOFT masks at feature-map resolution.
       Hard binary values on a 7×7 grid would be too coarse.
    2. Bilinear preserves smooth spatial gradients for attention gating.
    The alternative (nearest-neighbor) preserves binary mask values but
    produces harsh 0/1 boundaries that are inappropriate for soft guidance.

IMPORTANT — Soft anatomical priors (NOT hard masking):
    We do NOT simply multiply features by the mask (hard mask).
    Instead the AnatomyEncoder learns a soft gating signal.
    The AnatomyFusion module adds this as:
        F_guided = F * (1 + alpha * sigmoid(anatomy_gate))
    where alpha is learnable and starts near 0.
    Inside the anatomical region → gate > 0 → slight boost.
    Outside the anatomical region → gate < 0 → slight suppression.
    Information is never completely erased.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AnatomyEncoder(nn.Module):
    """
    Small CNN that encodes a 3-channel anatomical mask stack into
    a feature map matching the backbone's spatial resolution.

    Args:
        in_channels:   3 (left_lung, right_lung, heart) or 5 with boundaries.
        mid_channels:  Intermediate channels in the encoder (default 32).
        out_channels:  Must match backbone feature_dim C.
        use_boundary:  If True, expect 5-channel input (3 masks + 2 boundaries).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        out_channels: int = 1280,
        use_boundary: bool = False,
    ):
        super().__init__()
        self.use_boundary = use_boundary
        actual_in = in_channels + (2 if use_boundary else 0)

        self.encoder = nn.Sequential(
            # Layer 1: local texture
            nn.Conv2d(actual_in, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Layer 2: larger receptive field
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels * 2),
            nn.ReLU(inplace=True),
            # Project to match backbone channels
            nn.Conv2d(mid_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        anatomy_masks: Dict[str, torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Args:
            anatomy_masks: Dict with keys "left_lung", "right_lung", "heart",
                           each [B, 1, H_orig, W_orig], values in [0, 1].
                           Optional keys "lung_boundary", "heart_boundary".
            target_h, target_w: Feature map spatial resolution (e.g. 7, 7).

        Returns:
            anatomy_features: [B, C, target_h, target_w]
        """
        left  = anatomy_masks["left_lung"]    # [B, 1, H, W]
        right = anatomy_masks["right_lung"]   # [B, 1, H, W]
        heart = anatomy_masks["heart"]        # [B, 1, H, W]

        channels = [left, right, heart]

        if self.use_boundary:
            assert "lung_boundary" in anatomy_masks and "heart_boundary" in anatomy_masks, \
                "use_boundary=True but boundary masks are missing from anatomy_masks."
            channels.append(anatomy_masks["lung_boundary"])
            channels.append(anatomy_masks["heart_boundary"])

        # Stack → [B, 3 or 5, H_orig, W_orig]
        mask_stack = torch.cat(channels, dim=1)

        # Resize to backbone feature-map resolution using bilinear interpolation.
        # Bilinear chosen for smooth soft-prior gradients (see module docstring).
        if mask_stack.shape[-2:] != (target_h, target_w):
            mask_stack = F.interpolate(
                mask_stack,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        # mask_stack: [B, 3or5, target_h, target_w]

        # Encode into anatomy_features: [B, C, target_h, target_w]
        return self.encoder(mask_stack)


class AnatomyFusion(nn.Module):
    """
    Conditions backbone features on anatomy features via soft gating.

    Implements: F_guided = F * (1 + alpha * sigmoid(gate))

    where gate is generated from concat(F, anatomy_features).

    alpha is a learnable scalar initialised near 0 so that at the start
    of training the anatomy has minimal effect, letting the backbone
    features stabilise before anatomy guidance kicks in.

    Args:
        channels: Feature map channels C.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Learnable scale starts near 0 (tanh of -3 ≈ -0.995 → alpha ≈ 0.005)
        self._alpha_raw = nn.Parameter(torch.tensor(-3.0))

        # Gate generator: concat(F, anatomy) → [B, 2C, H, W] → [B, C, H, W]
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Constrain alpha to [0, 1] via sigmoid."""
        return torch.sigmoid(self._alpha_raw)

    def forward(
        self,
        F: torch.Tensor,
        anatomy_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        F:                  [B, C, H, W]
        anatomy_features:   [B, C, H, W]
        Returns:            [B, C, H, W]   (anatomy-guided features)
        """
        # Concatenate along channel axis
        combined = torch.cat([F, anatomy_features], dim=1)  # [B, 2C, H, W]
        gate = self.gate_gen(combined)                       # [B, C, H, W]

        # Soft multiplicative guidance
        # sigmoid(gate) ∈ (0, 1); alpha controls overall strength
        F_guided = F * (1.0 + self.alpha * torch.sigmoid(gate))
        return F_guided
