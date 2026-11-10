"""
train_phase3_regional_v3.py
============================
Phase 3: Modular Regional Attention for Chest X-Ray Classification.

Fixes every known bug from V1/V2. All components configurable via JSON config.

KEY FIXES vs V1/V2:
  1. Block-5 channels are backbone-specific (112 for B0/B1, 136 for B3).
  2. 14×14 attention maps are NOT downsampled. Instead f_final is upsampled.
  3. AnatomicalAttention alpha=0 replaced with bounded sigmoid gate.
  4. AnatomicalAttention now has a learned fallback spatial prior when mask=None.
  5. Concentration loss operates on properly normalized probability distributions.
  6. Optional Gram-matrix diversity loss across disease maps.
  7. Optional global residual pathway: F_d = F_global + lambda * F_attended_d.
  8. Spatial softmax vs sigmoid attention selectable via config.
  9. All tensor shapes verified for B0, B1, B3.

ABLATION VARIANTS (controlled by config):
  M0: Phase 1 baseline (not trained here, loaded for comparison)
  M1: V1 reproduction (7×7 maps, sigmoid)
  M2A: Native 14×14 — Block 5 features for both attention AND classification
  M2B: Upsample 14×14 — Block 8 → upsample → fuse with Block 5 (DEFAULT)
  M3: M2B + corrected anatomy-aware attention
  M4: M3 + Gram-matrix diversity loss
  M5: M3 + spatial softmax instead of sigmoid
  M6: Best combined (M2B + M3 + global residual + tuned losses)

Usage:
  python scripts/train_phase3_regional_v3.py --config configs/phase3/M2B_b1.json
  python scripts/train_phase3_regional_v3.py --config configs/phase3/M3_b1.json
  python scripts/train_phase3_regional_v3.py --smoke-test --backbone efficientnet_b1
"""

import os, sys, copy, json, time, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models, transforms
import mlflow

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_ROOT      = PROJECT_ROOT / "data" / "nih_cxr14"
IMAGES_DIR     = DATA_ROOT / "images" / "images-224"
CSV_PATH       = DATA_ROOT / "metadata" / "Data_Entry_2017.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3_regional"
CONFIG_DIR     = PROJECT_ROOT / "configs" / "phase3"

# Add scripts/ to path for Phase 1 imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms,
    NUM_CLASSES, CLASSES, AsymmetricLoss
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

# ── Backbone-specific constants (VERIFIED via forward pass) ────────────────────
BLOCK5_CHANNELS = {
    "efficientnet_b0": 112,
    "efficientnet_b1": 112,
    "efficientnet_b3": 136,
}
BLOCK8_CHANNELS = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b3": 1536,
}
ATT_RES_HIGH = 14   # Block 5 spatial resolution
ATT_RES_LOW  = 7    # Block 8 spatial resolution

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Multi-Scale Feature Block (corrected from V1)
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleFeatureBlock(nn.Module):
    """
    Captures features at 1×1, 3×3, dilated 3×3 (r=2), dilated 3×3 (r=3).
    All branches individually normalized before fusion.
    in_channels: Block 5 or Block 8 channel count (backbone-dependent).
    out_channels: configurable output dimension (default 256 for edge).
    """
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        mid = out_channels // 4

        def branch(k, d=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, mid, k, padding=k//2 * d, dilation=d, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
            )

        self.b1 = branch(1)        # 1×1: pointwise
        self.b2 = branch(3)        # 3×3: local
        self.b3 = branch(3, d=2)   # 3×3 dil=2: medium range
        self.b4 = branch(3, d=3)   # 3×3 dil=3: large range

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Channel Attention (improved CBAM — avg + max pool)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation with both avg and max pool (stronger than avg-only)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * gate


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Spatial Attention (CBAM-style)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Channel-pooling followed by spatial gating conv."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Corrected Anatomical Attention
# ─────────────────────────────────────────────────────────────────────────────
class AnatomicalAttention(nn.Module):
    """
    Anatomy-aware spatial gating — CORRECTED version.

    Fixes vs V1:
      - alpha no longer zero-initialized (was killing mask contribution)
      - Learned fallback when mask=None (not silent identity)
      - Bounded gate via sigmoid (instead of raw learnable alpha)
      - mask_strength init=0.5 (50% mask influence at start)

    Modes (controlled by anatomy_mode config):
      'gate'     : A_final = A_learned * sigmoid(mask_strength) * mask + A_learned * (1-sigmoid(mask_strength))
      'residual' : A_final = A_learned + sigmoid(mask_strength) * mask_guided_features
      'prior'    : A_final = A_learned * softmax(A_learned * prior_mask)  [soft prior]
      'none'     : A_final = A_learned  (no anatomical information)

    Parameters:
      channels    : number of feature channels
      anatomy_mode: 'gate' | 'residual' | 'prior' | 'none'
    """
    def __init__(self, channels: int, anatomy_mode: str = "gate"):
        super().__init__()
        self.anatomy_mode = anatomy_mode

        # Learned spatial attention (always active, provides fallback when mask=None)
        self.learned_spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

        if anatomy_mode in ("gate", "residual", "prior"):
            # Bounded mask strength: sigmoid(mask_gate) in (0, 1)
            # Init at 0.0 → sigmoid(0) = 0.5 (equal blend at start)
            self.mask_gate = nn.Parameter(torch.zeros(1))

            # Lightweight fusion for mask-guided features
            self.mask_proj = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape

        # Always compute learned spatial attention as base
        learned_att = self.learned_spatial(x)  # [B, 1, H, W]

        if self.anatomy_mode == "none" or mask is None:
            # Fallback: pure learned spatial attention
            return x * learned_att

        # Resize mask to feature-map resolution
        mask_resized = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
        mask_resized = mask_resized.clamp(0, 1)  # Ensure [0,1] range

        alpha = torch.sigmoid(self.mask_gate)  # bounded in (0, 1)

        if self.anatomy_mode == "gate":
            # Blend learned attention with anatomy-guided attention
            # alpha=0 → pure learned; alpha=1 → pure anatomy
            blended_att = (1 - alpha) * learned_att + alpha * mask_resized
            return x * blended_att

        elif self.anatomy_mode == "residual":
            # Residual: add anatomy-gated features on top of learned attention
            x_learned = x * learned_att
            x_masked = self.mask_proj(x * mask_resized)
            return x_learned + alpha * x_masked

        elif self.anatomy_mode == "prior":
            # Soft prior: anatomy mask modulates learned attention multiplicatively
            # A_final = normalize(A_learned * mask) applied to features
            prior_att = learned_att * mask_resized
            # Renormalize per spatial location to prevent suppression collapse
            prior_att = prior_att / (prior_att.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6))
            prior_att = torch.sigmoid(prior_att)  # re-gate
            return x * prior_att

        return x * learned_att


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: Disease Attention Generator (configurable normalization)
# ─────────────────────────────────────────────────────────────────────────────
class DiseaseAttentionGenerator(nn.Module):
    """
    Generates one spatial attention map per disease class.

    Input:  [B, in_channels, H, W]  (H=W=14 from Block 5)
    Output: [B, num_classes, H, W]  (H=W=14 — NOT downsampled)

    att_norm options:
      'sigmoid'         : independent gates per location (allows multi-peak)
      'spatial_softmax' : forces distribution over space (single peak)
    """
    def __init__(self, in_channels: int, num_classes: int = 14,
                 hidden: int = 128, att_norm: str = "sigmoid"):
        super().__init__()
        self.att_norm = att_norm
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, num_classes, 1),  # [B, num_classes, H, W]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)  # [B, C, H, W]

        if self.att_norm == "sigmoid":
            return torch.sigmoid(logits)

        elif self.att_norm == "spatial_softmax":
            B, C, H, W = logits.shape
            # Softmax over spatial dimension for each disease independently
            flat = logits.view(B, C, -1)             # [B, C, H*W]
            soft = F.softmax(flat, dim=2)             # sum over H*W = 1
            return soft.view(B, C, H, W)

        else:
            raise ValueError(f"Unknown att_norm: {self.att_norm}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL: RegionalAttentionXRayModel (V3)
# ─────────────────────────────────────────────────────────────────────────────
class RegionalAttentionXRayModel(nn.Module):
    """
    Phase 3 Regional Attention Model — fully corrected and modular.

    Architecture (M2B default):
        Input [B, 3, 224, 224]
            ↓ Frozen EfficientNet backbone
            ├── Block 5 → [B, 112, 14, 14]  (or 136 for B3)
            │       ↓
            │   DiseaseAttentionGenerator → [B, 14, 14, 14]   ← kept at 14×14!
            │
            └── Block 8 → [B, 1280, 7, 7]
                    ↓ MultiScaleFeatureBlock → [B, feat_dim, 7, 7]
                    ↓ ChannelAttention
                    ↓ SpatialAttention
                    ↓ AnatomicalAttention (mask optional)
                    ↓ BilinearUpsample → [B, feat_dim, 14, 14]
                    ↓ Fuse with Block 5 projected features (optional)
                    ↓
                    For each disease d=0..13:
                        f_shared × disease_att[d] → masked GAP
                        + global residual (optional)
                        → Linear(feat_dim, 1)
                    ↓
                [B, 14] logits

    Config keys:
        backbone       : str   — efficientnet_b0 / b1 / b3
        use_multiscale : bool  — whether to use MultiScaleFeatureBlock
        feat_dim       : int   — output channels of multi-scale block (default 256)
        use_channel_att: bool  — ChannelAttention after multi-scale
        use_spatial_att: bool  — SpatialAttention after channel
        anatomy_mode   : str   — none / gate / residual / prior
        att_norm       : str   — sigmoid / spatial_softmax
        use_global_res : bool  — add global GAP residual to each disease head
        att_resolution : int   — 7 (V1 mode) or 14 (V3 mode)
        use_b5_fusion  : bool  — fuse upsampled b8 with b5 features at 14×14
    """

    def __init__(self, original_backbone: nn.Module, backbone_name: str, cfg: dict):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes   = cfg.get("num_classes", 14)
        self.att_resolution = cfg.get("att_resolution", 14)
        self.use_global_res = cfg.get("use_global_res", True)
        self.use_b5_fusion  = cfg.get("use_b5_fusion", False)

        b5_ch = BLOCK5_CHANNELS[backbone_name]
        b8_ch = BLOCK8_CHANNELS[backbone_name]
        feat_dim = cfg.get("feat_dim", 256)

        # --- Frozen backbone (deep copy — original untouched) ---
        self.backbone = copy.deepcopy(original_backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Shared feature refinement path (on Block 8 features) ---
        if cfg.get("use_multiscale", True):
            self.refine = MultiScaleFeatureBlock(b8_ch, feat_dim)
        else:
            # Simple projection
            self.refine = nn.Sequential(
                nn.Conv2d(b8_ch, feat_dim, 1, bias=False),
                nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True)
            )

        if cfg.get("use_channel_att", True):
            self.channel_att = ChannelAttention(feat_dim, cfg.get("channel_reduction", 16))
        else:
            self.channel_att = nn.Identity()

        if cfg.get("use_spatial_att", True):
            self.spatial_att = SpatialAttention(cfg.get("spatial_kernel", 7))
        else:
            self.spatial_att = nn.Identity()

        anatomy_mode = cfg.get("anatomy_mode", "none")
        self.anatomy_att = AnatomicalAttention(feat_dim, anatomy_mode)

        # --- Optional Block-5 projection for fusion at 14×14 ---
        if self.use_b5_fusion and self.att_resolution == 14:
            self.b5_proj = nn.Sequential(
                nn.Conv2d(b5_ch, feat_dim // 2, 1, bias=False),
                nn.BatchNorm2d(feat_dim // 2), nn.ReLU(inplace=True)
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feat_dim + feat_dim // 2, feat_dim, 1, bias=False),
                nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True)
            )

        # --- Disease attention maps at target resolution ---
        self.disease_att = DiseaseAttentionGenerator(
            in_channels=b5_ch,
            num_classes=self.num_classes,
            hidden=cfg.get("att_hidden", 128),
            att_norm=cfg.get("att_norm", "sigmoid"),
        )

        # --- 14 independent classifiers ---
        head_in = feat_dim
        if self.use_global_res:
            # Global residual branch: separate global GAP path
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_proj = nn.Linear(feat_dim, feat_dim)

        self.classifiers = nn.ModuleList([
            nn.Linear(head_in, 1) for _ in range(self.num_classes)
        ])

        self.feat_dim = feat_dim
        self.b5_ch = b5_ch
        self.b8_ch = b8_ch

    def train(self, mode=True):
        """Override: backbone always stays in eval (frozen BN statistics)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _get_backbone_features(self, x: torch.Tensor):
        """
        Extract Block 5 and Block 8 features.
        Returns:
            f5: [B, b5_ch, 14, 14]
            f8: [B, b8_ch,  7,  7]
        """
        if hasattr(self.backbone, "backbone"):
            blocks = self.backbone.backbone.features
        else:
            blocks = self.backbone.features

        z = x
        f5 = None
        for i, block in enumerate(blocks):
            z = block(z)
            if i == 5:
                f5 = z  # [B, b5_ch, 14, 14]
        f8 = z            # [B, b8_ch, 7, 7]
        return f5, f8

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                return_attention: bool = False):
        B = x.shape[0]

        # ─── 1. Frozen backbone ────────────────────────────────────────────
        with torch.no_grad():
            f5, f8 = self._get_backbone_features(x)
        # f5: [B, b5_ch, 14, 14]
        # f8: [B, b8_ch,  7,  7]

        # ─── 2. Shared refinement on Block 8 (7×7) ────────────────────────
        f_ref = self.refine(f8)           # [B, feat_dim, 7, 7]
        f_ref = self.channel_att(f_ref)   # [B, feat_dim, 7, 7]
        f_ref = self.spatial_att(f_ref)   # [B, feat_dim, 7, 7]
        f_ref = self.anatomy_att(f_ref, mask)  # [B, feat_dim, 7, 7]

        # ─── 3. Upsample to target resolution ─────────────────────────────
        if self.att_resolution == 14:
            # Upsample refined 7×7 features to 14×14
            f_shared = F.interpolate(f_ref, size=(14, 14), mode="bilinear", align_corners=False)
            # [B, feat_dim, 14, 14]

            if self.use_b5_fusion:
                # Fuse with Block 5 features (same resolution, richer spatial detail)
                f5_proj = self.b5_proj(f5)       # [B, feat_dim//2, 14, 14]
                f_shared = self.fusion_conv(
                    torch.cat([f_shared, f5_proj], dim=1)
                )   # [B, feat_dim, 14, 14]
        else:
            # V1 mode: keep at 7×7
            f_shared = f_ref   # [B, feat_dim, 7, 7]

        # ─── 4. Disease attention maps (from Block 5 — 14×14) ─────────────
        disease_maps = self.disease_att(f5)   # [B, 14, 14, 14]

        if self.att_resolution == 7:
            # V1 compatibility: downsample attention maps to match 7×7 features
            disease_maps_for_cls = F.adaptive_avg_pool2d(disease_maps, (7, 7))
        else:
            disease_maps_for_cls = disease_maps  # keep at 14×14

        # ─── 5. Optional global residual ──────────────────────────────────
        if self.use_global_res:
            f_global = self.global_pool(f_shared).squeeze(-1).squeeze(-1)  # [B, feat_dim]
            f_global = self.global_proj(f_global)                           # [B, feat_dim]

        # ─── 6. Disease-specific classification ───────────────────────────
        # Mathematically equivalent to (f_shared.unsqueeze(1) * disease_maps_for_cls.unsqueeze(2)).mean((3,4))
        # but computes via a single batched matrix multiply (bmm), eliminating the heavy 5D tensor!
        H, W = f_shared.shape[2], f_shared.shape[3]
        f_flat = f_shared.flatten(2)                    # [B, feat_dim, H*W]
        d_flat = disease_maps_for_cls.flatten(2)        # [B, 14, H*W]
        z = torch.bmm(d_flat, f_flat.transpose(1, 2)) / (H * W)  # [B, 14, feat_dim]

        logits_list = []
        for d in range(self.num_classes):
            z_d = z[:, d, :]     # [B, feat_dim]
            if self.use_global_res:
                z_d = z_d + 0.1 * f_global   # weighted residual (0.1 keeps attention dominant)
            logit_d = self.classifiers[d](z_d)   # [B, 1]
            logits_list.append(logit_d)

        logits = torch.cat(logits_list, dim=1)   # [B, 14]

        if return_attention:
            return logits, disease_maps   # disease_maps at 14×14 always
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class AttentionConcentrationLoss(nn.Module):
    """
    Entropy-based spatial concentration loss.
    Operates on properly normalized probability distributions.

    L_entropy = mean over [B, D] of H(p_d)
    where p_d is a spatial probability distribution normalized to sum=1.

    Minimizing this encourages each disease map to concentrate on a small region.
    Use with a small lambda (e.g. 0.005–0.05) — too large collapses maps arbitrarily.
    """
    def __init__(self):
        super().__init__()

    def forward(self, disease_atts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_atts: [B, D, H, W] — raw attention maps (sigmoid or softmax output)
        Returns:
            scalar: mean normalized spatial entropy
        """
        B, D, H, W = disease_atts.shape
        eps = 1e-8

        # Normalize each map to a probability distribution (sum over spatial = 1)
        flat = disease_atts.reshape(B, D, -1)              # [B, D, H*W]
        denom = flat.sum(dim=2, keepdim=True).clamp(min=eps)
        p = flat / denom                                    # [B, D, H*W]

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -(p * torch.log(p + eps)).sum(dim=2)     # [B, D]

        # Normalize by max possible entropy so loss is in [0, 1]
        max_entropy = np.log(H * W)
        return (entropy / max_entropy).mean()


class DiversityLoss(nn.Module):
    """
    Gram-matrix diversity loss: penalizes disease maps attending identical regions.

    Computes pairwise cosine similarity of flattened attention maps.
    L_div = mean of upper-triangle similarities.

    Note: Some diseases DO share anatomy (e.g. Edema + Effusion → both in lungs).
    Use a small lambda (e.g. 0.001) and monitor that maps don't diverge artificially.
    """
    def __init__(self):
        super().__init__()

    def forward(self, disease_atts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_atts: [B, D, H, W]
        Returns:
            scalar: mean pairwise cosine similarity (minimize to increase diversity)
        """
        B, D, H, W = disease_atts.shape
        eps = 1e-8

        flat = disease_atts.reshape(B, D, -1)       # [B, D, H*W]
        # L2 normalize each disease map
        norms = flat.norm(dim=2, keepdim=True).clamp(min=eps)
        flat_norm = flat / norms                     # [B, D, H*W]

        # Gram matrix: [B, D, D] — pairwise cosine similarities
        gram = torch.bmm(flat_norm, flat_norm.transpose(1, 2))   # [B, D, D]

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(D, D, device=gram.device), diagonal=1).bool()
        similarities = gram[:, mask]    # [B, D*(D-1)/2]

        return similarities.mean()


class RegionalLoss(nn.Module):
    """
    Combined loss: ASL + optional concentration + optional diversity.
    All components logged independently.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cls_loss     = AsymmetricLoss(
            gamma_neg=cfg.get("asl_gamma_neg", 4),
            gamma_pos=cfg.get("asl_gamma_pos", 0),
            clip=cfg.get("asl_clip", 0.05),
        )
        self.lambda_entropy   = cfg.get("lambda_entropy", 0.0)
        self.lambda_diversity = cfg.get("lambda_diversity", 0.0)

        if self.lambda_entropy > 0:
            self.concentration = AttentionConcentrationLoss()
        if self.lambda_diversity > 0:
            self.diversity = DiversityLoss()

    def forward(self, logits, targets, disease_atts=None):
        """
        Returns:
            total_loss: scalar
            components: dict of individual loss values (for logging)
        """
        components = {}
        loss = self.cls_loss(logits, targets)
        components["loss_cls"] = loss.item()

        if self.lambda_entropy > 0 and disease_atts is not None:
            l_ent = self.concentration(disease_atts)
            loss = loss + self.lambda_entropy * l_ent
            components["loss_entropy"] = l_ent.item()

        if self.lambda_diversity > 0 and disease_atts is not None:
            l_div = self.diversity(disease_atts)
            loss = loss + self.lambda_diversity * l_div
            components["loss_diversity"] = l_div.item()

        components["loss_total"] = loss.item()
        return loss, components


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def build_model_from_config(cfg: dict, device: torch.device) -> RegionalAttentionXRayModel:
    """Load Phase 1 checkpoint and wrap it in RegionalAttentionXRayModel."""
    backbone_name = cfg["backbone"]
    ckpt_path = PROJECT_ROOT / "checkpoints" / cfg.get(
        "base_checkpoint", f"best_classifier_{backbone_name}.pth"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {ckpt_path}\n"
            "Run train_phase1_classifier.py first."
        )

    log.info(f"Loading Phase 1 checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()

    model = RegionalAttentionXRayModel(base_model, backbone_name, cfg).to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable: {trainable:,}  Frozen: {frozen:,}")

    return model


def verify_shapes(model: RegionalAttentionXRayModel, device: torch.device, cfg: dict):
    """Forward pass smoke test — verify all tensor shapes."""
    log.info("--- Shape verification ---")
    model.eval()
    dummy = torch.zeros(2, 3, 224, 224, device=device)
    dummy_mask = torch.ones(2, 1, 224, 224, device=device) * 0.5  # soft mask

    with torch.no_grad():
        if cfg.get("anatomy_mode", "none") != "none":
            logits, atts = model(dummy, mask=dummy_mask, return_attention=True)
        else:
            logits, atts = model(dummy, return_attention=True)

    att_res = cfg.get("att_resolution", 14)
    log.info(f"  logits:       {tuple(logits.shape)}  (expected [2, 14])")
    log.info(f"  disease_maps: {tuple(atts.shape)}  (expected [2, 14, 14, 14])")

    assert logits.shape == (2, 14), f"Logits shape mismatch: {logits.shape}"
    assert atts.shape == (2, 14, 14, 14), f"Attention map shape mismatch: {atts.shape}"
    log.info("  Shape verification PASSED")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg, scaler=None):
    model.train()
    total_comps = {}
    n_batches = 0
    t0 = time.time()

    need_atts = (cfg.get("lambda_entropy", 0) > 0 or cfg.get("lambda_diversity", 0) > 0)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            if need_atts:
                logits, disease_atts = model(images, return_attention=True)
                loss, comps = criterion(logits, labels, disease_atts)
            else:
                logits = model(images)
                loss, comps = criterion(logits, labels, None)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for k, v in comps.items():
            total_comps[k] = total_comps.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % 30 == 0:
            elapsed = time.time() - t0
            log.info(f"  Epoch {epoch:2d} | batch {batch_idx:4d} | "
                     f"loss={comps['loss_total']:.4f} | {elapsed:.0f}s")

    torch.cuda.empty_cache()
    return {k: v / n_batches for k, v in total_comps.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n = 0
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss, comps = criterion(logits, labels, None)
                
            total_loss += comps["loss_cls"]
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(labels.cpu())
            n += 1

            if batch_idx % 200 == 0:
                log.info(f"  Val batch {batch_idx:4d}/{total_batches} | val_loss={comps['loss_cls']:.4f}")

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    per_cls = []
    for i in range(NUM_CLASSES):
        try:
            per_cls.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            per_cls.append(0.5)

    macro_auroc = float(np.mean(per_cls))
    torch.cuda.empty_cache()
    return total_loss / max(n, 1), macro_auroc, per_cls


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Experiment: {cfg.get('experiment', 'unnamed')}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    exp_name = cfg.get("experiment", f"phase3_{cfg['backbone']}")
    save_path = CHECKPOINT_DIR / f"{exp_name}.pth"

    # ── File logging ──────────────────────────────────────────────────────────
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_path = CHECKPOINT_DIR / f"{exp_name}.log"
    fh = FlushFileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    log.info(f"Logging to: {log_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model_from_config(cfg, device)
    verify_shapes(model, device, cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    full_train = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("train"))
    full_val   = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))

    n = len(full_train)
    n_test, n_val = int(n * 0.20), int(n * 0.10)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(42)
    tr_idx, val_idx, _ = random_split(range(n), [n_train, n_val, n_test], generator=gen)

    # Subset for smoke test
    if cfg.get("smoke_test_n"):
        sn = cfg["smoke_test_n"]
        tr_idx  = list(tr_idx.indices)[:sn]
        val_idx = list(val_idx.indices)[:sn//5]
    else:
        tr_idx  = tr_idx.indices
        val_idx = val_idx.indices

    bs = cfg.get("batch_size", 8)
    train_loader = DataLoader(Subset(full_train, tr_idx), batch_size=bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(full_val, val_idx),  batch_size=bs, shuffle=False, num_workers=0)

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    criterion = RegionalLoss(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 20), eta_min=1e-6
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ChestXRay14_Phase3_Regional")

    best_auroc = 0.0
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", True) and device.type == "cuda" else None

    with mlflow.start_run(run_name=exp_name):
        mlflow.log_params(cfg)

        for epoch in range(1, cfg.get("epochs", 20) + 1):
            t0 = time.time()
            train_comps = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg, scaler)
            val_loss, macro_auroc, per_cls = validate(model, val_loader, criterion, device, cfg)
            scheduler.step()

            elapsed = time.time() - t0
            log.info(f"Epoch {epoch:3d}/{cfg.get('epochs',20)} | "
                     f"val_loss={val_loss:.4f} | val_AUROC={macro_auroc:.4f} | {elapsed:.0f}s")

            # Per-class AUROC
            for cls, auc in zip(CLASSES, per_cls):
                log.info(f"    {cls:<22s}: {auc:.4f}")

            metrics = {"val_loss": val_loss, "val_macro_auroc": macro_auroc, **train_comps}
            metrics.update({f"auroc_{c}": a for c, a in zip(CLASSES, per_cls)})
            mlflow.log_metrics(metrics, step=epoch)

            if macro_auroc > best_auroc:
                best_auroc = macro_auroc
                torch.save({
                    "epoch":      epoch,
                    "backbone":   cfg["backbone"],
                    "experiment": exp_name,
                    "cfg":        cfg,
                    "model_state": model.state_dict(),
                    "macro_auroc": macro_auroc,
                    "classes":    CLASSES,
                }, save_path)
                log.info(f"  >> New best AUROC={best_auroc:.4f} saved to {save_path.name}")

        log.info(f"Done. Best Val AUROC: {best_auroc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default=None,
                        help="Path to JSON config file (configs/phase3/<name>.json)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test (100 samples, 2 epochs)")
    parser.add_argument("--backbone",   type=str, default="efficientnet_b1",
                        choices=list(BLOCK5_CHANNELS.keys()))
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    elif args.smoke_test:
        # Quick smoke test to verify everything works
        cfg = {
            "experiment":     f"smoke_test_{args.backbone}",
            "backbone":       args.backbone,
            "att_resolution": 14,
            "use_multiscale": True,
            "feat_dim":       256,
            "use_channel_att": True,
            "use_spatial_att": True,
            "anatomy_mode":   "none",
            "att_norm":       "sigmoid",
            "use_global_res": True,
            "use_b5_fusion":  False,
            "att_hidden":     128,
            "lambda_entropy": 0.0,
            "lambda_diversity": 0.0,
            "asl_gamma_neg":  4,
            "asl_gamma_pos":  0,
            "asl_clip":       0.05,
            "batch_size":     8,
            "epochs":         2,
            "lr":             1e-4,
            "weight_decay":   1e-4,
            "seed":           42,
            "smoke_test_n":   80,
        }
        log.info("Running smoke test with 80 samples, 2 epochs...")
    else:
        parser.print_help()
        sys.exit(1)

    main(cfg)
