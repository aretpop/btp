"""
src/models/chexpert_b1_v2.py

CheXpertB1V2Model — EfficientNet-B1 backbone with regional attention,
FPN-style Block5 fusion, configurable attention pooling, and Stage A/B
training support.

Architecture matches Phase 3 (train_phase3_regional_v3.py) with extensions.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_CFG: Dict = {
    "backbone_name": "efficientnet_b1",
    "input_resolution": 224,
    "feat_dim": 256,
    "num_classes": 14,
    "fuse_block5": True,
    "attn_pool_mode": "norm_avg",   # "sum" | "norm_avg" | "softmax"
    "disable_dilation3": False,
    "attn_regularizer_weight": 0.0,
    "att_norm": "spatial_softmax",   # "sigmoid" | "spatial_softmax" | "temp_sigmoid"
    "attn_temperature": 0.5,
    "diversity_reg_weight": 0.0,
}

# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class MultiScaleFeatureBlock(nn.Module):
    """
    4-branch (or 3-branch when disable_dilation3=True) multi-scale feature
    extractor.  Each branch is independently BN+ReLU'd, then branches are
    concatenated and compressed back to feat_dim via 1×1 conv + BN + ReLU.
    """

    def __init__(self, in_channels: int, feat_dim: int = 256,
                 disable_dilation3: bool = False):
        super().__init__()
        self.disable_dilation3 = disable_dilation3

        # Branch 1: 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 3×3 conv (pad=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 3×3 dilated conv, dilation=2
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3×3 dilated conv, dilation=3  (optionally disabled)
        if not disable_dilation3:
            self.branch4 = nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, 3, padding=3, dilation=3, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
            )
            concat_ch = feat_dim * 4
        else:
            self.branch4 = None
            concat_ch = feat_dim * 3

        # 1×1 fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_ch, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch1(x), self.branch2(x), self.branch3(x)]
        if not self.disable_dilation3:
            branches.append(self.branch4(x))
        return self.fuse(torch.cat(branches, dim=1))


class ChannelAttention(nn.Module):
    """
    CBAM-style channel attention: avg_pool + max_pool → shared MLP → sigmoid.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = x.mean(dim=[2, 3])              # [B, C]
        mx = x.amax(dim=[2, 3])              # [B, C]
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # [B, C]
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """
    CBAM-style spatial attention: avg+max along channel dim → concat → 7×7 conv → sigmoid.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # [B,1,H,W]
        mx = x.amax(dim=1, keepdim=True)    # [B,1,H,W]
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * gate


class DiseaseAttentionGenerator(nn.Module):
    """
    Generates per-disease attention maps from spatial feature maps.
    Conv3×3(in_ch→128)→BN→ReLU→Conv3×3(128→64)→BN→ReLU→Conv1×1(64→num_classes).
    Supports att_norm = "spatial_softmax" (default), "sigmoid", or "temp_sigmoid".
    """

    def __init__(self, in_channels: int, num_classes: int = 14,
                 att_norm: str = "spatial_softmax", temperature: float = 0.5):
        super().__init__()
        self.att_norm = att_norm
        self.temperature = max(1e-4, float(temperature))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention maps: [B, num_classes, H, W]."""
        maps = self.net(x)
        if self.att_norm == "spatial_softmax":
            B, C, H, W = maps.shape
            maps = maps.view(B, C, -1) / self.temperature
            maps = torch.softmax(maps, dim=-1)
            maps = maps.view(B, C, H, W)
        elif self.att_norm == "temp_sigmoid":
            maps = torch.sigmoid(maps / self.temperature)
        else:  # sigmoid
            maps = torch.sigmoid(maps)
        return maps


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CheXpertB1V2Model(nn.Module):
    """
    EfficientNet-B1 regional attention model for CheXpert multi-label
    classification.

    Key features vs Phase 3:
    - Config-driven (dict), no argparse dependency
    - Dynamic spatial dim derivation (no hardcoded 7/14)
    - FPN-style Block5 lateral fusion
    - Three attention pooling modes: sum / norm_avg / softmax
    - Optional dilation3 disable
    - Attention entropy regularizer
    - Stage A/B training helpers
    - Checkpoint compat loader
    """

    arch_version: str = "b1v2"

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        # Merge with defaults
        full_cfg = {**DEFAULT_CFG}
        if cfg:
            full_cfg.update(cfg)

        self.cfg = full_cfg
        self.input_resolution: int = full_cfg["input_resolution"]
        self.feat_dim: int = full_cfg["feat_dim"]
        self.num_classes: int = full_cfg["num_classes"]
        self.fuse_block5: bool = full_cfg["fuse_block5"]
        self.attn_pool_mode: str = full_cfg["attn_pool_mode"]
        self.disable_dilation3: bool = full_cfg["disable_dilation3"]
        self.attn_reg_weight: float = full_cfg.get("attn_regularizer_weight", 0.0)
        self.att_norm: str = full_cfg.get("att_norm", "spatial_softmax")
        self.attn_temperature: float = full_cfg.get("attn_temperature", 0.5)
        self.diversity_reg_weight: float = full_cfg.get("diversity_reg_weight", 0.0)

        # ------------------------------------------------------------------ #
        # Backbone (EfficientNet-B1, pretrained ImageNet)
        # ------------------------------------------------------------------ #
        backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        # Keep only the feature-extraction blocks; drop avgpool + classifier
        self.backbone = backbone
        self._feature_blocks = backbone.features  # nn.Sequential of 9 blocks (0..8)

        # ------------------------------------------------------------------ #
        # Dynamically derive spatial dims by probing with a dummy tensor
        # ------------------------------------------------------------------ #
        H_high, H_low = self._probe_spatial_dims()
        self.H_high = H_high   # block-5 spatial size (larger)
        self.H_low = H_low     # block-8 spatial size (smaller)
        logger.info(
            "Spatial dims: Block5=%dx%d  Block8=%dx%d",
            H_high, H_high, H_low, H_low,
        )

        # Channel counts (EfficientNet-B1 fixed)
        b5_ch = 112    # Block 5 output channels
        b8_ch = 1280   # Block 8 output channels

        # ------------------------------------------------------------------ #
        # Deep feature processing (Block 8)
        # ------------------------------------------------------------------ #
        self.msfb = MultiScaleFeatureBlock(b8_ch, self.feat_dim, self.disable_dilation3)
        self.channel_attn = ChannelAttention(self.feat_dim, reduction=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # ------------------------------------------------------------------ #
        # FPN-style Block 5 lateral (optional)
        # ------------------------------------------------------------------ #
        if self.fuse_block5:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(b5_ch, self.feat_dim, 1, bias=False),
                nn.BatchNorm2d(self.feat_dim),
                nn.ReLU(inplace=True),
            )
            self.fuse_bn_relu = nn.Sequential(
                nn.BatchNorm2d(self.feat_dim),
                nn.ReLU(inplace=True),
            )
            dag_in_ch = self.feat_dim
        else:
            self.lateral_conv = None
            self.fuse_bn_relu = None
            dag_in_ch = b5_ch

        # ------------------------------------------------------------------ #
        # Disease attention generator
        # ------------------------------------------------------------------ #
        self.dag = DiseaseAttentionGenerator(
            dag_in_ch,
            self.num_classes,
            att_norm=self.att_norm,
            temperature=self.attn_temperature,
        )

        # ------------------------------------------------------------------ #
        # Per-disease classifiers
        # ------------------------------------------------------------------ #
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.feat_dim, 1) for _ in range(self.num_classes)]
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _probe_spatial_dims(self) -> Tuple[int, int]:
        """Pass a dummy input through the backbone to find Block5 and Block8 sizes."""
        res = self.input_resolution
        dummy = torch.zeros(1, 3, res, res)
        with torch.no_grad():
            x = dummy
            for i, block in enumerate(self._feature_blocks):
                x = block(x)
                if i == 5:
                    H_high = x.shape[-1]
                if i == 8:
                    H_low = x.shape[-1]
        return H_high, H_low

    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward x through EfficientNet feature blocks.
        Returns (f5, f8): outputs of block 5 and block 8.
        """
        f5 = None
        f8 = None
        for i, block in enumerate(self._feature_blocks):
            x = block(x)
            if i == 5:
                f5 = x
            if i == 8:
                f8 = x
                break  # block 8 is the last we need
        return f5, f8

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            logits: [B, num_classes]
            attn_reg_loss: scalar (0.0 when attn_regularizer_weight == 0)
        """
        B = x.size(0)

        # 1. Extract backbone features
        f5, f8 = self._extract_features(x)
        # f5: [B, 112, H_high, H_high]
        # f8: [B, 1280, H_low,  H_low]

        # 2. Deep feature processing
        deep = self.msfb(f8)                     # [B, feat_dim, H_low, H_low]
        deep = self.channel_attn(deep)
        deep = self.spatial_attn(deep)           # [B, feat_dim, H_low, H_low]

        # 3. Upsample deep features to H_high × H_high
        deep_up = F.interpolate(
            deep, size=(self.H_high, self.H_high), mode="bilinear", align_corners=False
        )  # [B, feat_dim, H_high, H_high]

        # 4. FPN fusion with Block 5
        if self.fuse_block5:
            lateral = self.lateral_conv(f5)              # [B, feat_dim, H_high, H_high]
            fused = self.fuse_bn_relu(deep_up + lateral) # [B, feat_dim, H_high, H_high]
            attn_input = fused
        else:
            attn_input = f5                              # [B, 112, H_high, H_high]

        # 5. Disease attention maps
        d_maps = self.dag(attn_input)   # [B, num_classes, H_high, H_high]

        # 6. Attention regularizer & Diversity Loss
        H, W = self.H_high, self.H_high
        HW = H * W
        d_maps_flat = d_maps.view(B, self.num_classes, HW)   # [B, C, H*W]
        attn_reg_loss = torch.tensor(0.0, device=x.device)

        # 6a. Spatial focus / sparsity penalty
        if self.attn_reg_weight > 0.0:
            if getattr(self.dag, "att_norm", "sigmoid") == "spatial_softmax":
                # Entropy penalty: low entropy = sharp, focused distribution
                entropy = -(d_maps_flat * torch.log(d_maps_flat + 1e-8)).sum(dim=-1).mean()
                attn_reg_loss = attn_reg_loss + self.attn_reg_weight * entropy
            else:
                # L1 sparsity penalty for sigmoid maps
                l1_sparse = d_maps_flat.mean()
                attn_reg_loss = attn_reg_loss + self.attn_reg_weight * l1_sparse

        # 6b. Cross-disease diversity loss (penalize cross-disease spatial correlation)
        if self.diversity_reg_weight > 0.0 and self.num_classes > 1:
            d_norm = d_maps_flat / (d_maps_flat.norm(dim=-1, keepdim=True) + 1e-6)
            gram = torch.bmm(d_norm, d_norm.transpose(1, 2))  # [B, C, C]
            eye = torch.eye(self.num_classes, device=x.device).unsqueeze(0)
            off_diag = gram * (1.0 - eye)
            div_loss = (off_diag ** 2).sum(dim=[1, 2]).mean() / (self.num_classes * (self.num_classes - 1))
            attn_reg_loss = attn_reg_loss + self.diversity_reg_weight * div_loss

        # 7. Attention-weighted pooling
        feat_flat = deep_up.view(B, self.feat_dim, HW).permute(0, 2, 1)  # [B, H*W, feat_dim]

        if self.attn_pool_mode == "softmax":
            d_maps_flat = torch.softmax(d_maps_flat, dim=-1)
            # z[B,C,feat_dim] = d_maps_flat[B,C,HW] @ feat_flat[B,HW,feat_dim]
            z = torch.bmm(d_maps_flat, feat_flat)              # [B, num_classes, feat_dim]
        elif self.attn_pool_mode == "norm_avg":
            denom = d_maps_flat.sum(dim=-1, keepdim=True) + 1e-6  # [B, C, 1]
            z = torch.bmm(d_maps_flat, feat_flat) / denom         # [B, num_classes, feat_dim]
        else:  # "sum" (old unnormalized)
            z = torch.bmm(d_maps_flat, feat_flat) / HW            # [B, num_classes, feat_dim]

        # 8. Per-disease classification
        logits = torch.cat(
            [self.classifiers[i](z[:, i, :]) for i in range(self.num_classes)],
            dim=1,
        )  # [B, num_classes]

        return logits, attn_reg_loss

    # ---------------------------------------------------------------------- #
    # Stage A/B training helpers
    # ---------------------------------------------------------------------- #

    def freeze_backbone(self):
        """Stage A: freeze backbone (eval mode + no grad)."""
        self.backbone.eval()
        for param in self.backbone.features.parameters():
            param.requires_grad_(False)
        logger.info("Backbone frozen (Stage A).")

    def unfreeze_for_stage_b(
        self,
        mode: str = "last3",
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> List[Dict]:
        """
        Stage B: partially or fully unfreeze backbone.
        Args:
            mode: "last3" (unfreeze features[5:]) or "full" (unfreeze all)
            backbone_lr: LR for backbone params
            head_lr: LR for head params
        Returns:
            AdamW-ready param groups list.
        """
        # Put backbone back in training mode
        self.backbone.train()

        if mode == "full":
            for param in self.backbone.features.parameters():
                param.requires_grad_(True)
            backbone_params = list(self.backbone.features.parameters())
        else:  # "last3"
            # Freeze everything first
            for param in self.backbone.features.parameters():
                param.requires_grad_(False)
            # Unfreeze features[5], [6], [7], [8]  (= "last 4" of 9 blocks)
            backbone_params = []
            for i in range(5, 9):
                block = self.backbone.features[i]
                for param in block.parameters():
                    param.requires_grad_(True)
                backbone_params.extend(block.parameters())

        # Head params = everything except backbone.features
        head_param_ids = {id(p) for p in backbone_params}
        head_params = [
            p for name, p in self.named_parameters()
            if id(p) not in head_param_ids and p.requires_grad
        ]

        logger.info(
            "Stage B (%s): backbone_params=%d  head_params=%d",
            mode, len(list(backbone_params)), len(head_params),
        )

        return [
            {"params": list(backbone_params), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        unfreeze_mode: str = "last3",
    ) -> List[Dict]:
        """Returns AdamW-ready param groups (calls unfreeze_for_stage_b internally)."""
        return self.unfreeze_for_stage_b(
            mode=unfreeze_mode, backbone_lr=backbone_lr, head_lr=head_lr
        )

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers
    # ---------------------------------------------------------------------- #

    @classmethod
    def load_checkpoint_compat(cls, path: str, cfg: Optional[Dict] = None) -> "CheXpertB1V2Model":
        """
        Load a checkpoint with graceful mismatch handling.
        Logs warnings for missing/extra keys; loads with strict=False.
        """
        model = cls(cfg)
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Checkpoint missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("Checkpoint unexpected keys (%d): %s", len(unexpected), unexpected[:10])
        logger.info("Loaded checkpoint from %s (arch_version=%s)", path, model.arch_version)
        return model
