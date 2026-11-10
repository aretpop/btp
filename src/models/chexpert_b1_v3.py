"""
src/models/chexpert_b1_v3.py

CheXpertV3Model — backbone-agnostic, USLFA-style architecture.

Key changes vs v2:
  - BackboneAdapter interface: no hardcoded channel counts or spatial dims
  - LabelAdaptiveHead: learnable per-disease embeddings + residual adapter + gate
  - Focus loss: normalized by ln(H*W) so weight is interpretable
  - Diversity loss: cooc_masked mode masks anatomically-overlapping pairs
  - recipe flag: 'v2_legacy' preserves old behavior; 'uslfa_v3' enables all new features
  - Per-class attention diagnostics logged every N steps
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_adapter import build_backbone, BackboneAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config — both recipes
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict = {
    # Recipe selector
    "recipe": "uslfa_v3",          # "v2_legacy" | "uslfa_v3"

    # Backbone
    "backbone": {
        "name": "efficientnet_b1",
        "pretrained": True,
        "mid_tap": 5,
        "deep_tap": 8,
    },
    "input_resolution": 224,
    "max_params": 20_000_000,       # assert total params ≤ this (B1 backbone ~6.5M + heads ~10M)

    # Feature head
    "feat_dim": 256,
    "num_classes": 14,
    "fuse_block5": True,            # lateral conv from mid tap into DAG
    "attn_pool_mode": "norm_avg",   # "sum" | "norm_avg"
    "disable_dilation3": False,

    # Attention normalization
    "att_norm": "spatial_softmax",  # "sigmoid" | "spatial_softmax" | "temp_sigmoid"
    "attn_temperature": 0.5,

    # Focus loss
    "attn_regularizer_weight": 0.05,
    "focus_normalize": True,        # divide entropy by ln(H*W) → [0,1]
    # Per-class focus weights (list of 14, default all 1.0 — set lower for diffuse)
    "focus_class_weights": None,    # None = all equal

    # Diversity loss
    "diversity_reg_weight": 0.05,
    "div_mode": "cooc_masked",      # "all_pairs" | "cooc_masked"
    "div_cooc_threshold": 0.1,      # Jaccard below this → penalize pair
    # Pairs always excluded from diversity penalty (indices into CHEXPERT_14_LABELS)
    "div_exclude_indices": [0, 13], # No Finding (0) and Support Devices (13)

    # Classification head
    "head": {
        "type": "label_adaptive",   # "linear14" | "label_adaptive"
        "hidden": 256,
        "emb_dim": 64,
        "use_adapter": True,
        "use_gate": True,
        "gate_init_bias": 2.0,      # sigmoid(2.0)≈0.88, starts gate near 1
    },

    # Attention diagnostics
    "log_attn_every_n_steps": 100,  # 0 = disabled
}

# v2_legacy overrides (applied on top of DEFAULT_CFG when recipe==v2_legacy)
_V2_LEGACY_OVERRIDES: Dict = {
    "fuse_block5": True,
    "att_norm": "spatial_softmax",
    "attn_temperature": 0.5,
    "attn_regularizer_weight": 0.05,
    "focus_normalize": False,       # v2 didn't normalize
    "focus_class_weights": None,
    "diversity_reg_weight": 0.05,
    "div_mode": "all_pairs",        # v2 used all pairs
    "head": {
        "type": "linear14",
        "hidden": 256,
        "emb_dim": 64,
        "use_adapter": False,
        "use_gate": False,
    },
}


def _build_cfg(user_cfg: Optional[Dict]) -> Dict:
    """Merge user_cfg onto DEFAULT_CFG, then apply recipe overrides."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CFG)
    if user_cfg:
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    # Apply legacy recipe overrides
    if cfg.get("recipe") == "v2_legacy":
        for k, v in _V2_LEGACY_OVERRIDES.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Multi-Scale Feature Block (unchanged from v2)
# ---------------------------------------------------------------------------

class MultiScaleFeatureBlock(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int = 256,
                 disable_dilation3: bool = False):
        super().__init__()
        self.disable_dilation3 = disable_dilation3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )
        if not disable_dilation3:
            self.branch4 = nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, 3, padding=3, dilation=3, bias=False),
                nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            )
            concat_ch = feat_dim * 4
        else:
            self.branch4 = None
            concat_ch = feat_dim * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_ch, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch1(x), self.branch2(x), self.branch3(x)]
        if not self.disable_dilation3:
            branches.append(self.branch4(x))
        return self.fuse(torch.cat(branches, dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * gate


# ---------------------------------------------------------------------------
# Disease Attention Generator (DAG)
# ---------------------------------------------------------------------------

class DiseaseAttentionGenerator(nn.Module):
    """
    Per-disease spatial attention maps.
    Supports att_norm: 'spatial_softmax' | 'sigmoid' | 'temp_sigmoid'.
    """

    def __init__(self, in_channels: int, num_classes: int = 14,
                 att_norm: str = "spatial_softmax", temperature: float = 0.5):
        super().__init__()
        self.att_norm = att_norm
        self.temperature = max(1e-4, float(temperature))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention maps [B, C, H, W]."""
        maps = self.net(x)
        if self.att_norm == "spatial_softmax":
            B, C, H, W = maps.shape
            maps = (maps.view(B, C, -1) / self.temperature)
            maps = torch.softmax(maps, dim=-1).view(B, C, H, W)
        elif self.att_norm == "temp_sigmoid":
            maps = torch.sigmoid(maps / self.temperature)
        else:   # sigmoid
            maps = torch.sigmoid(maps)
        return maps


# ---------------------------------------------------------------------------
# Label-Adaptive Head (Section 3 of prompt)
# ---------------------------------------------------------------------------

class LabelAdaptiveHead(nn.Module):
    """
    USLFA-style label-adaptive classification head.

    Input: f [B, C, d] — per-disease pooled features.
    Output: logits [B, C].

    Components (all behind flags):
      - Residual bottleneck adapter (W1, W2): identity at init (W2 zeroed)
      - Learnable per-disease gate via label embeddings q_c → W_gamma → sigmoid
      - Final: (f * cls_w).sum(-1) + cls_b

    Ablation flags: use_adapter, use_gate.
    """

    def __init__(
        self,
        d: int = 256,
        hidden: int = 256,
        C: int = 14,
        emb_dim: int = 64,
        use_adapter: bool = True,
        use_gate: bool = True,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.use_adapter = use_adapter
        self.use_gate = use_gate
        self.C = C

        # Residual bottleneck adapter
        if use_adapter:
            self.W1 = nn.Linear(d, hidden)
            self.W2 = nn.Linear(hidden, d)
            # Zero-init W2 → adapter starts as identity
            nn.init.zeros_(self.W2.weight)
            nn.init.zeros_(self.W2.bias)
            self.act = nn.GELU()

        # Label embedding gate
        if use_gate:
            self.label_emb = nn.Embedding(C, emb_dim)
            self.W_gamma = nn.Linear(emb_dim, d, bias=True)
            # Init gate bias so sigmoid ≈ gate_init_bias value → starts near 1
            nn.init.zeros_(self.W_gamma.weight)
            nn.init.constant_(self.W_gamma.bias, gate_init_bias)

        # Final per-class linear
        self.cls_w = nn.Parameter(torch.randn(C, d) * 0.01)
        self.cls_b = nn.Parameter(torch.zeros(C))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: [B, C, d]  per-disease pooled features
        Returns logits: [B, C]
        """
        if self.use_adapter:
            f = f + self.W2(self.act(self.W1(f)))   # residual adapter

        if self.use_gate:
            # label_emb.weight: [C, emb_dim] → W_gamma → [C, d] → sigmoid
            gamma = torch.sigmoid(self.W_gamma(self.label_emb.weight))  # [C, d]
            f = f * gamma.unsqueeze(0)   # [B, C, d]

        # cls_w: [C, d], f: [B, C, d]
        logits = (f * self.cls_w.unsqueeze(0)).sum(-1) + self.cls_b.unsqueeze(0)
        return logits   # [B, C]


class Linear14Head(nn.Module):
    """Legacy v2 head: 14 independent Linear(d, 1)."""

    def __init__(self, d: int = 256, C: int = 14):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(d, 1) for _ in range(C)]
        )
        self.C = C

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: [B, C, d] → logits [B, C]"""
        return torch.cat(
            [self.classifiers[i](f[:, i, :]) for i in range(self.C)],
            dim=1,
        )


# ---------------------------------------------------------------------------
# Main V3 Model
# ---------------------------------------------------------------------------

class CheXpertV3Model(nn.Module):
    """
    Backbone-agnostic CheXpert v3 model.

    Supports:
      recipe: 'uslfa_v3' (default) or 'v2_legacy'
      backbone: any BackboneAdapter (EfficientNet-B1 by default)
      head: 'label_adaptive' or 'linear14'
      Attention regularization with focus_normalize and cooc_masked diversity
    """

    arch_version: str = "b1v3"

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        self.cfg = _build_cfg(cfg)
        c = self.cfg   # shorthand

        self.recipe = c.get("recipe", "uslfa_v3")
        self.input_resolution: int = c["input_resolution"]
        self.feat_dim: int = c["feat_dim"]
        self.num_classes: int = c["num_classes"]
        self.fuse_block5: bool = c["fuse_block5"]
        self.attn_pool_mode: str = c["attn_pool_mode"]
        self.disable_dilation3: bool = c["disable_dilation3"]
        self.attn_reg_weight: float = c.get("attn_regularizer_weight", 0.05)
        self.att_norm: str = c.get("att_norm", "spatial_softmax")
        self.attn_temperature: float = c.get("attn_temperature", 0.5)
        self.diversity_reg_weight: float = c.get("diversity_reg_weight", 0.05)
        self.focus_normalize: bool = c.get("focus_normalize", True)
        self.div_mode: str = c.get("div_mode", "cooc_masked")
        self.div_cooc_threshold: float = c.get("div_cooc_threshold", 0.1)
        self.div_exclude_indices: List[int] = c.get("div_exclude_indices", [0, 13])
        self.log_attn_every_n_steps: int = c.get("log_attn_every_n_steps", 100)
        self._step_count: int = 0

        # Optional per-class focus weights [C]
        focus_w = c.get("focus_class_weights", None)
        if focus_w is not None:
            self.register_buffer(
                "focus_class_weights",
                torch.tensor(focus_w, dtype=torch.float32)
            )
        else:
            self.focus_class_weights = None

        # Optional co-occurrence mask for diversity loss [C, C]
        self.cooc_mask: Optional[torch.Tensor] = None   # set externally via set_cooc_mask()

        # ── Backbone ──────────────────────────────────────────────────────
        bb_cfg = c.get("backbone", {})
        self.backbone: BackboneAdapter = build_backbone(
            name=bb_cfg.get("name", "efficientnet_b1"),
            pretrained=bb_cfg.get("pretrained", True),
            input_resolution=self.input_resolution,
            mid_tap=bb_cfg.get("mid_tap", 5),
            deep_tap=bb_cfg.get("deep_tap", 8),
        )
        mid_ch = self.backbone.mid_channels
        deep_ch = self.backbone.deep_channels
        self.H_mid = self.backbone.mid_h
        self.W_mid = self.backbone.mid_w
        self.H_deep = self.backbone.deep_h
        self.W_deep = self.backbone.deep_w

        # ── Deep feature processing ───────────────────────────────────────
        self.msfb = MultiScaleFeatureBlock(deep_ch, self.feat_dim, self.disable_dilation3)
        self.channel_attn = ChannelAttention(self.feat_dim, reduction=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # ── FPN fusion (mid tap → DAG input) ─────────────────────────────
        if self.fuse_block5:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(mid_ch, self.feat_dim, 1, bias=False),
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
            dag_in_ch = mid_ch

        # ── Disease Attention Generator ───────────────────────────────────
        self.dag = DiseaseAttentionGenerator(
            dag_in_ch, self.num_classes,
            att_norm=self.att_norm,
            temperature=self.attn_temperature,
        )

        # ── Classification Head ───────────────────────────────────────────
        head_cfg = c.get("head", {})
        head_type = head_cfg.get("type", "label_adaptive")
        if head_type == "label_adaptive":
            self.head = LabelAdaptiveHead(
                d=self.feat_dim,
                hidden=head_cfg.get("hidden", 256),
                C=self.num_classes,
                emb_dim=head_cfg.get("emb_dim", 64),
                use_adapter=head_cfg.get("use_adapter", True),
                use_gate=head_cfg.get("use_gate", True),
                gate_init_bias=head_cfg.get("gate_init_bias", 2.0),
            )
        else:   # linear14
            self.head = Linear14Head(d=self.feat_dim, C=self.num_classes)

        # ── Parameter count logging ───────────────────────────────────────
        self._log_param_counts(c.get("max_params", 12_000_000))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_param_counts(self, max_params: int):
        def n_params(m):
            return sum(p.numel() for p in m.parameters())

        counts = {
            "backbone": n_params(self.backbone),
            "msfb":     n_params(self.msfb),
            "ch_attn":  n_params(self.channel_attn),
            "sp_attn":  n_params(self.spatial_attn),
            "dag":      n_params(self.dag),
            "head":     n_params(self.head),
        }
        total = sum(counts.values())
        logger.info("Parameter counts (M):")
        for k, v in counts.items():
            logger.info("  %-12s  %.3fM", k, v / 1e6)
        logger.info("  %-12s  %.3fM  [limit=%.1fM]", "TOTAL", total / 1e6, max_params / 1e6)
        if total > max_params:
            logger.warning(
                "WARN: Total params %.3fM exceeds max_params %.3fM",
                total / 1e6, max_params / 1e6,
            )

    def set_cooc_mask(self, jaccard: "np.ndarray", threshold: Optional[float] = None):
        """
        Set the co-occurrence penalty mask from a Jaccard matrix.
        Called externally once training data is loaded.

        Pairs with jaccard >= threshold are NOT penalized (they co-occur often
        in real anatomy so pushing them apart would hurt).
        Pairs involving div_exclude_indices are always excluded.
        """
        import numpy as np
        thr = threshold if threshold is not None else self.div_cooc_threshold
        C = self.num_classes
        mask = np.zeros((C, C), dtype=float)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                if i in self.div_exclude_indices or j in self.div_exclude_indices:
                    continue
                if jaccard[i, j] < thr:
                    mask[i, j] = 1.0
        self.cooc_mask = torch.tensor(mask, dtype=torch.float32)
        n_pairs = int(mask.sum())
        logger.info(
            "Co-occ mask set: %d pairs penalized (Jaccard < %.2f), "
            "%d pairs excluded (high co-occ or excluded indices)",
            n_pairs, thr, C * (C - 1) - n_pairs,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            logits [B, num_classes], attn_reg_loss scalar
        """
        B = x.size(0)
        self._step_count += 1

        # 1. Backbone
        feats = self.backbone(x)
        f_mid  = feats["mid"]   # [B, C_mid, H_mid, W_mid]
        f_deep = feats["deep"]  # [B, C_deep, H_deep, W_deep]

        # 2. Deep feature processing
        deep = self.msfb(f_deep)
        deep = self.channel_attn(deep)
        deep = self.spatial_attn(deep)   # [B, feat_dim, H_deep, W_deep]

        # 3. Upsample deep to mid resolution
        deep_up = F.interpolate(
            deep, size=(self.H_mid, self.W_mid),
            mode="bilinear", align_corners=False,
        )   # [B, feat_dim, H_mid, W_mid]

        # 4. FPN fusion with mid tap
        if self.fuse_block5 and self.lateral_conv is not None:
            lateral = self.lateral_conv(f_mid)
            attn_input = self.fuse_bn_relu(deep_up + lateral)
        else:
            attn_input = f_mid

        # 5. Disease attention maps [B, C, H_mid, W_mid]
        d_maps = self.dag(attn_input)

        H, W = self.H_mid, self.W_mid
        HW = H * W
        d_maps_flat = d_maps.view(B, self.num_classes, HW)   # [B, C, H*W]

        # 6. Attention regularization
        attn_reg_loss = torch.tensor(0.0, device=x.device)

        # 6a. Focus loss (spatial entropy penalty)
        if self.attn_reg_weight > 0.0:
            # For spatial_softmax maps, entropy = -sum(p * log(p))
            # maps are already probabilities (sum to 1)
            if self.att_norm == "spatial_softmax":
                log_p = torch.log(d_maps_flat + 1e-8)
                entropy = -(d_maps_flat * log_p).sum(dim=-1)  # [B, C]
            else:
                # sigmoid maps: use L1 sparsity fallback
                entropy = d_maps_flat.mean(dim=-1)            # [B, C]

            if self.focus_normalize and self.att_norm == "spatial_softmax":
                # Normalize by ln(H*W) so entropy ∈ [0, 1]
                entropy = entropy / math.log(HW + 1e-8)

            if self.focus_class_weights is not None:
                # Per-class weighting: [C] broadcast
                entropy = entropy * self.focus_class_weights.unsqueeze(0)

            focus_loss = entropy.mean()
            attn_reg_loss = attn_reg_loss + self.attn_reg_weight * focus_loss

        # 6b. Diversity loss (cross-disease Gram penalty)
        if self.diversity_reg_weight > 0.0 and self.num_classes > 1:
            d_norm = d_maps_flat / (d_maps_flat.norm(dim=-1, keepdim=True) + 1e-6)
            gram = torch.bmm(d_norm, d_norm.transpose(1, 2))  # [B, C, C]
            eye = torch.eye(self.num_classes, device=x.device).unsqueeze(0)
            off_diag = gram * (1.0 - eye)   # [B, C, C]

            if self.div_mode == "cooc_masked" and self.cooc_mask is not None:
                # Penalize only anatomically-independent pairs
                cmask = self.cooc_mask.to(x.device).unsqueeze(0)  # [1, C, C]
                off_diag = off_diag * cmask
                denom = cmask.sum().clamp(min=1.0)
                div_loss = (off_diag ** 2).sum(dim=[1, 2]).mean() / denom
            else:
                # all_pairs (v2 legacy)
                denom = float(self.num_classes * (self.num_classes - 1))
                div_loss = (off_diag ** 2).sum(dim=[1, 2]).mean() / denom

            attn_reg_loss = attn_reg_loss + self.diversity_reg_weight * div_loss

        # 6c. Per-class attention diagnostics (every N steps)
        if (self.log_attn_every_n_steps > 0 and
                self._step_count % self.log_attn_every_n_steps == 0):
            self._log_attn_stats(d_maps_flat.detach())

        # 7. Attention-weighted pooling
        feat_flat = deep_up.view(B, self.feat_dim, HW).permute(0, 2, 1)  # [B, H*W, feat_dim]
        if self.attn_pool_mode == "norm_avg":
            denom = d_maps_flat.sum(dim=-1, keepdim=True) + 1e-6
            z = torch.bmm(d_maps_flat, feat_flat) / denom   # [B, C, feat_dim]
        else:   # sum
            z = torch.bmm(d_maps_flat, feat_flat) / HW      # [B, C, feat_dim]

        # 8. Classification head
        logits = self.head(z)   # [B, C]

        return logits, attn_reg_loss

    # ------------------------------------------------------------------
    # Attention diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_attn_stats(self, d_maps_flat: torch.Tensor):
        """
        Log per-class mean entropy, peak value, area above 50% of peak.
        d_maps_flat: [B, C, H*W]
        """
        B, C, HW = d_maps_flat.shape
        # Mean over batch
        mean_map = d_maps_flat.mean(0)   # [C, H*W]

        for c_idx in range(C):
            p = mean_map[c_idx]   # [H*W]
            if self.att_norm == "spatial_softmax":
                entropy = -(p * torch.log(p + 1e-8)).sum().item()
                if self.focus_normalize:
                    entropy /= math.log(HW + 1e-8)
            else:
                entropy = p.mean().item()
            peak = p.max().item()
            area = (p >= 0.5 * peak).float().sum().item() / HW
            logger.debug(
                "  Attn step=%d class[%2d] entropy=%.4f peak=%.4f area@50pct=%.4f",
                self._step_count, c_idx, entropy, peak, area,
            )

    # ------------------------------------------------------------------
    # Stage A/B helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Stage A: freeze backbone."""
        self.backbone.freeze()
        logger.info("Backbone frozen (Stage A).")

    def unfreeze_for_stage_b(
        self,
        mode: str = "last4",
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> List[Dict]:
        """
        Stage B: partially unfreeze backbone.
        mode: 'last4' | 'full'
        Returns AdamW param groups.
        """
        if mode == "full":
            self.backbone.unfreeze_all()
            backbone_params = list(self.backbone.parameters())
        else:   # last4
            self.backbone.unfreeze_last_n_blocks(n=4)
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        backbone_ids = {id(p) for p in backbone_params}
        head_params = [
            p for p in self.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]
        logger.info(
            "Stage B (%s): backbone_params=%d  head_params=%d",
            mode, len(backbone_params), len(head_params),
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_checkpoint_compat(
        cls, path: str, cfg: Optional[Dict] = None
    ) -> "CheXpertV3Model":
        """
        Load checkpoint with strict=False (graceful mismatch handling).
        Compatible with v2 checkpoints when recipe=v2_legacy.
        """
        model = cls(cfg)
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Checkpoint missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("Checkpoint unexpected keys (%d): %s", len(unexpected), unexpected[:10])
        logger.info("Loaded checkpoint: %s (arch=%s, recipe=%s)", path, model.arch_version, model.recipe)
        return model
