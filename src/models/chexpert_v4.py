"""
src/models/chexpert_v4.py
=========================
Backbone-agnostic CheXpert V4 ("v4_gap") model architecture.

Pipeline:
  deep feature map [B, C_deep, H, W]
  -> MSFB (4 branches x 64ch: 1x1, 3x3, 3x3 dil2, 3x3 dil3; C_deep -> 256)
  -> SE Channel Attention (r=16)
  -> CBAM Spatial Attention (7x7)
  -> Global Average Pooling -> f [B, 256]
  -> LabelAdaptiveHead -> logits [B, 14]

Design:
  - Clean GAP representation (no learned disease attention generator or BMM)
  - No focus loss (L_focus) or diversity loss (L_div)
  - Backbone-agnostic: derives feature map dims dynamically via dummy forward
  - Assert total parameters <= 12M (configured max_params)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_adapter import build_backbone, BackboneAdapter
from src.data.labels import CHEXPERT_14_LABELS

logger = logging.getLogger(__name__)


class MSFBv4(nn.Module):
    """
    Multi-Scale Feature Block for V4.
    4 branches x 64 channels:
      branch 1: 1x1 conv
      branch 2: 3x3 conv (dilation 1)
      branch 3: 3x3 conv (dilation 2)
      branch 4: 3x3 conv (dilation 3) [if msfb_dil3 is True]
    Outputs concatenated channels -> 1x1 fuse conv to feat_dim (256).
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 64,
        out_channels: int = 256,
        msfb_dil3: bool = True,
    ):
        super().__init__()
        self.msfb_dil3 = msfb_dil3
        self.branch_channels = branch_channels

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        if msfb_dil3:
            self.b4 = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 3, padding=3, dilation=3, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            concat_channels = branch_channels * 4
        else:
            self.b4 = None
            concat_channels = branch_channels * 3

        self.fuse = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        if self.b4 is not None:
            b4 = self.b4(x)
            out = torch.cat([b1, b2, b3, b4], dim=1)
        else:
            out = torch.cat([b1, b2, b3], dim=1)
        return self.fuse(out)


class SEChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, in_channels: int = 256, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class CBAMSpatialAttention(nn.Module):
    """CBAM spatial attention with 7x7 conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * gate


class LabelAdaptiveHead(nn.Module):
    """
    LabelAdaptiveHead per Section 4 specification.
    Learnable per-disease embeddings + residual adapter + gate.
    Identity at initialization.
    """

    def __init__(
        self,
        d: int = 256,
        hidden: int = 256,
        C: int = 14,
        emb_dim: int = 64,
        use_adapter: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_adapter = use_adapter
        self.use_gate = use_gate
        self.W1 = nn.Linear(d, hidden)
        self.W2 = nn.Linear(hidden, d)
        # Identity at init: W2=0, b2=0 -> residual W2(act(W1(f))) = 0
        nn.init.zeros_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)
        self.act = nn.GELU()
        self.label_emb = nn.Embedding(C, emb_dim)
        self.W_gamma = nn.Linear(emb_dim, d, bias=False)
        self.cls_w = nn.Parameter(torch.randn(C, d) * 0.01)
        self.cls_b = nn.Parameter(torch.zeros(C))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B, d]
        if self.use_adapter:
            f = f + self.W2(self.act(self.W1(f)))
        f = f[:, None, :].expand(-1, self.cls_w.size(0), -1)  # [B, C, d]
        if self.use_gate:
            gamma = torch.sigmoid(self.W_gamma(self.label_emb.weight))  # [C, d]
            f = f * gamma[None]
        return (f * self.cls_w[None]).sum(-1) + self.cls_b[None]  # [B, C]


class CheXpertV4Model(nn.Module):
    """
    CheXpert V4 ("v4_gap") model wrapper.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        if cfg is None:
            cfg = {}

        # Resolve config settings
        bb_cfg = cfg.get("backbone", {})
        bb_name = bb_cfg.get("name", "efficientnet_b1")
        pretrained = bb_cfg.get("pretrained", True)
        res = cfg.get("input_size", cfg.get("input_resolution", 240))
        mid_tap = bb_cfg.get("mid_tap", 5)
        deep_tap = bb_cfg.get("deep_tap", 8)

        feat_dim = cfg.get("feat_dim", 256)
        num_classes = cfg.get("num_classes", len(CHEXPERT_14_LABELS))
        msfb_dil3 = cfg.get("msfb_dil3", True)
        max_params = cfg.get("max_params", 12_000_000)

        head_cfg = cfg.get("head", {})
        use_adapter = head_cfg.get("use_adapter", True)
        use_gate = head_cfg.get("use_gate", True)
        emb_dim = head_cfg.get("emb_dim", 64)
        hidden = head_cfg.get("hidden", 256)

        # 1. Backbone Adapter
        self.backbone = build_backbone(
            name=bb_name,
            pretrained=pretrained,
            input_resolution=res,
            mid_tap=mid_tap,
            deep_tap=deep_tap,
        )
        deep_channels = self.backbone.deep_channels
        self.input_size = res

        # 2. MSFB (deep_channels -> feat_dim)
        self.msfb = MSFBv4(
            in_channels=deep_channels,
            branch_channels=64,
            out_channels=feat_dim,
            msfb_dil3=msfb_dil3,
        )

        # 3. SE Channel Attention
        self.se = SEChannelAttention(in_channels=feat_dim, reduction=16)

        # 4. CBAM Spatial Attention
        self.cbam = CBAMSpatialAttention(kernel_size=7)

        # 5. LabelAdaptiveHead
        self.head = LabelAdaptiveHead(
            d=feat_dim,
            hidden=hidden,
            C=num_classes,
            emb_dim=emb_dim,
            use_adapter=use_adapter,
            use_gate=use_gate,
        )

        # Audit and verify parameter budget
        self.param_counts = self._audit_params()
        total_p = self.param_counts["total"]
        logger.info(f"CheXpertV4Model initialized ({bb_name} @ {res}px):")
        for k, v in self.param_counts.items():
            logger.info(f"  {k:15s}: {v:10,d} parameters")

        assert total_p <= max_params, (
            f"Model parameter count {total_p:,} exceeds max_params {max_params:,}!"
        )

    def _audit_params(self) -> Dict[str, int]:
        bb_p = sum(p.numel() for p in self.backbone.parameters())
        msfb_p = sum(p.numel() for p in self.msfb.parameters())
        se_p = sum(p.numel() for p in self.se.parameters())
        cbam_p = sum(p.numel() for p in self.cbam.parameters())
        head_p = sum(p.numel() for p in self.head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "backbone": bb_p,
            "msfb": msfb_p,
            "se": se_p,
            "cbam": cbam_p,
            "head": head_p,
            "total": total,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            logits: [B, num_classes]
        """
        bb_out = self.backbone(x)
        deep = bb_out["deep"]  # [B, C_deep, H_feat, W_feat]

        feat = self.msfb(deep)  # [B, 256, H_feat, W_feat]
        feat = self.se(feat)    # [B, 256, H_feat, W_feat]
        feat = self.cbam(feat)  # [B, 256, H_feat, W_feat]

        f = feat.mean(dim=[2, 3])  # Global average pooling -> [B, 256]
        logits = self.head(f)      # [B, 14]
        return logits

    def freeze_stage_a(self):
        """
        Stage A: Backbone frozen AND in eval() mode (BN running stats frozen).
        MSFB, SE, CBAM, and Head remain trainable.
        """
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for m in [self.msfb, self.se, self.cbam, self.head]:
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)
        logger.info("Stage A configuration applied: Backbone frozen in eval(); heads trainable.")

    def unfreeze_stage_b(self, unfreeze_from: int = 5):
        """
        Stage B: Unfreeze backbone from unfreeze_from onwards (e.g. features[5:]).
        BN in frozen blocks stays in eval(); BN in unfrozen blocks trains normally.
        """
        features = self.backbone.features
        num_blocks = len(features)

        if unfreeze_from == 0:
            # Full unfreeze
            self.backbone.train()
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            logger.info("Stage B configuration: full backbone unfrozen.")
        else:
            # Freeze features[:unfreeze_from]
            for i in range(min(unfreeze_from, num_blocks)):
                block = features[i]
                block.eval()
                for p in block.parameters():
                    p.requires_grad_(False)

            # Unfreeze features[unfreeze_from:]
            for i in range(unfreeze_from, num_blocks):
                block = features[i]
                block.train()
                for p in block.parameters():
                    p.requires_grad_(True)

            logger.info(f"Stage B configuration: unfroze backbone features[{unfreeze_from}:].")

        # Heads remain trainable
        for m in [self.msfb, self.se, self.cbam, self.head]:
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)

