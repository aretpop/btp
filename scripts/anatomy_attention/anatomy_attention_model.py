"""
anatomy_attention_model.py
===========================
Main model — assembles M0 through M3 from individual modules.

Ablation stages controlled by config dict:
  M0: backbone + simple GAP + Linear (original Phase 1 behaviour)
  M1: + MultiScaleBlock + ChannelAttention + SpatialAttention
  M2: M1 + AnatomyEncoder + AnatomyFusion
  M3: M2 + DiseaseAttentionGenerator + DiseaseHead

Forward API:
  logits = model(x)                            → [B, 14]
  logits, attn = model(x, return_attention=True)
  logits, attn = model(x, anatomy_masks=masks, return_attention=True)

attention dict keys:
  "multiscale_features"    [B, C, H, W]
  "channel_attention"      [B, C, 1, 1]   (the A_ch weights)
  "spatial_attention"      [B, 1, H, W]   (the A_sp weights)
  "anatomy_features"       [B, C, H, W]   (if use_anatomy)
  "anatomy_gate"           [B, C, H, W]   (if use_anatomy)
  "disease_attention"      [B, 14, H, W]  (if use_disease_attention)
"""

import sys
import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing siblings
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from backbone_loader import FrozenBackboneWrapper, load_frozen_backbone
from attention_modules import ChannelAttention, SpatialAttention, MultiScaleBlock
from anatomy_encoder import AnatomyEncoder, AnatomyFusion
from disease_attention import DiseaseAttentionGenerator, DiseaseHead

log = logging.getLogger(__name__)

# Default configuration — every flag can be toggled independently
DEFAULT_CONFIG = {
    "backbone": "efficientnet_b1",
    "checkpoint_path": None,          # required
    "freeze_backbone": True,

    # M1 components
    "use_multiscale": True,
    "multiscale_dilations": (1, 2, 3),
    "use_channel_attention": True,
    "channel_reduction": 16,
    "use_spatial_attention": True,
    "spatial_kernel": 7,

    # M2 components
    "use_anatomy": False,
    "anatomy_mid_channels": 32,
    "use_boundary": False,

    # M3 components
    "use_disease_attention": True,
    "use_anatomy_conditioning": False,   # condition disease att on anatomy
    "use_global_local": False,           # global+local fusion in DiseaseHead
    "disease_att_hidden": 128,

    # Loss
    "lambda_anatomy": 0.0,
    "lambda_diversity": 0.0,
    "asl_gamma_neg": 4,
    "asl_gamma_pos": 0,
    "asl_clip": 0.05,

    "num_classes": 14,
}


class AnatomyDiseaseAttentionModel(nn.Module):
    """
    Anatomy-conditioned, disease-specific attention model.

    All modules are conditional on the config dict so that a single class
    supports M0 through M3 via flag toggling.
    """

    def __init__(self, config: Dict):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **config}
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]

        assert cfg["checkpoint_path"] is not None, \
            "config must include 'checkpoint_path' pointing to a Phase 1 .pth file."

        # ------------------------------------------------------------------ #
        # 1. Load frozen backbone
        # ------------------------------------------------------------------ #
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_backbone, feature_dim, backbone_name = load_frozen_backbone(
            cfg["checkpoint_path"], device, verify_immutability=True
        )
        self.feature_dim = feature_dim
        self.backbone_name = backbone_name
        self.backbone = FrozenBackboneWrapper(feat_backbone)

        # ------------------------------------------------------------------ #
        # 2. M1 — Multi-scale + Channel + Spatial attention
        # ------------------------------------------------------------------ #
        self.multiscale = (
            MultiScaleBlock(feature_dim, dilations=cfg["multiscale_dilations"])
            if cfg["use_multiscale"] else nn.Identity()
        )
        self.channel_att = (
            ChannelAttention(feature_dim, cfg["channel_reduction"])
            if cfg["use_channel_attention"] else nn.Identity()
        )
        self.spatial_att = (
            SpatialAttention(cfg["spatial_kernel"])
            if cfg["use_spatial_attention"] else nn.Identity()
        )

        # ------------------------------------------------------------------ #
        # 3. M2 — Anatomy encoder + fusion
        # ------------------------------------------------------------------ #
        self.use_anatomy = cfg["use_anatomy"]
        if self.use_anatomy:
            anatomy_in_ch = 3 + (2 if cfg["use_boundary"] else 0)
            self.anatomy_encoder = AnatomyEncoder(
                in_channels=anatomy_in_ch,
                mid_channels=cfg["anatomy_mid_channels"],
                out_channels=feature_dim,
                use_boundary=cfg["use_boundary"],
            )
            self.anatomy_fusion = AnatomyFusion(feature_dim)

        # ------------------------------------------------------------------ #
        # 4. M3 — Disease-specific attention + classifiers
        # ------------------------------------------------------------------ #
        self.use_disease_att = cfg["use_disease_attention"]
        if self.use_disease_att:
            self.disease_att_gen = DiseaseAttentionGenerator(
                feature_channels=feature_dim,
                num_classes=self.num_classes,
                use_anatomy_conditioning=cfg["use_anatomy_conditioning"] and self.use_anatomy,
                hidden_channels=cfg["disease_att_hidden"],
            )
            self.disease_head = DiseaseHead(
                feature_channels=feature_dim,
                num_classes=self.num_classes,
                use_global_local=cfg["use_global_local"],
            )
        else:
            # M0 / M1 / M2 fallback: simple GAP + shared linear
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(feature_dim, self.num_classes)

        log.info(
            f"AnatomyDiseaseAttentionModel built:\n"
            f"  backbone={backbone_name} (frozen={cfg['freeze_backbone']})\n"
            f"  multiscale={cfg['use_multiscale']}, "
            f"channel_att={cfg['use_channel_attention']}, "
            f"spatial_att={cfg['use_spatial_attention']}\n"
            f"  anatomy={self.use_anatomy}, "
            f"disease_att={self.use_disease_att}"
        )

    # ---------------------------------------------------------------------- #
    # Train mode override: backbone stays in eval() permanently              #
    # ---------------------------------------------------------------------- #
    def train(self, mode: bool = True) -> "AnatomyDiseaseAttentionModel":
        super().train(mode)
        self.backbone.eval()  # FrozenBackboneWrapper also enforces this
        return self

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        anatomy_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            x:              [B, 3, 224, 224]  ImageNet-normalised
            anatomy_masks:  Optional dict with keys "left_lung", "right_lung",
                            "heart" (each [B, 1, H_orig, W_orig]).
                            Required when use_anatomy=True.
            return_attention: If True, return (logits, attention_dict).

        Returns:
            logits: [B, 14]  raw logits — use BCEWithLogitsLoss
            attention_dict (optional): dict of intermediate tensors.
        """
        attn: Dict[str, torch.Tensor] = {}

        # ------ Backbone -------------------------------------------------- #
        # F: [B, C, Hf, Wf]  — Hf=Wf=7 for 224×224 input
        F = self.backbone(x)
        B, C, Hf, Wf = F.shape

        # Validate anatomy masks if anatomy is enabled
        if self.use_anatomy:
            if anatomy_masks is None:
                raise ValueError(
                    "use_anatomy=True but anatomy_masks=None. "
                    "Either provide masks or set use_anatomy=False."
                )
            required = ["left_lung", "right_lung", "heart"]
            missing = [k for k in required if k not in anatomy_masks]
            if missing:
                raise ValueError(f"anatomy_masks is missing keys: {missing}")

        # ------ M1: Multi-scale ------------------------------------------- #
        F_ms = self.multiscale(F)          # [B, C, Hf, Wf]
        if return_attention:
            attn["multiscale_features"] = F_ms.detach()

        # ------ M1: Channel attention ------------------------------------- #
        if self.cfg["use_channel_attention"]:
            # Capture A_ch for visualisation
            avg_out = self.channel_att.mlp(self.channel_att.avg_pool(F_ms))
            max_out = self.channel_att.mlp(self.channel_att.max_pool(F_ms))
            A_ch = torch.sigmoid(avg_out + max_out)      # [B, C, 1, 1]
            F_ca = F_ms * A_ch
            if return_attention:
                attn["channel_attention"] = A_ch.detach()
        else:
            F_ca = F_ms

        # ------ M1: Spatial attention ------------------------------------- #
        if self.cfg["use_spatial_attention"]:
            avg_map = F_ca.mean(dim=1, keepdim=True)
            max_map, _ = F_ca.max(dim=1, keepdim=True)
            sp_in = torch.cat([avg_map, max_map], dim=1)
            A_sp = torch.sigmoid(self.spatial_att.conv(sp_in))  # [B, 1, Hf, Wf]
            F_sa = F_ca * A_sp
            if return_attention:
                attn["spatial_attention"] = A_sp.detach()
        else:
            F_sa = F_ca

        # ------ M2: Anatomy encoding + soft fusion ------------------------ #
        anatomy_feats: Optional[torch.Tensor] = None
        if self.use_anatomy and anatomy_masks is not None:
            anatomy_feats = self.anatomy_encoder(anatomy_masks, Hf, Wf)
            if return_attention:
                attn["anatomy_features"] = anatomy_feats.detach()
            F_refined = self.anatomy_fusion(F_sa, anatomy_feats)
        else:
            F_refined = F_sa

        # ------ M3: Disease-specific attention + classification ----------- #
        if self.use_disease_att:
            disease_atts = self.disease_att_gen(F_refined, anatomy_feats)
            if return_attention:
                attn["disease_attention"] = disease_atts.detach()
            logits = self.disease_head(F_refined, disease_atts)
        else:
            # M0/M1/M2 path: simple GAP
            z = self.gap(F_refined).flatten(1)   # [B, C]
            logits = self.classifier(z)           # [B, 14]
            disease_atts = None

        assert logits.shape == (B, self.num_classes), \
            f"Unexpected logits shape: {logits.shape}"

        if return_attention:
            return logits, attn, disease_atts
        return logits, None, disease_atts

    # ---------------------------------------------------------------------- #
    # Parameter report                                                        #
    # ---------------------------------------------------------------------- #
    def param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
