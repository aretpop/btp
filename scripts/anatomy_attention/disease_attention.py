"""
disease_attention.py
====================
Disease-specific spatial attention heads (M3 component).

Generates A ∈ [B, 14, H, W] — one independent spatial map per disease.
The shared backbone is NOT duplicated. Only a lightweight 1×1 conv head
is disease-specific.

Design principle (from prompt §14-16):
  - ONE shared backbone
  - ONE shared attention pathway (channel + spatial + anatomy)
  - ONE lightweight 14-channel disease attention generator
  - 14 tiny per-disease classifiers

The disease attention can optionally be conditioned on anatomy features
(§16): disease_attention = DiseaseAttention(refined_features, anatomy_features)

Conditioning strategy:
  concat(refined_F, anatomy_F) → conv → [B, 14, H, W]
  This lets the model learn "given this anatomical context, where should
  I look for disease d?" — without hard-coding any disease-organ mapping.
"""

import torch
import torch.nn as nn
from typing import Optional


class DiseaseAttentionGenerator(nn.Module):
    """
    Generates 14 independent spatial attention maps from shared refined features,
    optionally conditioned on anatomy features.

    Two variants depending on use_anatomy_conditioning:
      False: conv(refined_F)              → [B, 14, H, W]
      True:  conv(concat(refined_F, anat))→ [B, 14, H, W]

    Args:
        feature_channels:   C (backbone feature dim, 1280 or 1536)
        num_classes:        14
        use_anatomy_conditioning: If True, anatomy features are concatenated.
        hidden_channels:    Intermediate channels (default 128)
    """

    def __init__(
        self,
        feature_channels: int,
        num_classes: int = 14,
        use_anatomy_conditioning: bool = False,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.use_anatomy = use_anatomy_conditioning

        in_ch = feature_channels * 2 if use_anatomy_conditioning else feature_channels

        self.attention_head = nn.Sequential(
            # Hidden layer for richer mapping
            nn.Conv2d(in_ch, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # Output: one channel per disease (no sigmoid — applied in forward)
            nn.Conv2d(hidden_channels, num_classes, 1, bias=True),
        )

    def forward(
        self,
        F_refined: torch.Tensor,
        anatomy_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        F_refined:          [B, C, H, W]
        anatomy_features:   [B, C, H, W]  or None
        Returns:
            disease_atts:   [B, 14, H, W]  values in (0, 1)
        """
        if self.use_anatomy and anatomy_features is not None:
            gen_input = torch.cat([F_refined, anatomy_features], dim=1)  # [B, 2C, H, W]
        else:
            gen_input = F_refined  # [B, C, H, W]

        att_logits = self.attention_head(gen_input)  # [B, 14, H, W]
        return torch.sigmoid(att_logits)              # [B, 14, H, W]


class DiseaseHead(nn.Module):
    """
    Applies disease-specific attention and classifies.

    For disease d:
        F_d    = F_refined * A_d          [B, C, H, W]
        z_d    = GAP(F_d)                 [B, C]
        logit_d = Linear(C, 1)            [B, 1]

    Optional global + local fusion (§54):
        z_global = GAP(F_refined)         [B, C]
        z_d      = concat(z_global, z_local_d)  [B, 2C]
        logit_d  = Linear(2C, 1)         [B, 1]

    Args:
        feature_channels:  C
        num_classes:       14
        use_global_local:  If True, concat global + local pooled features.
    """

    def __init__(
        self,
        feature_channels: int,
        num_classes: int = 14,
        use_global_local: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_global_local = use_global_local

        head_in = feature_channels * 2 if use_global_local else feature_channels
        # 14 independent lightweight linear classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(head_in, 1) for _ in range(num_classes)
        ])

    def forward(
        self,
        F_refined: torch.Tensor,
        disease_atts: torch.Tensor,
    ) -> torch.Tensor:
        """
        F_refined:      [B, C, H, W]
        disease_atts:   [B, 14, H, W]   values in (0, 1)
        Returns:
            logits:     [B, 14]          raw logits (use BCEWithLogitsLoss)
        """
        B, C, H, W = F_refined.shape

        # Optional global representation  [B, C]
        if self.use_global_local:
            z_global = F_refined.mean(dim=(2, 3))  # GAP → [B, C]

        # Vectorised: [B, 1, C, H, W] * [B, 14, 1, H, W] = [B, 14, C, H, W]
        # Then mean over H, W: [B, 14, C]
        F_expanded = F_refined.unsqueeze(1)                    # [B, 1, C, H, W]
        A_expanded = disease_atts.unsqueeze(2)                 # [B, 14, 1, H, W]
        F_weighted = (F_expanded * A_expanded).mean(dim=(3, 4))  # [B, 14, C]

        logits = []
        for d in range(self.num_classes):
            z_d = F_weighted[:, d, :]  # [B, C]
            if self.use_global_local:
                z_d = torch.cat([z_global, z_d], dim=1)  # [B, 2C]
            logits.append(self.classifiers[d](z_d))       # [B, 1]

        return torch.cat(logits, dim=1)  # [B, 14]
