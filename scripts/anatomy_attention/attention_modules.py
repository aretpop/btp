"""
attention_modules.py
====================
Reusable attention modules for the anatomy-conditioned architecture.

M1 components:
  - ChannelAttention   (Squeeze-and-Excitation style, dual avg+max pool)
  - SpatialAttention   (CBAM style, 7×7 conv on avg+max channel maps)
  - MultiScaleBlock    (three parallel dilated convs + fusion)

All modules are documented with tensor dimension comments.
None of these are claimed as novel — they are standard literature components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



class ChannelAttention(nn.Module):
    """
    Dual-pool Squeeze-and-Excitation channel attention (CBAM variant).

    Generates A_channel ∈ [B, C, 1, 1] and returns F * A_channel.

    Args:
        channels:  Number of input channels C.
        reduction: Bottleneck reduction ratio (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # [B, C, 1, 1]

        # Shared MLP applied to both pools
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        F:          [B, C, H, W]
        Returns:    [B, C, H, W]   (F scaled by channel attention)
        """
        avg_out = self.mlp(self.avg_pool(F))   # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(F))   # [B, C, 1, 1]
        A_ch = torch.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return F * A_ch


class SpatialAttention(nn.Module):
    """
    CBAM-style spatial attention.

    Collapses the channel dimension via avg+max, then learns a
    spatial gate with a 7×7 convolution.

    Args:
        kernel_size: Convolution kernel (3 or 7, default 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        F:          [B, C, H, W]
        Returns:    [B, C, H, W]   (F scaled by spatial attention)
        """
        avg_map = F.mean(dim=1, keepdim=True)        # [B, 1, H, W]
        max_map, _ = F.max(dim=1, keepdim=True)      # [B, 1, H, W]
        spatial_in = torch.cat([avg_map, max_map], dim=1)  # [B, 2, H, W]
        A_sp = torch.sigmoid(self.conv(spatial_in))         # [B, 1, H, W]
        return F * A_sp


class MultiScaleBlock(nn.Module):
    """
    Lightweight multi-scale feature extraction using parallel dilated convolutions.

    Motivation: diseases differ vastly in scale:
      Nodule      → small local finding   (dilation 1)
      Mass/Edema  → medium regional       (dilation 2)
      Cardiomegaly→ large global structure (dilation 3)
      Global ctx  → 1×1 projection        (dilation 0)

    Output channels = in_channels (projected via 1×1 fusion).

    Args:
        in_channels:    C (backbone feature dim)
        branch_channels: channels per branch (default in_channels // 4)
        dilations:       dilation rates for the 3 conv branches
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: Optional[int] = None,
        dilations: tuple = (1, 2, 3),
    ):
        super().__init__()
        if branch_channels is None:
            branch_channels = max(in_channels // 4, 16)

        # Branch 0: 1×1 for global context / cheap projection
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branches 1-3: 3×3 dilated convs for different receptive fields
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])

        total_branch_channels = branch_channels * (1 + len(dilations))

        # Fuse all branches back to in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(total_branch_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        F:          [B, C, H, W]
        Returns:    [B, C, H, W]   (multi-scale fused features, same spatial size)
        """
        branches = [self.branch_1x1(F)]
        for branch in self.dilated_branches:
            branches.append(branch(F))

        # Concatenate along channel dim: [B, total_branch_channels, H, W]
        cat = torch.cat(branches, dim=1)
        return self.fuse(cat)   # [B, C, H, W]

