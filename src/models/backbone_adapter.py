"""
src/models/backbone_adapter.py

BackboneAdapter interface and EfficientNet-B1 implementation.
Registry stubs for DenseNet-121 and ConvNeXt-Pico (timm).

The adapter returns a dict with keys:
  mid   – intermediate feature map  [B, C_mid, H_mid, W_mid]
  deep  – final feature map         [B, C_deep, H_deep, W_deep]
  channels – dict(mid=int, deep=int)
  stride   – dict(mid=int, deep=int) (relative to input resolution)

All spatial dims and channel counts are derived by a dummy forward pass at
the configured resolution — no hardcoded constants.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BackboneAdapter(ABC, nn.Module):
    """
    Abstract backbone adapter.
    Subclasses must implement `forward` returning the standard dict and
    expose `mid_channels`, `deep_channels`, `mid_stride`, `deep_stride`
    computed at construction via a dummy forward pass.
    """

    def __init__(self, input_resolution: int = 224):
        super().__init__()
        self.input_resolution = input_resolution
        # Filled by _probe() in subclasses:
        self.mid_channels: int = 0
        self.deep_channels: int = 0
        self.mid_h: int = 0
        self.mid_w: int = 0
        self.deep_h: int = 0
        self.deep_w: int = 0
        self.mid_stride: int = 0
        self.deep_stride: int = 0

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            {
                'mid':  [B, C_mid,  H_mid,  W_mid],
                'deep': [B, C_deep, H_deep, W_deep],
            }
        """
        ...

    def probe_info(self) -> Dict:
        return {
            "mid_channels": self.mid_channels,
            "deep_channels": self.deep_channels,
            "mid_h": self.mid_h,
            "mid_w": self.mid_w,
            "deep_h": self.deep_h,
            "deep_w": self.deep_w,
            "mid_stride": self.mid_stride,
            "deep_stride": self.deep_stride,
        }


# ---------------------------------------------------------------------------
# EfficientNet-B1 adapter
# ---------------------------------------------------------------------------

class EfficientNetB1Adapter(BackboneAdapter):
    """
    EfficientNet-B1 adapter.
    mid_tap: index into backbone.features (default 5)
    deep_tap: index into backbone.features (default 8)
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_resolution: int = 224,
        mid_tap: int = 5,
        deep_tap: int = 8,
    ):
        super().__init__(input_resolution=input_resolution)
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b1(weights=weights)
        self.features = backbone.features   # nn.Sequential of blocks 0..8
        self.mid_tap = mid_tap
        self.deep_tap = deep_tap
        self._probe()

    def _probe(self):
        res = self.input_resolution
        dummy = torch.zeros(1, 3, res, res)
        with torch.no_grad():
            x = dummy
            for i, block in enumerate(self.features):
                x = block(x)
                if i == self.mid_tap:
                    B, C, H, W = x.shape
                    self.mid_channels = C
                    self.mid_h = H
                    self.mid_w = W
                    self.mid_stride = res // H
                if i == self.deep_tap:
                    B, C, H, W = x.shape
                    self.deep_channels = C
                    self.deep_h = H
                    self.deep_w = W
                    self.deep_stride = res // H
                    break
        logger.info(
            "EfficientNetB1Adapter probe @ %dpx: mid=[%d,%dx%d] stride=%d  "
            "deep=[%d,%dx%d] stride=%d",
            res,
            self.mid_channels, self.mid_h, self.mid_w, self.mid_stride,
            self.deep_channels, self.deep_h, self.deep_w, self.deep_stride,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mid = None
        deep = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == self.mid_tap:
                mid = x
            if i == self.deep_tap:
                deep = x
                break
        return {"mid": mid, "deep": deep}

    def freeze(self):
        """Freeze all backbone weights (Stage A)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_last_n_blocks(self, n: int = 4):
        """Unfreeze the last n feature blocks (Stage B)."""
        self.train()
        for p in self.parameters():
            p.requires_grad_(False)
        start = max(0, len(self.features) - n)
        for i in range(start, len(self.features)):
            for p in self.features[i].parameters():
                p.requires_grad_(True)
        logger.info("EfficientNetB1: unfroze features[%d:]", start)

    def unfreeze_all(self):
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Registry stubs (not tested) — adding a new backbone is a 1-liner here
# ---------------------------------------------------------------------------

class DenseNet121Adapter(BackboneAdapter):
    """STUB — not implemented yet. Requires torchvision DenseNet-121."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError(
            "DenseNet121Adapter is a registry stub. "
            "Implement forward() to activate."
        )

    def forward(self, x):
        raise NotImplementedError


class ConvNeXtPicoAdapter(BackboneAdapter):
    """STUB — not implemented yet. Requires timm features_only."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError(
            "ConvNeXtPicoAdapter is a registry stub. "
            "Implement forward() to activate."
        )

    def forward(self, x):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY = {
    "efficientnet_b1": EfficientNetB1Adapter,
    "densenet121":     DenseNet121Adapter,
    "convnext_pico":   ConvNeXtPicoAdapter,
}


def build_backbone(
    name: str = "efficientnet_b1",
    pretrained: bool = True,
    input_resolution: int = 224,
    mid_tap: int = 5,
    deep_tap: int = 8,
) -> BackboneAdapter:
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(_BACKBONE_REGISTRY.keys())}"
        )
    cls = _BACKBONE_REGISTRY[name]
    return cls(
        pretrained=pretrained,
        input_resolution=input_resolution,
        mid_tap=mid_tap,
        deep_tap=deep_tap,
    )

