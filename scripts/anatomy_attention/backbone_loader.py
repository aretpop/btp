"""
backbone_loader.py
==================
Safe, immutable frozen feature extractor from existing Phase 1 checkpoints.

Verified feature map shapes (224x224 input):
  B0: [B, 1280, 7, 7]
  B1: [B, 1280, 7, 7]
  B3: [B, 1536, 7, 7]

The original checkpoint is NEVER modified. A deep copy of the model is created,
the old classifier is detached, and the backbone is frozen.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Backbone feature dims — verified by live probe 2026-10-03
BACKBONE_FEATURE_DIMS: Dict[str, int] = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b3": 1536,
}


def load_frozen_backbone(
    checkpoint_path: str,
    device: torch.device,
    verify_immutability: bool = True,
) -> tuple[nn.Module, int, str]:
    """
    Load a Phase 1 checkpoint and return a frozen feature-extractor backbone.

    The original checkpoint object is never touched. We deepcopy the model,
    strip the old classifier, and freeze all backbone parameters.

    Args:
        checkpoint_path: Path to a Phase 1 .pth checkpoint.
        device:          Target device.
        verify_immutability: If True, snapshot original weights and assert
                             they are identical after deepcopy.

    Returns:
        backbone:     nn.Module with `.features(x)` returning [B, C, 7, 7].
        feature_dim:  C (1280 or 1536).
        backbone_name: "efficientnet_b0" | "efficientnet_b1" | "efficientnet_b3"
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_phase1_classifier import ChestXRayModel, NUM_CLASSES

    ckpt_path = Path(checkpoint_path)
    log.info(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    log.info(f"Backbone detected: {backbone_name}")

    # --- 1. Build and load the original model ---
    original_model = ChestXRayModel(
        backbone_name=backbone_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    original_model.load_state_dict(ckpt["model_state"])
    original_model.eval()

    # --- 2. Snapshot original weights for immutability check ---
    if verify_immutability:
        original_snapshot: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone()
            for k, v in original_model.state_dict().items()
        }

    # --- 3. Deep-copy into a NEW object — original_model is never touched again ---
    backbone_model = copy.deepcopy(original_model)

    # --- 4. Verify the original was not mutated by deepcopy ---
    if verify_immutability:
        for k, v in original_model.state_dict().items():
            assert torch.equal(v.cpu(), original_snapshot[k]), \
                f"IMMUTABILITY VIOLATION: {k} was mutated during deepcopy!"
        log.info("✓ Immutability check passed — original checkpoint untouched.")

    # --- 5. Expose only the EfficientNet feature extractor ---
    # backbone_model.backbone.features(x) → [B, C, 7, 7]
    # We return backbone_model.backbone (the torchvision EfficientNet object)
    # so callers can call .features(x) directly.
    feature_backbone: nn.Module = backbone_model.backbone

    # Detach/remove the old classifier to prevent accidental use.
    # We replace it with Identity so state_dict keys remain valid
    # but no classification is performed.
    feature_backbone.classifier = nn.Identity()

    # --- 6. Freeze all backbone parameters ---
    for param in feature_backbone.parameters():
        param.requires_grad = False

    feature_dim = BACKBONE_FEATURE_DIMS[backbone_name]
    log.info(f"Backbone frozen. Feature dim: {feature_dim}, Device: {device}")

    feature_backbone.to(device).eval()
    return feature_backbone, feature_dim, backbone_name


class FrozenBackboneWrapper(nn.Module):
    """
    Thin wrapper around a frozen EfficientNet backbone.
    Keeps the backbone in eval() regardless of the parent model's mode.

    Usage:
        wrapper = FrozenBackboneWrapper(feature_backbone)
        F = wrapper(x)   # [B, C, Hf, Wf]
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Ensure it's frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "FrozenBackboneWrapper":
        """Override: backbone always stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  (224×224 ImageNet-normalized)
        Returns:
            F: [B, C, Hf, Wf]  — spatial feature maps before GAP
        """
        with torch.no_grad():
            return self.backbone.features(x)
