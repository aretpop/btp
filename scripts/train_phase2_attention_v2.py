"""
train_phase2_attention_v2.py
=============================
Phase 2 Attention Model — Version 2 with sharper attention maps.

KEY CHANGES vs V1 (train_phase2_attention.py):
  Fix 1: Disease attention generator now runs at 14×14 resolution
          by tapping Block 5 features instead of the final 7×7 Block 8.
          This gives 4× spatial resolution on the attention maps.
          Classification still uses 7×7 Block 8 (1280-dim) for full power.

  Fix 2: Attention Concentration Loss forces each disease attention
          map to concentrate on a small region by minimising its entropy.
          lambda_concentration=0.01 by default (configurable).

Original backbone models remain completely untouched (deepcopy).
All existing checkpoints are forward-compatible for comparison.
"""

import os
import copy
import time
import argparse
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

import mlflow
from sklearn.metrics import roc_auc_score

from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms,
    NUM_CLASSES, CLASSES, CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# ===========================================================================
# SAME SHARED MODULES AS V1 (reused unchanged)
# ===========================================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        los_pos = los_pos * torch.pow(1 - xs_pos, self.gamma_pos)
        los_neg = los_neg * torch.pow(1 - xs_neg, self.gamma_neg)
        return -(los_pos + los_neg).mean()


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class MultiScaleFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(out_channels // 4, 16)
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=3, dilation=3, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(mid * 4, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))


# ===========================================================================
# FIX 2: ATTENTION CONCENTRATION LOSS
# ===========================================================================
def attention_concentration_loss(disease_atts: torch.Tensor) -> torch.Tensor:
    """
    Minimise the spatial entropy of each disease attention map.

    Low entropy = attention is concentrated on a small sharp region.
    High entropy = attention is spread uniformly across the whole image (our problem).

    Entropy of a probability distribution p: H = -sum(p * log(p))
    We normalise the spatial map to a probability distribution first.

    Args:
        disease_atts: [B, 14, H, W]  values in (0, 1)
    Returns:
        Scalar: mean spatial entropy across all 14 maps and all images.
    """
    B, D, H, W = disease_atts.shape
    eps = 1e-8

    # Normalise each map to a spatial probability distribution
    flat = disease_atts.view(B, D, -1)                           # [B, 14, H*W]
    p = flat / (flat.sum(dim=2, keepdim=True) + eps)              # [B, 14, H*W]

    # Entropy: -sum(p * log(p)) per map
    entropy = -(p * torch.log(p + eps)).sum(dim=2)                # [B, 14]

    # Normalise by log(H*W) so entropy is in [0, 1]
    max_entropy = np.log(H * W)
    normalised_entropy = entropy / max_entropy                     # [B, 14]

    return normalised_entropy.mean()


# ===========================================================================
# FIX 1 + 2: IMPROVED MultiAttentionXRayModel V2
# ===========================================================================
class MultiAttentionXRayModelV2(nn.Module):
    """
    Improved Phase 2 attention model with sharper disease-specific maps.

    Architecture:
        Frozen Backbone (Block 0-8, 1280@7x7)
                |
                +-- Block 5 output (112@14x14)  --> Disease Attention Generator
                |                                    --> [B, 14, 14, 14] SHARP MAPS
                |
                +-- Block 8 output (1280@7x7)   --> Multi-Scale --> Channel --> Spatial
                                                    --> Shared feature refinement
                                                    --> 14 × (Weighted GAP → Linear → logit)

    Key difference from V1:
        Disease attention maps are 14×14 instead of 7×7 (4× sharper).
        Attention concentration loss pushes maps to focus on small regions.
    """

    def __init__(self, original_backbone_model: nn.Module, feature_dim: int, num_classes=14):
        super().__init__()

        self.backbone = copy.deepcopy(original_backbone_model)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = feature_dim
        self.c2 = 512

        # Shared refinement on full-resolution 7×7 features (same as V1)
        self.multi_scale = MultiScaleFeatureBlock(in_channels=feature_dim, out_channels=self.c2)
        self.channel_att = ChannelAttention(channels=self.c2)
        self.spatial_att = SpatialAttention(kernel_size=7)

        # FIX 1: Disease attention taps Block 5 (14×14, 112 channels)
        # Block 5 output: [B, 112, 14, 14] — 4× higher resolution than Block 8
        block5_channels = 112
        self.disease_attention_generator = nn.Sequential(
            nn.Conv2d(block5_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),   # [B, 14, 14, 14]
        )

        # 14 lightweight classifiers on the refined 7×7 shared features
        self.classifiers = nn.ModuleList([
            nn.Linear(self.c2, 1) for _ in range(num_classes)
        ])

    def train(self, mode=True):
        """Backbone must never enter training mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _get_backbone_features(self, x):
        """
        Returns:
            f_early:  [B, 112, 14, 14] from Block 5 — used for attention maps
            f_final:  [B, 1280, 7, 7]  from Block 8 — used for classification
        """
        if hasattr(self.backbone, 'backbone'):
            blocks = self.backbone.backbone.features
        else:
            blocks = self.backbone.features

        z = x
        f_early = None
        for i, block in enumerate(blocks):
            z = block(z)
            if i == 5:
                f_early = z   # [B, 112, 14, 14]
        f_final = z           # [B, 1280, 7, 7]
        return f_early, f_final

    def forward(self, x, mask=None, return_attention=False):
        B = x.shape[0]

        # --- FROZEN BACKBONE (two resolutions) ---
        with torch.no_grad():
            f_early, f_final = self._get_backbone_features(x)
        # f_early: [B, 112, 14, 14]
        # f_final: [B, 1280, 7, 7]

        # --- SHARED REFINEMENT on 7×7 features ---
        f_ms     = self.multi_scale(f_final)   # [B, 512, 7, 7]
        f_ca     = self.channel_att(f_ms)       # [B, 512, 7, 7]
        f_shared = self.spatial_att(f_ca)       # [B, 512, 7, 7]

        # --- FIX 1: DISEASE ATTENTION at 14×14 ---
        att_logits   = self.disease_attention_generator(f_early)   # [B, 14, 14, 14]
        disease_atts = torch.sigmoid(att_logits)                   # [B, 14, 14, 14]

        # Downsample attention maps to 7×7 to match f_shared for classification
        disease_atts_7 = F.adaptive_avg_pool2d(disease_atts, (7, 7))  # [B, 14, 7, 7]

        # --- DISEASE-SPECIFIC CLASSIFICATION ---
        f_expanded = f_shared.unsqueeze(1)                           # [B, 1, 512, 7, 7]
        a_expanded = disease_atts_7.unsqueeze(2)                     # [B, 14, 1, 7, 7]
        z = (f_expanded * a_expanded).mean(dim=(3, 4))               # [B, 14, 512]

        logits = torch.cat([self.classifiers[d](z[:, d, :]) for d in range(14)], dim=1)  # [B, 14]

        if return_attention:
            return logits, disease_atts   # disease_atts at full 14×14 resolution
        return logits


# ===========================================================================
# TRAINING
# ===========================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Backbone: {args.backbone}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    full_train = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("train"))
    full_val   = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))
    n = len(full_train)
    n_test, n_val = int(n * 0.20), int(n * 0.10)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(42)
    tr_idx, val_idx, _ = random_split(range(n), [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(Subset(full_train, tr_idx.indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(Subset(full_val, val_idx.indices), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # Load Phase 1 backbone
    ckpt_path = CHECKPOINT_DIR / f"best_classifier_{args.backbone}.pth"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    base_model = ChestXRayModel(backbone_name=args.backbone, num_classes=NUM_CLASSES, pretrained=False)
    base_model.load_state_dict(ckpt["model_state"])

    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModelV2(base_model, feature_dim=in_features).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable: {trainable:,}  Frozen: {frozen:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = AsymmetricLoss()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ChestXRay14_Phase2_AttentionV2")
    best_auroc = 0.0

    with mlflow.start_run(run_name=f"phase2v2_{args.backbone}"):
        mlflow.log_params(vars(args))

        for epoch in range(1, args.epochs + 1):
            # --- Train ---
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                if args.lambda_concentration > 0:
                    logits, disease_atts = model(images, return_attention=True)
                    cls_loss  = criterion(logits, labels)
                    conc_loss = attention_concentration_loss(disease_atts)
                    loss = cls_loss + args.lambda_concentration * conc_loss
                else:
                    logits = model(images)
                    loss   = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # --- Validate ---
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    logits = model(images)
                    all_probs.append(torch.sigmoid(logits).cpu())
                    all_labels.append(labels)

            probs  = torch.cat(all_probs).numpy()
            labels = torch.cat(all_labels).numpy()
            per_cls = []
            for i in range(NUM_CLASSES):
                try:    per_cls.append(roc_auc_score(labels[:, i], probs[:, i]))
                except: per_cls.append(0.5)
            macro_auroc = float(np.mean(per_cls))

            scheduler.step()
            log.info(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val AUROC: {macro_auroc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_macro_auroc": macro_auroc}, step=epoch)

            if macro_auroc > best_auroc:
                best_auroc = macro_auroc
                ckpt_save = CHECKPOINT_DIR / f"best_attention_v2_{args.backbone}.pth"
                torch.save({
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "model_state": model.state_dict(),
                    "macro_auroc": macro_auroc,
                    "classes": CLASSES,
                    "version": "v2",
                }, ckpt_save)
                log.info(f"  >> New best AUROC={best_auroc:.4f} saved to {ckpt_save.name}")

        log.info(f"Done. Best Val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",             default="efficientnet_b1",
                        choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b3"])
    parser.add_argument("--epochs",               type=int,   default=20)
    parser.add_argument("--batch-size",           type=int,   default=32)
    parser.add_argument("--lr",                   type=float, default=1e-4)
    parser.add_argument("--weight-decay",         type=float, default=1e-4)
    parser.add_argument("--num-workers",          type=int,   default=0)
    parser.add_argument("--lambda-concentration", type=float, default=0.01,
                        help="Weight for attention concentration loss (Fix 2). 0=disabled.")
    args = parser.parse_args()
    main(args)
