"""
scripts/train_chexpert_ensemble_b0.py
====================================
Train EfficientNet-B0 on the Materialized CheXpert Dataset using:
  1. Method 1: Tiered 40/40/20 Multi-Label Batch Sampler (Batch size = 32)
     - 14 images from Pool 1 (Rare/Focal: Pneumonia, Pleural Other, Fracture, Lung Lesion, Consolidation, Enlarged Cardiomediastinum)
     - 12 images from Pool 2 (Common/Diffuse: Cardiomegaly, Edema, Effusion, Pneumothorax, Atelectasis, Lung Opacity)
     - 6 images from Pool 3 (Clean Normal Controls: No Finding)
  2. Three Loss Function Variants:
     - 'bce'     : Standard BCEWithLogitsLoss
     - 'bce_wpos': BCEWithLogitsLoss with Sqrt-smoothed pos_weight
     - 'asl'     : Asymmetric Loss tuned with moderate negative focusing (gamma_neg=2.0, gamma_pos=0, clip=0.05)
  3. Two-Stage Training Protocol:
     - Stage 1: Freeze backbone, train classification head (higher LR: 1e-3)
     - Stage 2: Unfreeze entire backbone, fine-tune end-to-end (lower LR: 2e-5)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "chexpert_cache"
INDEX_CSV = DATA_DIR / "chexpert_subset" / "chexpert_materialized_index.csv"
VAL_INDEX_CSV = DATA_DIR / "chexpert_subset" / "valid_index.csv"
VAL_CACHE_DIR = DATA_DIR / "chexpert_val_cache"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "chexpert_ensemble"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
NUM_CLASSES = len(CHEXPERT_14_LABELS)

POOL1_DISEASES = [
    "Pneumonia", "Pleural Other", "Fracture",
    "Lung Lesion", "Consolidation", "Enlarged Cardiomediastinum"
]
POOL2_DISEASES = [
    "Cardiomegaly", "Edema", "Pleural Effusion",
    "Pneumothorax", "Atelectasis", "Lung Opacity"
]

# Sqrt-smoothed pos_weight derived from our 48,493 dataset:
SQRT_POS_WEIGHT = torch.tensor([
    2.24,  # No Finding
    3.63,  # Enlarged Cardiomediastinum
    2.06,  # Cardiomegaly
    1.02,  # Lung Opacity
    4.22,  # Lung Lesion
    1.80,  # Edema
    3.58,  # Consolidation
    8.26,  # Pneumonia
    2.28,  # Atelectasis
    2.72,  # Pneumothorax
    1.28,  # Pleural Effusion
    6.51,  # Pleural Other
    4.24,  # Fracture
    0.95   # Support Devices
], dtype=torch.float32)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("chexpert_b0_train")


# ==============================================================================
# 1. Dataset & Tiered Multi-Label Batch Sampler
# ==============================================================================
class CheXpertLocalDataset(Dataset):
    """Loads localized CheXpert chest X-rays with multi-label binary vectors."""
    def __init__(self, df: pd.DataFrame, transform=None, is_val=False, val_cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_val = is_val
        self.val_cache_dir = Path(val_cache_dir) if val_cache_dir else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["file_id"]
        path = CACHE_DIR / f"{fid}.jpg"

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        # Multi-label vector (1.0 for positive, 0.0 for negative/uncertain/nan under U-zeros protocol)
        labels = []
        for lbl in CHEXPERT_14_LABELS:
            val = row.get(lbl, 0.0)
            labels.append(1.0 if val == 1.0 else 0.0)

        return image, torch.tensor(labels, dtype=torch.float32)


class TieredMultiLabelBatchSampler(Sampler):
    """
    Method 1: 40/40/20 Tiered Multi-Label Sampler (Batch size = 32)
      - 14 images from Pool 1 (Rare/Focal)
      - 12 images from Pool 2 (Common/Diffuse)
      - 6 images from Pool 3 (Clean Normal Controls)
    """
    def __init__(self, df: pd.DataFrame, batches_per_epoch: int = 1000, seed: int = 42):
        self.batches_per_epoch = batches_per_epoch
        self.rng = np.random.default_rng(seed)

        # Build index pools
        self.p1_buckets = [self.rng.permutation(df[df[d] == 1.0].index.values).tolist() for d in POOL1_DISEASES]
        self.p2_buckets = [self.rng.permutation(df[df[d] == 1.0].index.values).tolist() for d in POOL2_DISEASES]
        self.p3_indices = self.rng.permutation(df[(df["No Finding"] == 1.0) & (df["num_confirmed"] <= 1)].index.values).tolist()

        self.p1_pointers = [0] * len(self.p1_buckets)
        self.p2_pointers = [0] * len(self.p2_buckets)
        self.p3_pointer = 0

    def _get_samples(self, buckets, pointers, count):
        samples = []
        n_b = len(buckets)
        for i in range(count):
            b_idx = i % n_b
            ptr = pointers[b_idx]
            if ptr >= len(buckets[b_idx]):
                self.rng.shuffle(buckets[b_idx])
                pointers[b_idx] = 0
                ptr = 0
            samples.append(buckets[b_idx][ptr])
            pointers[b_idx] += 1
        return samples

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            # 14 from Pool 1 (Rare)
            batch_p1 = self._get_samples(self.p1_buckets, self.p1_pointers, 14)
            # 12 from Pool 2 (Common)
            batch_p2 = self._get_samples(self.p2_buckets, self.p2_pointers, 12)
            # 6 from Pool 3 (Clean Normal Controls)
            batch_p3 = []
            for _ in range(6):
                if self.p3_pointer >= len(self.p3_indices):
                    self.rng.shuffle(self.p3_indices)
                    self.p3_pointer = 0
                batch_p3.append(self.p3_indices[self.p3_pointer])
                self.p3_pointer += 1

            full_batch = batch_p1 + batch_p2 + batch_p3
            self.rng.shuffle(full_batch)
            yield full_batch

    def __len__(self):
        return self.batches_per_epoch


# ==============================================================================
# 2. Loss Functions (BCE, BCE+wpos, Tuned ASL)
# ==============================================================================
class TunedAsymmetricLoss(nn.Module):
    """
    Tuned Asymmetric Loss (ASL) for Multi-Label Radiography
    Tuned with moderate negative focusing (gamma_neg=2.0) and gamma_pos=0 so that
    rare sampled positives are preserved with full gradient signal while easy negatives
    are moderately down-weighted without causing precision collapse or over-weighting.
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # x: logits, y: targets
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric margin clipping for easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Standard binary cross entropy
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = (pt0 + pt1).detach()
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


def build_loss(loss_type: str, device: torch.device):
    if loss_type == "bce":
        log.info("Using Loss: Standard BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    elif loss_type == "bce_wpos":
        log.info(f"Using Loss: BCEWithLogitsLoss with Sqrt pos_weight: {SQRT_POS_WEIGHT.tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=SQRT_POS_WEIGHT.to(device))
    elif loss_type == "asl":
        log.info("Using Loss: Tuned Asymmetric Loss (gamma_neg=2.0, gamma_pos=0.0, clip=0.05)")
        return TunedAsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==============================================================================
# 3. Model Construction (EfficientNet-B0)
# ==============================================================================
def build_efficientnet_b0(pretrained: bool = True, num_classes: int = 14):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


# ==============================================================================
# 4. Evaluation Function
# ==============================================================================
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    aurocs = []
    for c in range(NUM_CLASSES):
        # Only compute if at least one positive and one negative sample exists
        if len(np.unique(targets_arr[:, c])) > 1:
            auc = roc_auc_score(targets_arr[:, c], probs_arr[:, c])
            aurocs.append(auc)
        else:
            aurocs.append(0.5)

    macro_auroc = np.mean(aurocs)
    preds_bin = (probs_arr >= 0.5).astype(int)
    macro_f1 = f1_score(targets_arr, preds_bin, average="macro", zero_division=0)
    return macro_auroc, macro_f1, aurocs


# ==============================================================================
# 5. Training Orchestration (Two-Stage Protocol)
# ==============================================================================
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load Materialized Index
    df_all = pd.read_csv(INDEX_CSV)
    log.info(f"Loaded materialized CheXpert index: {len(df_all):,} samples")

    # 90/10 Train/Validation Split of Materialized Cohort
    rng = np.random.default_rng(42)
    val_mask = rng.random(len(df_all)) < 0.10
    train_df = df_all[~val_mask].reset_index(drop=True)
    val_df = df_all[val_mask].reset_index(drop=True)

    if args.smoke_test:
        val_df = val_df.head(64)
    else:
        val_df = val_df.head(1000)  # Standardized 1,000-image validation cohort
    log.info(f"Cohort Split: {len(train_df):,} Train / {len(val_df):,} Validation")

    train_ds = CheXpertLocalDataset(train_df, transform=train_transform)
    val_ds = CheXpertLocalDataset(val_df, transform=val_transform)

    batch_sampler = TieredMultiLabelBatchSampler(
        train_df,
        batches_per_epoch=args.batches_per_epoch if not args.smoke_test else 2,
        seed=42
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.workers if not args.smoke_test else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.workers if not args.smoke_test else 0,
        pin_memory=True
    )

    # Instantiate Model
    model = build_efficientnet_b0(pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    criterion = build_loss(args.loss, device)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_auroc = 0.0
    checkpoint_name = f"best_b0_{args.loss}.pth"
    save_path = CHECKPOINT_DIR / checkpoint_name

    # ==========================================================================
    # STAGE 1: Head Warmup (Backbone Frozen)
    # ==========================================================================
    stage1_epochs = args.stage1_epochs if not args.smoke_test else 1
    log.info("\n" + "=" * 80)
    log.info(f"STAGE 1: TRAINING CLASSIFICATION HEAD ({stage1_epochs} Epochs, Backbone Frozen, LR=1e-3)")
    log.info("=" * 80)

    # Freeze features backbone
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_head.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer_head)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                log.info(f"  Stage 1 Ep {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {train_loss/(batch_idx+1):.4f}")

        val_auc, val_f1, _ = evaluate(model, val_loader, device)
        log.info(f"[STAGE 1 - Ep {epoch}] Train Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auc:.4f} | Val F1: {val_f1:.4f} ({time.time()-t0:.1f}s)")

        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "loss_type": args.loss,
                "model_state": model.state_dict(),
                "val_auroc": val_auc,
                "val_f1": val_f1
            }, save_path)
            log.info(f"  --> Checkpoint saved: {save_path.name} (AUROC: {val_auc:.4f})")

    # ==========================================================================
    # STAGE 2: Full Backbone Fine-Tuning (End-to-End)
    # ==========================================================================
    stage2_epochs = args.stage2_epochs if not args.smoke_test else 1
    log.info("\n" + "=" * 80)
    log.info(f"STAGE 2: FULL BACKBONE FINE-TUNING ({stage2_epochs} Epochs, Unfrozen, LR=2e-5)")
    log.info("=" * 80)

    # Unfreeze entire network
    for param in model.parameters():
        param.requires_grad = True

    optimizer_full = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_full.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer_full)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                log.info(f"  Stage 2 Ep {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {train_loss/(batch_idx+1):.4f}")

        scheduler.step()
        val_auc, val_f1, aurocs = evaluate(model, val_loader, device)
        log.info(f"[STAGE 2 - Ep {epoch}] Train Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auc:.4f} | Val F1: {val_f1:.4f} ({time.time()-t0:.1f}s)")

        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            torch.save({
                "epoch": epoch,
                "stage": 2,
                "loss_type": args.loss,
                "model_state": model.state_dict(),
                "val_auroc": val_auc,
                "val_f1": val_f1,
                "per_class_auroc": {CHEXPERT_14_LABELS[i]: aurocs[i] for i in range(NUM_CLASSES)}
            }, save_path)
            log.info(f"  --> Checkpoint saved: {save_path.name} (AUROC: {val_auc:.4f})")

    log.info("\n" + "=" * 80)
    log.info(f"TRAINING COMPLETE FOR MODEL: EfficientNet-B0 [{args.loss}]")
    log.info(f"Best Validation AUROC: {best_val_auroc:.4f}")
    log.info(f"Final Model Saved: {save_path}")
    log.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 with Method 1 Tiered Sampler")
    parser.add_argument("--loss", type=str, required=True, choices=["bce", "bce_wpos", "asl"],
                        help="Loss function: 'bce', 'bce_wpos', or 'asl'")
    parser.add_argument("--stage1-epochs", type=int, default=3, help="Stage 1 head warmup epochs")
    parser.add_argument("--stage2-epochs", type=int, default=5, help="Stage 2 full fine-tuning epochs")
    parser.add_argument("--batches-per-epoch", type=int, default=800, help="Batches per epoch (800 * 32 = 25,600 scans/ep)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-batch smoke test")
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
