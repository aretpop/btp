"""
=============================================================================
train_phase1_classifier.py
=============================================================================
Phase 1: Vision Feature Extractor — Multi-Label Classifier
Dataset:  NIH ChestX-ray14 (224x224 resized)
Backbone: EfficientNet-B0 (ImageNet pretrained)
Outputs:  14 raw logits (one per disease class)

This script is intentionally scoped to ONLY the baseline classifier.
No XAI, no Grad-CAM, no text generation hooks.

Pipeline overview:
  1. CXRDataset       — reads CSV, builds multi-hot label vectors
  2. ChestXRayModel   — EfficientNet-B0 with a replaced linear head
  3. compute_pos_weight — dynamically balances extreme class imbalance
  4. train_one_epoch  — training loop with BCEWithLogitsLoss
  5. validate         — AUROC calculation via torchmetrics
  6. main             — MLflow-wrapped orchestration + checkpoint saving

Author: Antigravity | BTP Edge-AI Pipeline
=============================================================================
"""

import os
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchmetrics.classification import MultilabelAUROC

import mlflow
import mlflow.pytorch

# ---------------------------------------------------------------------------
# Global Configuration — override via CLI args or edit here
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_ROOT      = PROJECT_ROOT / "data" / "nih_cxr14"
IMAGES_DIR     = DATA_ROOT / "images" / "images-224"
METADATA_DIR   = DATA_ROOT / "metadata"
CSV_PATH       = METADATA_DIR / "Data_Entry_2017.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# The 14 canonical NIH ChestX-ray14 disease labels (alphabetically sorted
# for reproducibility — index 0 = Atelectasis, index 13 = Pneumothorax)
CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(CLASSES)  # 14

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. DATASET
# ===========================================================================
class CXRDataset(Dataset):
    """
    Multi-label Dataset for NIH ChestX-ray14.

    Reads Data_Entry_2017.csv and produces:
        image  : (3, 224, 224) normalised tensor
        label  : (14,) multi-hot FloatTensor

    "No Finding" rows map to an all-zero label vector — they are kept
    in the dataset so the model learns negative examples.

    Args:
        csv_path   : Path to Data_Entry_2017.csv
        images_dir : Folder containing .png X-ray files
        transform  : torchvision transform pipeline
        classes    : ordered list of the 14 target class names
    """

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        transform=None,
        classes: list = CLASSES,
    ):
        self.images_dir = Path(images_dir)
        self.transform  = transform
        self.classes    = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # ----- Load and validate the CSV --------------------------------
        df = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
        df.columns = ["filename", "labels"]

        # Drop rows where the image file doesn't actually exist on disk
        # (some Kaggle resized releases are missing a handful of images)
        existing = {p.name for p in self.images_dir.rglob("*.png")}
        if not existing:
            # Fallback: also try .jpg
            existing = {p.name for p in self.images_dir.rglob("*.jpg")}

        before = len(df)
        df = df[df["filename"].isin(existing)].reset_index(drop=True)
        after = len(df)
        if before != after:
            log.warning(
                f"Dropped {before - after} rows whose image files are missing."
            )

        self.df = df
        log.info(f"Dataset loaded: {len(self.df)} samples, {NUM_CLASSES} classes.")

    # ------------------------------------------------------------------
    def _encode_labels(self, label_str: str) -> torch.Tensor:
        """
        Convert a pipe-separated label string to a 14-element multi-hot vector.

        Example:
            "Cardiomegaly|Emphysema"  ->  [0,1,0,0,0,1,0,0,0,0,0,0,0,0]
            "No Finding"              ->  [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        """
        vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for label in label_str.split("|"):
            label = label.strip()
            if label in self.class_to_idx:
                vec[self.class_to_idx[label]] = 1.0
        return vec

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]

        # Open image — X-rays are often stored as grayscale PNGs.
        # Convert to RGB so the 3-channel ImageNet backbone works unchanged.
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self._encode_labels(row["labels"])
        return image, label


# ===========================================================================
# 2. MODEL
# ===========================================================================
class ChestXRayModel(nn.Module):
    """
    EfficientNet backbone with a replaced classification head for multi-label classification.
    Supports dynamic backbone swapping (b0, b1, b3).
    """
    def __init__(self, backbone_name: str = "efficientnet_b3", num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            dropout = 0.3
        elif backbone_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b1(weights=weights)
            dropout = 0.2
        else: # default b0
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            dropout = 0.2

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        log.info(f"ChestXRayModel ready: {backbone_name} -> Linear({in_features}, {num_classes})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# ===========================================================================
# 2.5. ASYMMETRIC LOSS
# ===========================================================================
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification
    Paper: https://arxiv.org/abs/2009.14119
    Down-weights easy negatives to heavily focus on hard negatives and positives.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ x: logits, y: targets """
        # Compute probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic BCE
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = (pt0 + pt1).detach() if self.disable_torch_grad_focal_loss else (pt0 + pt1)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


# ===========================================================================
# 3. CLASS-IMBALANCE WEIGHTING
# ===========================================================================
def compute_pos_weight(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Dynamically compute BCEWithLogitsLoss pos_weight for each class.

    Formula:  pos_weight[c] = N_neg[c] / N_pos[c]

    Reads labels directly from the pandas DataFrame (no image I/O) so
    it completes in milliseconds instead of minutes.

    Args:
        df     : DataFrame slice for the training split (must have 'labels' column)
        device : target device for the returned tensor

    Returns:
        pos_weight tensor of shape (14,) on the given device
    """
    log.info("Computing per-class positive weights from CSV (fast path)...")
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    N_pos = torch.zeros(NUM_CLASSES)
    for label_str in df["labels"]:
        for label in label_str.split("|"):
            label = label.strip()
            if label in class_to_idx:
                N_pos[class_to_idx[label]] += 1

    N        = len(df)
    N_pos    = N_pos.clamp(min=1)          # avoid div-by-zero for rare classes
    N_neg    = N - N_pos
    pos_weight = (N_neg / N_pos).to(device)

    for i, cls in enumerate(CLASSES):
        log.info(f"  {cls:<22s}: pos={int(N_pos[i]):5d}, neg={int(N_neg[i]):6d}, "
                 f"weight={pos_weight[i]:.2f}")

    return pos_weight


# ===========================================================================
# 4. TRANSFORMS
# ===========================================================================
def get_transforms(split: str) -> transforms.Compose:
    """
    Returns appropriate torchvision transform pipeline.

    Training:    RandomHorizontalFlip + RandomRotation(-10°..+10°) + ImageNet normalise
    Validation:  Only resize + ImageNet normalise (no augmentation)

    All images are already 224x224, but we include Resize for safety
    in case a handful of images have slightly different dimensions.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> float:
    """
    Run one full training epoch.

    Returns:
        avg_loss : mean BCEWithLogitsLoss over all batches
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    t0         = time.time()

    log.info(f"--- Epoch {epoch} | Training | {n_batches} batches ---")

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)      # slightly faster than zero_grad()

        logits = model(images)                      # (B, 14) raw logits
        loss   = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 20 batches so the terminal stays active
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            log.info(
                f"  Epoch {epoch} | Batch {batch_idx+1:4d}/{n_batches} | "
                f"Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s"
            )

    avg_loss = total_loss / n_batches
    return avg_loss


# ===========================================================================
# 6. VALIDATION LOOP
# ===========================================================================
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    metric:    MultilabelAUROC,
    device:    torch.device,
) -> tuple[float, float, list[float]]:
    """
    Run validation — compute loss and AUROC metrics.

    Returns:
        avg_loss    : mean BCEWithLogitsLoss over validation set
        macro_auroc : mean AUROC across all 14 classes
        per_cls_auc : list of 14 per-class AUROC scores
    """
    model.eval()
    metric.reset()

    total_loss = 0.0
    n_batches  = len(loader)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            # torchmetrics expects probabilities (0..1), not raw logits
            probs = torch.sigmoid(logits)
            metric.update(probs, labels.int())

    avg_loss    = total_loss / n_batches
    per_cls_auc = metric.compute().cpu().tolist()     # list of 14 AUROC values
    macro_auroc = float(np.mean(per_cls_auc))

    return avg_loss, macro_auroc, per_cls_auc


# ===========================================================================
# 7. MAIN — MLflow-wrapped orchestration
# ===========================================================================
def main(args):
    print("[DEBUG] Entered main()")
    # ----- Device selection ---------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Using device: {device}")
    print(f"[DEBUG] Device selected: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("[DEBUG] Loading datasets...")
    # ----- Full dataset — load with training transforms first ----------
    # Three-way split: 70% train / 10% val / 20% test
    #   - train : weight updates
    #   - val   : checkpoint selection (best Macro-AUROC)
    #   - test  : SEALED — never touched here; used only in evaluate_final.py
    full_train_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("train"))
    full_val_ds   = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))

    n_total = len(full_train_ds)
    n_test  = int(n_total * 0.20)   # 22,424 — sealed, never used here
    n_val   = int(n_total * 0.10)   # 11,212 — checkpoint selection only
    n_train = n_total - n_val - n_test  # 78,484 — training

    generator = torch.Generator().manual_seed(42)   # deterministic, reproducible
    train_indices, val_indices, _test_indices = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )
    # _test_indices intentionally unused here — evaluated only in evaluate_final.py

    # Subset wrappers to apply different transforms per split
    from torch.utils.data import Subset
    train_ds = Subset(full_train_ds, train_indices.indices)
    val_ds   = Subset(full_val_ds,   val_indices.indices)

    log.info(f"Split: {len(train_ds)} train / {len(val_ds)} val / {len(_test_indices)} test (SEALED)")

    # ----- DataLoaders --------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ----- Positive weights (computed on training split only) ----------
    # Read labels directly from the CSV DataFrame — no image I/O needed.
    train_df = full_train_ds.df.iloc[train_indices.indices].reset_index(drop=True)
    pos_weight = compute_pos_weight(train_df, device)

    # ----- Model --------------------------------------------------------
    model = ChestXRayModel(backbone_name=args.backbone, num_classes=NUM_CLASSES, pretrained=True).to(device)

    # ----- Loss, Optimizer, Scheduler -----------------------------------
    # Using Asymmetric Loss — handles extreme imbalance natively
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # CosineAnnealingLR decays LR smoothly from lr -> min_lr following a
    # cosine curve over T_max epochs. Better than ReduceLROnPlateau for
    # longer runs because it never gets stuck at a plateau.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,   # full cosine cycle over all epochs
        eta_min=1e-6,        # minimum LR at the end of the cycle
    )

    # ----- torchmetrics AUROC -------------------------------------------
    auroc_metric = MultilabelAUROC(
        num_labels=NUM_CLASSES,
        average=None,          # returns per-class scores; we'll average manually
        thresholds=None,
    ).to(device)

    # ----- MLflow run ---------------------------------------------------
    print("[DEBUG] Setting MLflow experiment...")
    mlflow.set_experiment(args.experiment_name)
    
    print(f"[DEBUG] Starting MLflow run: phase1_{args.backbone}_bs{args.batch_size}")
    with mlflow.start_run(run_name=f"phase1_{args.backbone}_bs{args.batch_size}"):
        print("[DEBUG] MLflow run started!")
        # Log hyperparameters
        mlflow.log_params({
            "backbone":     args.backbone,
            "pretrained":   True,
            "num_classes":  NUM_CLASSES,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "epochs":       args.epochs,
            "train_samples": len(train_ds),
            "val_samples":   len(val_ds),
            "device":        str(device),
        })

        best_macro_auroc = 0.0
        best_epoch       = -1

        log.info("=" * 60)
        log.info("Starting training...")
        log.info("=" * 60)

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # ---- Train -------------------------------------------------
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # ---- Validate ----------------------------------------------
            val_loss, macro_auroc, per_cls_auc = validate(
                model, val_loader, criterion, auroc_metric, device
            )

            epoch_time = time.time() - epoch_start

            # ---- LR scheduler step (CosineAnnealingLR needs no metric) ---
            scheduler.step()

            # ---- Logging -----------------------------------------------
            log.info(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Macro-AUROC: {macro_auroc:.4f}"
            )

            # Per-class AUROC printout
            log.info("  Per-class AUROC:")
            for cls_name, auc in zip(CLASSES, per_cls_auc):
                log.info(f"    {cls_name:<22s}: {auc:.4f}")

            # MLflow: log scalars
            mlflow.log_metrics({
                "train_loss":   train_loss,
                "val_loss":     val_loss,
                "macro_auroc":  macro_auroc,
                **{f"auroc_{cls}": auc for cls, auc in zip(CLASSES, per_cls_auc)},
            }, step=epoch)

            # ---- Checkpoint (best Macro-AUROC only) --------------------
            if macro_auroc > best_macro_auroc:
                best_macro_auroc = macro_auroc
                best_epoch       = epoch

                ckpt_path = CHECKPOINT_DIR / f"best_classifier_{args.backbone}.pth"
                torch.save(
                    {
                        "epoch":       epoch,
                        "backbone":    args.backbone,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "macro_auroc": best_macro_auroc,
                        "classes":     CLASSES,
                    },
                    ckpt_path,
                )
                log.info(
                    f"  >> New best! Macro-AUROC={best_macro_auroc:.4f} — "
                    f"checkpoint saved to {ckpt_path}"
                )
                mlflow.log_artifact(str(ckpt_path))

        # ---- Final summary ---------------------------------------------
        log.info("=" * 60)
        log.info(f"Training complete.")
        log.info(f"Best Macro-AUROC: {best_macro_auroc:.4f} at epoch {best_epoch}")
        log.info(f"Checkpoint: {CHECKPOINT_DIR / f'best_classifier_{args.backbone}.pth'}")
        log.info("=" * 60)

        mlflow.log_metrics({
            "best_macro_auroc": best_macro_auroc,
            "best_epoch":       best_epoch,
        })


# ===========================================================================
# 8. CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: EfficientNet Multi-Label Classifier — NIH ChestX-ray14"
    )
    parser.add_argument("--backbone",        type=str,   default="efficientnet_b3",
                        choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b3"],
                        help="Which EfficientNet backbone to use.")
    parser.add_argument("--epochs",          type=int,   default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size",      type=int,   default=64,
                        help="Batch size for train and val loaders (default: 64)")
    parser.add_argument("--lr",              type=float, default=1e-4,
                        help="AdamW learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay",    type=float, default=1e-2,
                        help="AdamW weight decay (default: 1e-2)")
    parser.add_argument("--num-workers",     type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--experiment-name", type=str,   default="ChestXRay14_Phase1",
                        help="MLflow experiment name")
    args = parser.parse_args()

    main(args)
