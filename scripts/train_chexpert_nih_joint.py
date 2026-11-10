"""
scripts/train_chexpert_nih_joint.py
===================================
Balanced Joint Fine-Tuning: CheXpert B1v2 + NIH ChestX-ray14
Targets weak classes (Pneumonia, Atelectasis, Consolidation) by borrowing clean
supervision from NIH-14 while preserving CheXpert performance on strong classes.

Features:
- Dual-stream balanced data loading (CheXpert batches + NIH batches).
- Masked loss over overlapping classes for NIH samples.
- Differential learning rates:
    - Target weak heads (Pneumonia, Atelectasis): lr = 1e-4
    - Strong heads (Cardiomegaly, Effusion, etc.): lr = 1e-6
    - Upper backbone & attention modules: lr = 1e-5
- Evaluates on CheXpert validation set after each epoch.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from scripts.train_chexpert_b1_v2 import (
    CheXpertLocalDatasetV2,
    TieredSamplerV2,
    make_patient_splits,
    CHEXPERT_14_LABELS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("joint_finetune")

# Mapping between CheXpert and NIH
OVERLAPPING_MAPPING = {
    # CheXpert label -> NIH label
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Cardiomegaly": "Cardiomegaly",
    "Pneumothorax": "Pneumothorax",
}

CHEX_TO_OVERLAP_IDX = {
    CHEXPERT_14_LABELS.index(chex_lbl): nih_lbl
    for chex_lbl, nih_lbl in OVERLAPPING_MAPPING.items()
}

# =============================================================================
# NIH Weak-Class Balanced Dataset
# =============================================================================

class NIHWeakClassDataset(Dataset):
    """
    NIH dataset targeted for balanced weak class fine-tuning.
    Provides images transformed to match model resolution and CheXpert-aligned labels.
    """
    def __init__(
        self,
        metadata_csv: Path,
        images_dir: Path,
        split_list_file: Optional[Path] = None,
        resolution: int = 224,
    ):
        self.images_dir = images_dir
        self.resolution = resolution
        
        df = pd.read_csv(metadata_csv)
        if split_list_file and split_list_file.exists():
            with open(split_list_file, "r") as f:
                valid_images = set(line.strip() for line in f if line.strip())
            df = df[df["Image Index"].isin(valid_images)].reset_index(drop=True)

        self.df = df
        
        # Build 14-dim CheXpert targets + active masks
        self.targets = []
        self.masks = []
        
        for _, row in df.iterrows():
            finding = str(row["Finding Labels"])
            target = np.zeros(len(CHEXPERT_14_LABELS), dtype=np.float32)
            mask = np.zeros(len(CHEXPERT_14_LABELS), dtype=np.float32)
            
            # Active only on overlapping classes
            for chex_idx, nih_lbl in CHEX_TO_OVERLAP_IDX.items():
                mask[chex_idx] = 1.0
                if nih_lbl in finding:
                    target[chex_idx] = 1.0
                    
            self.targets.append(target)
            self.masks.append(mask)
            
        self.targets = np.array(self.targets)
        self.masks = np.array(self.masks)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.images_dir / img_name
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_t = self.transform(img)
        except Exception:
            img_t = torch.zeros(3, self.resolution, self.resolution)
            
        return (
            img_t,
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.masks[idx]),
        )

    def get_sampling_weights(self) -> torch.Tensor:
        """
        Boost sampling probability for Pneumonia and Atelectasis positives
        so NIH batches provide strong, balanced signal.
        """
        pneu_idx = CHEXPERT_14_LABELS.index("Pneumonia")
        atel_idx = CHEXPERT_14_LABELS.index("Atelectasis")
        
        weights = np.ones(len(self.df), dtype=np.float32)
        is_pneu = self.targets[:, pneu_idx] == 1.0
        is_atel = self.targets[:, atel_idx] == 1.0
        
        # Pneumonia is rarer (~1.3k cases) -> give 10x weight
        weights[is_pneu] += 10.0
        # Atelectasis is common (~11k cases) -> give 3x weight
        weights[is_atel] += 3.0
        
        return torch.from_numpy(weights)


# =============================================================================
# Evaluation function
# =============================================================================

@torch.no_grad()
def evaluate_val_set(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_probs = []
    all_masks = []

    for images, targets, masks in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(images)
            probs = torch.sigmoid(logits)
        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())
        all_masks.append(masks.numpy())

    targets_all = np.concatenate(all_targets, axis=0)
    probs_all = np.concatenate(all_probs, axis=0)
    masks_all = np.concatenate(all_masks, axis=0)

    results = {}
    valid_aurocs = []
    valid_auprcs = []

    for i, lbl in enumerate(CHEXPERT_14_LABELS):
        obs = masks_all[:, i] > 0.5
        t = targets_all[obs, i]
        p = probs_all[obs, i]
        if len(np.unique(t)) < 2:
            results[f"auroc_{lbl}"] = float("nan")
            results[f"auprc_{lbl}"] = float("nan")
            continue
        try:
            auc = roc_auc_score(t, p)
            pr = average_precision_score(t, p)
            results[f"auroc_{lbl}"] = auc
            results[f"auprc_{lbl}"] = pr
            valid_aurocs.append(auc)
            valid_auprcs.append(pr)
        except Exception:
            results[f"auroc_{lbl}"] = float("nan")
            results[f"auprc_{lbl}"] = float("nan")

    results["macro_auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0
    results["macro_auprc"] = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
    return results


# =============================================================================
# Main Training Routine
# =============================================================================

def train_joint(
    checkpoint_path: str = "checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth",
    output_dir: str = "checkpoints/chexpert_nih_joint",
    epochs: int = 6,
    batch_size: int = 32,
    batches_per_epoch: int = 400,
    chex_ratio: float = 0.5,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Joint training on device: %s", device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Attach file handler for training log
    fh = logging.FileHandler(out_path / "joint_training.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    # Calibrated pos_weights for CheXpert 14 classes (giving high gradient push to rare classes)
    pos_weight = torch.tensor(
        [2.24, 3.63, 2.06, 1.02, 4.22, 1.80, 3.58, 8.26, 2.28, 2.72, 1.28, 6.51, 4.24, 0.95],
        device=device,
        dtype=torch.float32,
    )

    # 1. Load CheXpert checkpoint
    ckpt_file = PROJECT_ROOT / checkpoint_path
    log.info("Loading baseline checkpoint: %s", ckpt_file)
    ckpt = torch.load(ckpt_file, map_location=device)
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    # Enable spatial softmax sharpening and diversity loss
    model_cfg["att_norm"] = "spatial_softmax"
    model_cfg["attn_temperature"] = 0.5
    model_cfg["attn_regularizer_weight"] = 0.05
    model_cfg["diversity_reg_weight"] = 0.05

    model = CheXpertB1V2Model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)

    # 2. Setup CheXpert data
    log.info("Preparing CheXpert dataset (with uncertain_policy=smooth for soft probabilistic labels)...")
    index_csv = PROJECT_ROOT / "data/chexpert_subset/chexpert_materialized_index.csv"
    chex_df = pd.read_csv(index_csv)
    train_df, val_df, _ = make_patient_splits(chex_df, 0.1, 0.1, seed)

    chex_train_ds = CheXpertLocalDatasetV2(
        train_df,
        split="train",
        cfg={
            "uncertain_policy": "smooth",  # Map -1.0 to 0.70 so model is not penalized
            "cache_dir": str(PROJECT_ROOT / "data/chexpert_cache"),
            "model_input_resolution": model_cfg.get("input_resolution", 224),
        },
    )
    chex_val_ds = CheXpertLocalDatasetV2(
        val_df,
        split="val",
        cfg={
            "uncertain_policy": "u_zeros",
            "cache_dir": str(PROJECT_ROOT / "data/chexpert_cache"),
            "model_input_resolution": model_cfg.get("input_resolution", 224),
        },
    )

    chex_sampler = TieredSamplerV2(
        df=train_df,
        pool_counts=[14, 12, 6],
        labels=CHEXPERT_14_LABELS,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        intra_pool_mode="sqrt_freq",
        seed=seed,
    )
    chex_loader = DataLoader(
        chex_train_ds,
        batch_sampler=chex_sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        chex_val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Setup NIH dataset
    log.info("Preparing NIH weak-class balanced dataset...")
    nih_meta = PROJECT_ROOT / "data/nih_cxr14/metadata/Data_Entry_2017.csv"
    nih_imgs = PROJECT_ROOT / "data/nih_cxr14/images/images-224"
    nih_train_list = PROJECT_ROOT / "data/nih_cxr14/metadata/train_val_list_NIH.txt"

    nih_ds = NIHWeakClassDataset(
        metadata_csv=nih_meta,
        images_dir=nih_imgs,
        split_list_file=nih_train_list,
        resolution=model_cfg.get("input_resolution", 224),
    )
    nih_weights = nih_ds.get_sampling_weights()
    nih_sampler = WeightedRandomSampler(
        weights=nih_weights,
        num_samples=batches_per_epoch * batch_size,
        replacement=True,
    )
    nih_loader = DataLoader(
        nih_ds,
        batch_size=batch_size,
        sampler=nih_sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Optimizer with Differential Learning Rates
    pneu_idx = CHEXPERT_14_LABELS.index("Pneumonia")
    atel_idx = CHEXPERT_14_LABELS.index("Atelectasis")
    con_idx = CHEXPERT_14_LABELS.index("Consolidation")
    weak_indices = {pneu_idx, atel_idx, con_idx}

    param_groups = []
    # Lower backbone (blocks 0..4) frozen
    for param in model.backbone.features[:5].parameters():
        param.requires_grad = False

    # Upper backbone (blocks 5..8): lr = 5e-6
    upper_params = list(model.backbone.features[5:].parameters())
    param_groups.append({"params": upper_params, "lr": 5e-6, "weight_decay": 1e-4})

    # MSFB & Attention fusion: lr = 1e-5
    fusion_params = (
        list(model.msfb.parameters())
        + list(model.channel_attn.parameters())
        + list(model.spatial_attn.parameters())
        + list(model.lateral_conv.parameters())
        + list(model.dag.parameters())
    )
    param_groups.append({"params": fusion_params, "lr": 1e-5, "weight_decay": 1e-4})

    # Weak heads: active learning rate lr = 1e-4
    weak_head_params = []
    for idx in weak_indices:
        weak_head_params.extend(list(model.classifiers[idx].parameters()))
    param_groups.append({"params": weak_head_params, "lr": 1e-4, "weight_decay": 1e-4})

    # Strong heads: tiny learning rate lr = 1e-6 (protect against forgetting)
    strong_head_params = []
    for idx in range(len(CHEXPERT_14_LABELS)):
        if idx not in weak_indices:
            strong_head_params.extend(list(model.classifiers[idx].parameters()))
    param_groups.append({"params": strong_head_params, "lr": 1e-6, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Initial validation baseline
    log.info("Running baseline validation check before joint training...")
    init_metrics = evaluate_val_set(model, val_loader, device)
    log.info(
        "Initial Val: Macro AUROC=%.4f | Pneumonia=%.4f | Atelectasis=%.4f",
        init_metrics["macro_auroc"],
        init_metrics.get("auroc_Pneumonia", 0.0),
        init_metrics.get("auroc_Atelectasis", 0.0),
    )

    best_pneu_auc = init_metrics.get("auroc_Pneumonia", 0.0)
    best_macro_auc = init_metrics["macro_auroc"]

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        # Freeze BN in backbone to preserve statistics
        model.backbone.eval()

        t0 = time.time()
        chex_iter = iter(chex_loader)
        nih_iter = iter(nih_loader)
        running_loss = 0.0

        for step in range(batches_per_epoch):
            # Interleave streams: 50% CheXpert, 50% NIH
            if np.random.rand() < chex_ratio:
                try:
                    images, targets, masks = next(chex_iter)
                except StopIteration:
                    chex_iter = iter(chex_loader)
                    images, targets, masks = next(chex_iter)
            else:
                try:
                    images, targets, masks = next(nih_iter)
                except StopIteration:
                    nih_iter = iter(nih_loader)
                    images, targets, masks = next(nih_iter)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, reg_loss = model(images)
                # Masked BCE loss with calibrated pos_weight
                bce = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight, reduction="none"
                )
                loss = (bce * masks).sum() / masks.sum().clamp(min=1.0) + reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                log.info("Epoch %02d [%d/%d] loss=%.4f", epoch, step + 1, batches_per_epoch, running_loss / (step + 1))

        # Epoch evaluation on CheXpert validation set
        val_metrics = evaluate_val_set(model, val_loader, device)
        elapsed = time.time() - t0
        log.info(
            "Epoch %02d completed in %.0fs | Loss=%.4f | Macro AUROC=%.4f (Pneumonia=%.4f, Atelectasis=%.4f, Consolidation=%.4f)",
            epoch,
            elapsed,
            running_loss / batches_per_epoch,
            val_metrics["macro_auroc"],
            val_metrics.get("auroc_Pneumonia", 0.0),
            val_metrics.get("auroc_Atelectasis", 0.0),
            val_metrics.get("auroc_Consolidation", 0.0),
        )

        # Checkpoint saving if target classes or macro improved
        current_pneu = val_metrics.get("auroc_Pneumonia", 0.0)
        current_macro = val_metrics["macro_auroc"]
        
        save_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": val_metrics,
            "cfg": cfg,
        }
        
        if current_pneu > best_pneu_auc or current_macro >= best_macro_auc:
            best_pneu_auc = max(best_pneu_auc, current_pneu)
            best_macro_auc = max(best_macro_auc, current_macro)
            save_path = out_path / "chexpert_nih_joint_best.pth"
            torch.save(save_ckpt, save_path)
            log.info("✓ New best joint model saved to %s", save_path)

    log.info("Joint fine-tuning complete! Best Pneumonia AUROC: %.4f | Best Macro AUROC: %.4f", best_pneu_auc, best_macro_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced Joint Fine-Tuning CheXpert + NIH")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth")
    parser.add_argument("--output_dir", type=str, default="checkpoints/chexpert_nih_joint")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches_per_epoch", type=int, default=400)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        train_joint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            epochs=1,
            batch_size=8,
            batches_per_epoch=4,
        )
    else:
        train_joint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
        )
