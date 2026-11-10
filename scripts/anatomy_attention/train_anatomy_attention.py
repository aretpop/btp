"""
train_anatomy_attention.py
===========================
Training script for M1 → M3 ablation stages.

Usage:
    python scripts/anatomy_attention/train_anatomy_attention.py \
        --config scripts/anatomy_attention/configs/M3_b1.json

All checkpoints saved to: checkpoints/anatomy_attention/<experiment_name>/
MLflow experiment: ChestXRay14_AnatomyAttention
"""

import sys
import json
import argparse
import logging
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_phase1_classifier import (
    CXRDataset, get_transforms, CSV_PATH, IMAGES_DIR,
    CHECKPOINT_DIR, CLASSES, NUM_CLASSES, compute_pos_weight
)
from anatomy_attention.anatomy_attention_model import AnatomyDiseaseAttentionModel
from anatomy_attention.losses import CombinedLoss

import mlflow
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANATOMY_CKPT_DIR = CHECKPOINT_DIR / "anatomy_attention"
ANATOMY_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(batch_size: int, num_workers: int = 0):
    """Reuse the exact same 70/10/20 split with seed=42 as Phase 1."""
    full_train = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("train"))
    full_val   = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))

    n_total = len(full_train)
    n_test  = int(n_total * 0.20)
    n_val   = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(range(n_total), [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(
        Subset(full_train, train_idx.indices),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx.indices),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    train_df = full_train.df.iloc[train_idx.indices].reset_index(drop=True)
    return train_loader, val_loader, train_df


def validate_epoch(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, disease_atts = model(images)
            loss_dict = criterion(logits, labels, disease_atts)
            total_loss += loss_dict["total"].item()
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    per_class = []
    for i in range(NUM_CLASSES):
        try:
            per_class.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            per_class.append(0.5)

    macro_auroc = float(np.mean(per_class))
    return total_loss / len(loader), macro_auroc, per_class


def main(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader, train_df = build_dataloaders(
        cfg.get("batch_size", 32)
    )

    # Build model
    model = AnatomyDiseaseAttentionModel(cfg).to(device)
    counts = model.param_count()
    log.info(f"Parameters — total: {counts['total']:,} | trainable: {counts['trainable']:,} | frozen: {counts['frozen']:,}")

    # Only train the non-frozen attention parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 20), eta_min=1e-6)

    criterion = CombinedLoss(
        lambda_anatomy=cfg.get("lambda_anatomy", 0.0),
        lambda_diversity=cfg.get("lambda_diversity", 0.0),
        asl_gamma_neg=cfg.get("asl_gamma_neg", 4),
        asl_gamma_pos=cfg.get("asl_gamma_pos", 0),
        asl_clip=cfg.get("asl_clip", 0.05),
    )

    exp_name = cfg.get("experiment", "M3_b1")
    save_dir = ANATOMY_CKPT_DIR / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoint
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ChestXRay14_AnatomyAttention")

    best_auroc = 0.0

    with mlflow.start_run(run_name=exp_name):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})
        mlflow.log_params({"trainable_params": counts["trainable"], "frozen_params": counts["frozen"]})

        for epoch in range(1, cfg.get("epochs", 20) + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                logits, _, disease_atts = model(images)
                loss_dict = criterion(logits, labels, disease_atts)
                loss_dict["total"].backward()

                optimizer.step()
                train_loss += loss_dict["total"].item()

            train_loss /= len(train_loader)

            # --- Validate ---
            val_loss, macro_auroc, per_class_auroc = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()

            log.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val AUROC: {macro_auroc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auroc": macro_auroc,
                **{f"auroc_{cls}": auc for cls, auc in zip(CLASSES, per_class_auroc)},
            }, step=epoch)

            # --- Checkpoint ---
            if macro_auroc > best_auroc:
                best_auroc = macro_auroc
                ckpt_path = save_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "config": cfg,
                    "model_state": model.state_dict(),
                    "macro_auroc": macro_auroc,
                    "per_class_auroc": {c: a for c, a in zip(CLASSES, per_class_auroc)},
                }, ckpt_path)
                log.info(f"  >> New best AUROC={best_auroc:.4f} → saved to {ckpt_path}")

        log.info(f"Training complete. Best Val AUROC: {best_auroc:.4f}")
        mlflow.log_metric("best_val_auroc", best_auroc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    # Allow CLI overrides of any config key
    parser.add_argument("--backbone",            type=str)
    parser.add_argument("--checkpoint-path",     type=str)
    parser.add_argument("--epochs",              type=int)
    parser.add_argument("--batch-size",          type=int)
    parser.add_argument("--lr",                  type=float)
    parser.add_argument("--lambda-anatomy",      type=float)
    parser.add_argument("--lambda-diversity",    type=float)
    parser.add_argument("--no-multiscale",       action="store_true")
    parser.add_argument("--no-channel-att",      action="store_true")
    parser.add_argument("--no-spatial-att",      action="store_true")
    parser.add_argument("--no-disease-att",      action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Apply CLI overrides
    if args.backbone:           cfg["backbone"]            = args.backbone
    if args.checkpoint_path:    cfg["checkpoint_path"]     = args.checkpoint_path
    if args.epochs:             cfg["epochs"]              = args.epochs
    if args.batch_size:         cfg["batch_size"]          = args.batch_size
    if args.lr:                 cfg["lr"]                  = args.lr
    if args.lambda_anatomy:     cfg["lambda_anatomy"]      = args.lambda_anatomy
    if args.lambda_diversity:   cfg["lambda_diversity"]    = args.lambda_diversity
    if args.no_multiscale:      cfg["use_multiscale"]      = False
    if args.no_channel_att:     cfg["use_channel_attention"] = False
    if args.no_spatial_att:     cfg["use_spatial_attention"] = False
    if args.no_disease_att:     cfg["use_disease_attention"] = False

    main(cfg)
