"""
scripts/train_chexpert.py
-------------------------
Train a DenseNet-121 classifier on CheXpert+ using streaming images from Redivis.
- Downloads images on-the-fly (no 720 GB local storage needed)
- 5 GB rolling LRU disk cache
- Stratified sampler ensures uniform label coverage per epoch
- BCEWithLogitsLoss with pos_weight for class imbalance

Usage:
    python scripts/train_chexpert.py --config configs/chexpert.yaml
    python scripts/train_chexpert.py --config configs/chexpert.yaml --subset 1000  # quick test
"""

import os, sys, argparse, time, logging
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from datasets.chexpert_streaming import StreamingCheXpertDataset, MultiLabelStratifiedSampler

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int, pretrained: bool, dropout: float):
    """Build classification model from torchvision backbone."""
    if backbone == "densenet121":
        model = tv_models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet50":
        model = tv_models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif backbone == "vit_b_16":
        model = tv_models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.heads.head.in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_targets, label_names):
    """Compute per-label AUC and mean AUC (approximate, no sklearn needed)."""
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        preds_np = torch.sigmoid(torch.tensor(all_preds)).numpy()
        targets_np = torch.tensor(all_targets).numpy()
        aucs = {}
        for i, name in enumerate(label_names):
            col_t = targets_np[:, i]
            col_p = preds_np[:, i]
            if col_t.sum() > 0 and (1 - col_t).sum() > 0:
                aucs[name] = roc_auc_score(col_t, col_p)
        mean_auc = sum(aucs.values()) / len(aucs) if aucs else 0.0
        return aucs, mean_auc
    except ImportError:
        return {}, 0.0


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            elapsed = time.time() - t0
            imgs_done = (batch_idx + 1) * loader.batch_size
            log.info(
                f"  Epoch {epoch} | batch {batch_idx:4d} | "
                f"loss={loss.item():.4f} | "
                f"{imgs_done} imgs in {elapsed:.0f}s"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_names):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(logits.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    aucs, mean_auc = compute_metrics(all_preds, all_targets, label_names)
    return avg_loss, mean_auc, aucs


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chexpert.yaml")
    parser.add_argument("--subset", type=int, default=None,
                        help="Override train_subset_n for a quick test")
    parser.add_argument("--api-token", default=None,
                        help="Redivis API token (or set REDIVIS_API_TOKEN env var)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Setup ─────────────────────────────────────────────────────────────────
    api_token = (
        args.api_token
        or os.environ.get("REDIVIS_API_TOKEN")
        or "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    label_names = cfg["labeler"]["label_set"]
    train_cfg   = cfg["training"]
    ds_cfg      = cfg["dataset"]
    sampler_cfg = cfg["sampler"]

    train_subset = args.subset or ds_cfg.get("train_subset_n")
    seed         = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Building streaming datasets...")

    train_ds = StreamingCheXpertDataset(
        index_csv   = ds_cfg["redivis"]["tables"]["train_index"],
        api_token   = api_token,
        table_ref   = ds_cfg["redivis"]["tables"]["train"],
        cache_dir   = ds_cfg["cache"]["dir"],
        cache_size_gb = ds_cfg["cache"]["size_gb"],
        frontal_only  = ds_cfg["frontal_only"],
        uncertainty_policy = ds_cfg["uncertainty_policy"],
    )

    valid_ds = StreamingCheXpertDataset(
        index_csv   = ds_cfg["redivis"]["tables"]["valid_index"],
        api_token   = api_token,
        table_ref   = ds_cfg["redivis"]["tables"]["valid"],
        cache_dir   = ds_cfg["cache"]["dir"],
        cache_size_gb = ds_cfg["cache"]["size_gb"],
        frontal_only  = ds_cfg["frontal_only"],
        uncertainty_policy = ds_cfg["uncertainty_policy"],
    )

    log.info(f"Train: {len(train_ds):,} | Valid: {len(valid_ds):,}")

    # ── Sampler ───────────────────────────────────────────────────────────────
    num_samples = train_subset or sampler_cfg["num_samples_per_epoch"]
    log.info(f"Stratified sampler: {num_samples} samples/epoch")

    train_sampler = MultiLabelStratifiedSampler(
        df          = train_ds.df,
        num_samples = num_samples,
        seed        = seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = train_cfg["batch_size"],
        sampler     = train_sampler,
        num_workers = train_cfg["num_workers"],
        pin_memory  = device.type == "cuda",
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size  = train_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = train_cfg["num_workers"],
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Building model: {train_cfg['backbone']}")
    model = build_model(
        backbone    = train_cfg["backbone"],
        num_classes = len(label_names),
        pretrained  = train_cfg["pretrained"],
        dropout     = train_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params/1e6:.1f}M")

    # ── Loss (weighted BCE for class imbalance) ────────────────────────────────
    pos_weight = train_ds.get_label_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg["lr"],
        weight_decay = train_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    ) if train_cfg["scheduler"] == "cosine" else None

    scaler = GradScaler() if (train_cfg["amp"] and device.type == "cuda") else None

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    # ── Training ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Starting training: {train_cfg['epochs']} epochs, "
             f"batch={train_cfg['batch_size']}, device={device}")
    log.info("=" * 60)

    for epoch in range(1, train_cfg["epochs"] + 1):
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        val_loss, val_auc, aucs = evaluate(
            model, valid_loader, criterion, device, label_names
        )

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t_epoch
        log.info(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mean_AUC={val_auc:.4f} | {elapsed:.0f}s | "
            f"cache={train_ds.cache.num_files}files/{train_ds.cache.size_gb:.2f}GB"
        )

        # Per-label AUCs
        if aucs:
            for name, auc in sorted(aucs.items(), key=lambda x: -x[1]):
                log.info(f"    {name:30s}: AUC={auc:.4f}")

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = ckpt_dir / "best.pth"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_auc":    val_auc,
                "aucs":       aucs,
                "label_names": label_names,
            }, ckpt_path)
            log.info(f"    >> Best model saved (AUC={val_auc:.4f}): {ckpt_path}")

        # Periodic checkpoint
        if epoch % train_cfg["save_every_n_epochs"] == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                ckpt_dir / f"epoch_{epoch:03d}.pth"
            )

    log.info(f"Training complete. Best val AUC: {best_auc:.4f}")
    log.info(f"Best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
