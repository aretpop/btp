"""
scripts/transfer_chexpert.py
==============================
Staged transfer learning: NIH ChestX-ray14 → CheXpert Plus.

Stage A: Freeze EfficientNet backbone, train new 14-class CheXpert head.
Stage B: Load Stage A best, selectively unfreeze last N EfficientNet blocks.

Uses the streaming dataset (src/datasets/chexpert_streaming.py) so no
720GB download is required.

Usage:
  # Stage A (frozen backbone):
  python scripts/transfer_chexpert.py --stage A --backbone efficientnet_b1

  # Stage B (unfreeze last 3 blocks):
  python scripts/transfer_chexpert.py --stage B --backbone efficientnet_b1 \
      --stage-a-ckpt checkpoints/chexpert_transfer/stageA_efficientnet_b1.pth \
      --unfreeze-blocks 3

  # Quick smoke test:
  python scripts/transfer_chexpert.py --stage A --backbone efficientnet_b1 --smoke-test
"""

import os, sys, json, time, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_phase1_classifier import ChestXRayModel, NUM_CLASSES as NIH_CLASSES, AsymmetricLoss
from train_phase3_regional_v3 import (
    RegionalAttentionXRayModel, MultiScaleFeatureBlock,
    ChannelAttention, SpatialAttention, AnatomicalAttention, DiseaseAttentionGenerator,
    BLOCK5_CHANNELS, BLOCK8_CHANNELS
)
from datasets.chexpert_streaming import StreamingCheXpertDataset, MultiLabelStratifiedSampler

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CKPT_TRANSFER  = PROJECT_ROOT / "checkpoints" / "chexpert_transfer"
CKPT_TRANSFER.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

# ── CheXpert 11-class label set (removed: Atelectasis, Enlarged Cardiomediastinum, Fracture) ─
CHEXPERT_LABELS = [
    "No Finding",          # 0
    "Cardiomegaly",        # 1
    "Lung Opacity",        # 2
    "Lung Lesion",         # 3
    "Edema",               # 4
    "Consolidation",       # 5
    "Pneumonia",           # 6
    "Pneumothorax",        # 7
    "Pleural Effusion",    # 8
    "Pleural Other",       # 9
    "Support Devices",     # 10
]
NUM_CHEXPERT = len(CHEXPERT_LABELS)

# NIH label indices that correspond to CheXpert labels (for weight transfer)
# Used to initialize the new classification head with NIH-learned weights
NIH_TO_CHEXPERT_HEAD_IDX = {
    # (nih_idx, chexpert_idx)
    # NIH CLASSES (alphabetical):
    # 0: Atelectasis (removed from CheXpert head)
    # 1: Cardiomegaly(1) → CheXpert Cardiomegaly(1)
    # 2: Consolidation(2) → CheXpert Consolidation(5)
    # 3: Edema(3) → CheXpert Edema(4)
    # 4: Effusion(4) → CheXpert Pleural Effusion(8)
    # 12: Pneumonia(12) → CheXpert Pneumonia(6)
    # 13: Pneumothorax(13) → CheXpert Pneumothorax(7)
    1: 1,
    2: 5,
    3: 4,
    4: 8,
    12: 6,
    13: 7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Transfer model: CheXpert head on NIH backbone
# ─────────────────────────────────────────────────────────────────────────────
class CheXpertTransferModel(nn.Module):
    """
    Transfer learning wrapper.

    Stage A: Frozen NIH backbone → new CheXpert 14-class head.
    Stage B: Selectively unfrozen backbone layers + same head.

    Can optionally use the Phase 3 regional attention modules.
    """
    def __init__(self, nih_backbone: nn.Module, backbone_name: str,
                 use_attention: bool = True, num_classes: int = NUM_CHEXPERT):
        super().__init__()
        import copy
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.backbone = copy.deepcopy(nih_backbone)
        # Start fully frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        b8_ch = BLOCK8_CHANNELS[backbone_name]
        b5_ch = BLOCK5_CHANNELS[backbone_name]
        feat_dim = 256

        if use_attention:
            # Lightweight attention head (same as Phase 3)
            self.refine      = MultiScaleFeatureBlock(b8_ch, feat_dim)
            self.channel_att = ChannelAttention(feat_dim)
            self.spatial_att = SpatialAttention()
            self.disease_att = DiseaseAttentionGenerator(b5_ch, num_classes, att_norm="sigmoid")
            self.classifiers = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(num_classes)])
        else:
            # Simple linear head (for ablation)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(b8_ch, num_classes),
            )

    def train(self, mode=True):
        """Ensure frozen blocks stay in eval (preserve BN running stats)."""
        super().train(mode)
        # Keep frozen backbone blocks in eval mode
        for block_idx, block in enumerate(self._get_blocks()):
            frozen = all(not p.requires_grad for p in block.parameters())
            if frozen:
                block.eval()
        return self

    def _get_blocks(self):
        if hasattr(self.backbone, "backbone"):
            return list(self.backbone.backbone.features)
        return list(self.backbone.features)

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last N feature blocks of the backbone."""
        blocks = self._get_blocks()
        total = len(blocks)
        log.info(f"Unfreezing last {n}/{total} blocks: blocks {total-n}..{total-1}")
        for i, block in enumerate(blocks):
            if i >= total - n:
                for p in block.parameters():
                    p.requires_grad = True
                log.info(f"  Block {i}: unfrozen ({sum(p.numel() for p in block.parameters()):,} params)")

    def _get_features(self, x):
        blocks = self._get_blocks()
        z = x
        f5 = None
        for i, block in enumerate(blocks):
            z = block(z)
            if i == 5:
                f5 = z
        return f5, z   # f5: [B,b5,14,14]  z: [B,b8,7,7]

    def forward(self, x, return_attention=False):
        f5, f8 = self._get_features(x)

        if not self.use_attention:
            return self.head(f8)

        f_ref = self.refine(f8)
        f_ref = self.channel_att(f_ref)
        f_ref = self.spatial_att(f_ref)
        f_shared = F.interpolate(f_ref, size=(14, 14), mode="bilinear", align_corners=False)

        disease_maps = self.disease_att(f5)   # [B, num_classes, 14, 14]

        H, W = f_shared.shape[2], f_shared.shape[3]
        f_flat = f_shared.flatten(2)                    # [B, feat_dim, H*W]
        d_flat = disease_maps.flatten(2)                # [B, num_classes, H*W]
        z = torch.bmm(d_flat, f_flat.transpose(1, 2)) / (H * W)  # [B, num_classes, feat_dim]

        logits = torch.cat([self.classifiers[d](z[:, d, :]) for d in range(self.num_classes)], dim=1)

        if return_attention:
            return logits, disease_maps
        return logits


def initialize_head_from_nih(model: CheXpertTransferModel, nih_ckpt_path: str, device):
    """
    Initialize CheXpert head from NIH checkpoint.
    If checkpoint is from Phase 3 (Regional Attention):
      - Loads shared refinement & attention modules directly
      - Copies matching classifier head weights [1, 256] and biases [1]
    If checkpoint is from Phase 1 (Standard Linear):
      - Copies class bias priors (log-odds) for overlapping diseases
    """
    nih_ckpt = torch.load(nih_ckpt_path, map_location=device, weights_only=False)
    state = nih_ckpt.get("model_state", nih_ckpt)

    if not model.use_attention:
        log.info("Simple head: skipping class-specific weight transfer")
        return

    is_phase3 = any("refine" in k or "disease_att" in k for k in state.keys())
    n_transferred = 0

    if is_phase3:
        # Transfer shared feature refinement and attention modules
        own_state = model.state_dict()
        for k, v in state.items():
            if any(m in k for m in ["refine", "channel_att", "spatial_att"]):
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k].copy_(v)
        log.info("Transferred Phase 3 attention modules (refine, channel_att, spatial_att)")

        # Transfer overlapping classifier heads
        for nih_idx, chex_idx in NIH_TO_CHEXPERT_HEAD_IDX.items():
            w_key = f"classifiers.{nih_idx}.weight"
            b_key = f"classifiers.{nih_idx}.bias"
            if w_key in state and chex_idx < len(model.classifiers):
                model.classifiers[chex_idx].weight.data.copy_(state[w_key])
                model.classifiers[chex_idx].bias.data.copy_(state[b_key])
                n_transferred += 1
                log.info(f"  Transferred Phase 3 head: NIH[{nih_idx}] → CheXpert[{chex_idx}] ({CHEXPERT_LABELS[chex_idx]})")
    else:
        # Phase 1: Transfer bias prior log-odds for overlapping diseases
        if "backbone.classifier.1.bias" in state:
            nih_b = state["backbone.classifier.1.bias"]
            for nih_idx, chex_idx in NIH_TO_CHEXPERT_HEAD_IDX.items():
                if chex_idx < len(model.classifiers):
                    model.classifiers[chex_idx].bias.data.copy_(nih_b[nih_idx].unsqueeze(0))
                    n_transferred += 1
                    log.info(f"  Transferred Phase 1 bias: NIH[{nih_idx}] → CheXpert[{chex_idx}] ({CHEXPERT_LABELS[chex_idx]})")

    log.info(f"Head weight transfer: {n_transferred}/{len(NIH_TO_CHEXPERT_HEAD_IDX)} matching classes initialized from NIH")


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
def masked_bce_loss(logits, labels, uncertain_mask=None):
    """
    BCEWithLogitsLoss that ignores uncertain (-1) labels.
    uncertain_mask: boolean tensor [B, C] where True = uncertain, exclude from loss.
    """
    loss_per = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if uncertain_mask is not None:
        loss_per = loss_per * (~uncertain_mask).float()
        n_valid = (~uncertain_mask).float().sum().clamp(min=1)
        return loss_per.sum() / n_valid
    return loss_per.mean()


@torch.no_grad()
def evaluate(model, loader, device, label_names):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_clean = labels.clamp(0, 1)   # treat uncertain (-1) as 0 for eval
            with torch.amp.autocast('cuda'):
                logits = model(images)
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(labels_clean.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    per_cls, valid_cls = [], []
    for i, name in enumerate(label_names):
        col = labels[:, i]
        if col.sum() > 0 and (1 - col).sum() > 0:
            try:
                per_cls.append(roc_auc_score(col, probs[:, i]))
                valid_cls.append(name)
            except Exception:
                per_cls.append(0.5)
        else:
            per_cls.append(float("nan"))

    valid_aucs = [a for a in per_cls if not np.isnan(a)]
    macro = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return macro, per_cls


def train_stage(model, train_loader, val_loader, optimizer, scheduler,
                cfg, device, save_path, label_names, stage_name):
    """Generic training loop for both Stage A and B."""
    criterion = AsymmetricLoss(
        gamma_neg=cfg.get("asl_gamma_neg", 4),
        gamma_pos=cfg.get("asl_gamma_pos", 0),
        clip=cfg.get("asl_clip", 0.05),
    )
    scaler = torch.amp.GradScaler('cuda') if (cfg.get("amp", True) and device.type == "cuda") else None
    best_auroc = 0.0
    patience_count = 0
    patience = cfg.get("early_stopping_patience", 5)

    if hasattr(val_loader.dataset, "prefetch"):
        log.info(f"Prefetching all {len(val_loader.dataset)} validation images in parallel...")
        val_loader.dataset.prefetch(range(len(val_loader.dataset)), max_workers=12)

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        if hasattr(train_loader.dataset, "prefetch") and hasattr(train_loader, "sampler"):
            if hasattr(train_loader.sampler, "generate_indices"):
                # MultiLabelStratifiedSampler path (legacy)
                sampler_indices = train_loader.sampler.generate_indices(seed=cfg.get("seed", 42) + epoch)
            elif hasattr(train_loader.sampler, "generator") and train_loader.sampler.generator is not None:
                # RandomSampler path: re-seed per epoch so prefetch & DataLoader see same indices
                epoch_seed = cfg.get("seed", 42) + epoch
                train_loader.sampler.generator.manual_seed(epoch_seed)
                sampler_indices = list(train_loader.sampler)   # snapshot this epoch's indices
                # Re-seed again so DataLoader's own __iter__ produces the same sequence
                train_loader.sampler.generator.manual_seed(epoch_seed)
            else:
                sampler_indices = list(range(len(train_loader.dataset)))
            log.info(f"Prefetching {len(sampler_indices)} training images for epoch {epoch} in parallel...")
            train_loader.dataset.prefetch(sampler_indices, max_workers=12)

        n_batches = len(train_loader)
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).clamp(0, 1)   # treat -1 as 0

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 25 == 0 or (step + 1) == n_batches:
                log.info(f"  [{stage_name}] Ep {epoch} Step {step+1:3d}/{n_batches} | loss={loss.item():.4f}")

        if scheduler:
            scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        val_auroc, per_cls = evaluate(model, val_loader, device, label_names)
        elapsed = time.time() - t0

        log.info(f"[{stage_name}] Epoch {epoch:3d}/{cfg.get('epochs',20)} | "
                 f"train_loss={train_loss:.4f} | val_AUROC={val_auroc:.4f} | {elapsed:.0f}s")

        for name, auc in zip(label_names, per_cls):
            if not np.isnan(auc):
                log.info(f"    {name:<30s}: {auc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "stage":       stage_name,
                "backbone":    model.backbone_name,
                "cfg":         cfg,
                "model_state": model.state_dict(),
                "val_auroc":   val_auroc,
                "per_cls_auroc": per_cls,
                "label_names": label_names,
            }, save_path)
            log.info(f"  >> Best saved: AUROC={val_auroc:.4f} → {save_path.name}")
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info(f"Early stopping (no improvement for {patience} epochs)")
                break

    log.info(f"[{stage_name}] Done. Best Val AUROC: {best_auroc:.4f}")
    return best_auroc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      required=True, choices=["A", "B"])
    parser.add_argument("--backbone",   default="efficientnet_b1",
                        choices=list(BLOCK5_CHANNELS.keys()))
    parser.add_argument("--stage-a-ckpt", default=None,
                        help="Required for Stage B: path to Stage A checkpoint")
    parser.add_argument("--unfreeze-blocks", type=int, default=3,
                        help="Stage B: number of backbone blocks to unfreeze (from end)")
    parser.add_argument("--no-attention", action="store_true",
                        help="Use simple linear head instead of attention modules")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with tiny subset to verify pipeline")
    parser.add_argument("--nih-checkpoint", default=None,
                        help="Source NIH checkpoint to transfer from (defaults to best Phase 3 or Phase 1)")
    parser.add_argument("--api-token", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Stage: {args.stage} | Backbone: {args.backbone}")

    api_token = (args.api_token or os.environ.get("REDIVIS_API_TOKEN")
                 or "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")

    cfg = {
        "backbone":       args.backbone,
        "use_attention":  not args.no_attention,
        "epochs":         5 if args.smoke_test else 20,
        "batch_size":     8 if args.smoke_test else 16,
        "lr_head":        1e-3,
        "lr_backbone":    1e-5,
        "weight_decay":   1e-4,
        "asl_gamma_neg":  4,
        "asl_gamma_pos":  0,
        "asl_clip":       0.05,
        "amp":            True,
        "seed":           42,
        "early_stopping_patience": 5,
        "num_samples_per_epoch":   500 if args.smoke_test else 8000,
    }

    # ── Load NIH base model ───────────────────────────────────────────────────
    if args.nih_checkpoint:
        transfer_source_path = args.nih_checkpoint
    else:
        # Check if Phase 3 M2B checkpoint exists first
        m2b_candidate = PROJECT_ROOT / "checkpoints" / "phase3_regional" / f"M2B_upsample14_{args.backbone[-2:]}.pth"
        if m2b_candidate.exists():
            transfer_source_path = str(m2b_candidate)
        else:
            transfer_source_path = str(PROJECT_ROOT / "checkpoints" / f"best_classifier_{args.backbone}.pth")

    # Load base EfficientNet backbone
    phase1_path = str(PROJECT_ROOT / "checkpoints" / f"best_classifier_{args.backbone}.pth")
    nih_ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
    nih_base = ChestXRayModel(backbone_name=args.backbone, num_classes=NIH_CLASSES, pretrained=False)
    nih_base.load_state_dict(nih_ckpt["model_state"])
    log.info(f"NIH base backbone loaded from: {phase1_path}")
    log.info(f"Transfer learning source: {transfer_source_path}")

    # ── Build transfer model ──────────────────────────────────────────────────
    model = CheXpertTransferModel(
        nih_base, args.backbone,
        use_attention=not args.no_attention,
        num_classes=NUM_CHEXPERT,
    ).to(device)

    if args.stage == "B":
        if not args.stage_a_ckpt:
            raise ValueError("--stage-a-ckpt required for Stage B")
        log.info(f"Loading Stage A checkpoint: {args.stage_a_ckpt}")
        ckpt_a = torch.load(args.stage_a_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_a["model_state"])
        model.unfreeze_last_n_blocks(args.unfreeze_blocks)
    else:
        # Stage A: keep backbone frozen, initialize head & attention from NIH
        initialize_head_from_nih(model, transfer_source_path, device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable: {trainable:,}  Frozen: {frozen:,}")

    # ── Datasets (streaming from Redivis) ─────────────────────────────────────
    log.info("Building CheXpert+ streaming datasets...")
    train_ds = StreamingCheXpertDataset(
        index_csv  = str(PROJECT_ROOT / "data/chexpert_subset/train_index.csv"),
        api_token  = api_token,
        table_ref  = "png_train:s6cj",
        cache_dir  = str(PROJECT_ROOT / "data/chexpert_cache"),
        cache_size_gb = 30.0,
        labels     = CHEXPERT_LABELS,
        frontal_only  = True,
        uncertainty_policy = "zeros",
    )
    val_ds = StreamingCheXpertDataset(
        index_csv  = str(PROJECT_ROOT / "data/chexpert_subset/valid_index.csv"),
        api_token  = api_token,
        table_ref  = "PNG_valid:41v9",
        cache_dir  = str(PROJECT_ROOT / "data/chexpert_val_cache"),
        cache_size_gb = 5.0,
        labels     = CHEXPERT_LABELS,
        frontal_only  = True,
        uncertainty_policy = "zeros",
    )
    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    n_samples = cfg["num_samples_per_epoch"]
    # Uniform random sampler — draw n_samples indices from the full training set each epoch.
    # Using replacement=False so each image appears at most once per epoch.
    # This respects the true class distribution (including "No Finding") unlike stratified sampling.
    rng_g = torch.Generator()
    rng_g.manual_seed(42)
    sampler = torch.utils.data.RandomSampler(
        train_ds, replacement=False, num_samples=n_samples, generator=rng_g
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,  batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=0)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if args.stage == "A":
        # Only head/attention params trainable
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr_head"], weight_decay=cfg["weight_decay"]
        )
    else:
        # Separate LR groups: backbone (lower) vs head (higher)
        backbone_params = [p for block in model._get_blocks()
                           for p in block.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" not in n]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
            {"params": head_params,     "lr": cfg["lr_head"]},
        ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    # ── Save path ─────────────────────────────────────────────────────────────
    mode_str = "attention" if not args.no_attention else "linear"
    save_name = f"stage{args.stage}_{args.backbone}_{mode_str}.pth"
    save_path = CKPT_TRANSFER / save_name

    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_path = CKPT_TRANSFER / f"stage{args.stage}_{args.backbone}_{mode_str}.log"
    fh = FlushFileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    log.info(f"Logging to: {log_path}")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_auroc = train_stage(
        model, train_loader, val_loader, optimizer, scheduler,
        cfg, device, save_path, CHEXPERT_LABELS,
        stage_name=f"Stage{args.stage}_{args.backbone}"
    )

    log.info(f"Transfer learning Stage {args.stage} complete.")
    log.info(f"Best Val AUROC: {best_auroc:.4f}")
    log.info(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
