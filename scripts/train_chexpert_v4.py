"""
scripts/train_chexpert_v4.py
============================
Full training and evaluation pipeline for CheXpert V4 ("v4_gap").

Execution Protocol:
1. Environment & reproducibility setup (seed, git, GPU, resolved config).
2. Compute & print alpha_c class balance weights from TRAIN split only.
3. Stage A: Backbone frozen in eval() mode, train heads (5 epochs, AdamW lr=1e-3).
4. Stage B: Selective fine-tuning features[5:] (20 epochs, backbone lr=1e-5, head lr=1e-4).
5. Post-processing & evaluation:
   - Val calibration (per-class temperature scaling fitted via L-BFGS -> temperatures.json)
   - Val threshold tuning (maximizing F1 per class -> thresholds.json)
   - Test evaluation with frozen parameters (1,000 bootstrap iterations -> val_metrics.csv, test_metrics.csv, test_summary.md)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import CHEXPERT_14_LABELS, convert_labels
from src.data.splits import make_patient_splits
from src.data.dataset import build_v4_dataloaders
from src.models.chexpert_v4 import CheXpertV4Model
from src.models.loss_v4 import compute_alpha_c, USLTLossV4
from src.eval.protocol import (
    fit_temperature_scaling,
    tune_thresholds_on_val,
    evaluate_predictions,
    save_markdown_summary,
)

log_file_path = PROJECT_ROOT / "logs" / "chexpert_b1_v4_gap.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file_path), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_v4")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: USLTLossV4,
    scaler: GradScaler,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    stage: str = "A",
    max_steps: Optional[int] = None,
) -> float:
    model.train()
    if stage == "A":
        # Keep backbone frozen and in eval mode (BN stats fixed)
        model.backbone.eval()
    elif stage == "B":
        # Keep frozen blocks in eval mode
        unfreeze_from = cfg.get("training", {}).get("stage_b", {}).get("unfreeze_from", 5)
        if unfreeze_from > 0:
            for i in range(min(unfreeze_from, len(model.backbone.features))):
                model.backbone.features[i].eval()

    grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)
    use_amp = cfg.get("training", {}).get("amp", True) and torch.cuda.is_available()

    total_loss = 0.0
    num_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        raw_labels = batch["raw_labels"].to(device, non_blocking=True)

        # Convert labels for training
        y, u, blank, _, mask = convert_labels(raw_labels, cfg=cfg, mode="train")

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits=logits, y=y, u=u, blank=blank, mask=mask)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 50 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1e-5)
            logger.info(
                f"  Stage {stage} | Ep {epoch:02d} [{step+1:4d}/{len(loader):4d}] "
                f"loss: {loss.item():.4f} (avg: {total_loss/num_batches:.4f}) | {speed:.1f} batch/s"
            )

        if max_steps is not None and (step + 1) >= max_steps:
            logger.info(f"  Reached max_steps={max_steps} for smoke test.")
            break

    return total_loss / max(1, num_batches)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect model predictions and evaluation labels."""
    model.eval()
    all_logits = []
    all_targets = []
    all_masks = []
    all_pids = []

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        raw_labels = batch["raw_labels"].to(device, non_blocking=True)
        pids = batch["patient_ids"]

        # EVAL convention: blank -> 0.0, uncertain (-1) -> mask = 0.0
        y, _, _, _, mask = convert_labels(raw_labels, cfg=cfg, mode="eval")

        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)

        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(y.float().cpu().numpy())
        all_masks.append(mask.float().cpu().numpy())
        all_pids.extend(pids)

        if max_steps is not None and (step + 1) >= max_steps:
            break

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)
    pids_arr = np.array(all_pids)

    return logits_arr, targets_arr, masks_arr, pids_arr


def main():
    parser = argparse.ArgumentParser(description="Train CheXpert V4 (v4_gap)")
    parser.add_argument("--config", type=str, default="configs/chexpert_b1_v4_gap.yaml", help="Path to config yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 steps of Stage A and B on tiny batch")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints if available")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only using existing Stage B checkpoint")
    args = parser.parse_args()

    # Load configuration
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git_commit = get_git_commit()
    logger.info("=" * 80)
    logger.info(f"CheXpert V4 Pipeline Launch | Commit: {git_commit}")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Recipe: {cfg.get('recipe')} | Backbone: {cfg.get('backbone', {}).get('name')}")
    logger.info("=" * 80)

    # Output directories
    ckpt_dir = PROJECT_ROOT / cfg.get("output", {}).get("checkpoints_dir", "checkpoints/chexpert_v4")
    reports_dir = PROJECT_ROOT / cfg.get("output", {}).get("reports_dir", "reports")
    splits_dir = PROJECT_ROOT / cfg.get("output", {}).get("splits_dir", "splits")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    resolved_cfg_path = ckpt_dir / "config_resolved.yaml"
    with open(resolved_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"Saved resolved config to {resolved_cfg_path}")

    # Load splits
    train_split_path = splits_dir / "v4_seed42_train.csv"
    val_split_path = splits_dir / "v4_seed42_val.csv"
    test_split_path = splits_dir / "v4_seed42_test.csv"

    if not (train_split_path.exists() and val_split_path.exists() and test_split_path.exists()):
        logger.info("Splits not found on disk. Generating patient-level splits...")
        index_csv = PROJECT_ROOT / cfg.get("data", {}).get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv")
        full_df = pd.read_csv(index_csv)
        train_df, val_df, test_df, _ = make_patient_splits(
            full_df,
            train_frac=cfg.get("data", {}).get("train_frac", 0.70),
            val_frac=cfg.get("data", {}).get("val_frac", 0.10),
            test_frac=cfg.get("data", {}).get("test_frac", 0.20),
            seed=seed,
            official_valid_csv=PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv",
        )
        train_df.to_csv(train_split_path, index=False)
        val_df.to_csv(val_split_path, index=False)
        test_df.to_csv(test_split_path, index=False)
    else:
        logger.info(f"Loading existing splits from {splits_dir}...")
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
        test_df = pd.read_csv(test_split_path)

    logger.info(f"Dataset splits loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Compute alpha_c from TRAIN split only
    loss_cfg = cfg.get("loss", {})
    alpha_tensor, alpha_df = compute_alpha_c(
        train_df=train_df,
        labels=CHEXPERT_14_LABELS,
        alpha_mode=loss_cfg.get("alpha_mode", "effective_number"),
        beta=float(loss_cfg.get("beta", 0.999)),
        reports_dir=reports_dir,
    )

    # Initialize model
    model = CheXpertV4Model(cfg).to(device)

    # Criterion
    criterion = USLTLossV4(
        alpha=alpha_tensor.to(device),
        lam_u=float(loss_cfg.get("lam_u", 0.5)),
        eps=float(loss_cfg.get("eps", 0.1)),
        blank_weight=float(loss_cfg.get("blank_weight", 1.0)),
    )

    # Build DataLoaders
    train_loader, val_loader, test_loader = build_v4_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg.get("data", {}),
        seed=seed,
    )

    bb_name = cfg.get("backbone", {}).get("name", "efficientnet_b1")
    stage_a_ckpt = ckpt_dir / f"stageA_{bb_name}_gap_labeladaptive_14cls.pth"
    stage_b_ckpt = ckpt_dir / f"stageB_{bb_name}_gap_labeladaptive_14cls.pth"

    scaler = GradScaler(enabled=cfg.get("training", {}).get("amp", True) and torch.cuda.is_available())
    bootstrap_val = 0 if args.smoke_test else cfg.get("evaluation", {}).get("bootstrap_iters_val", 200)

    if not args.eval_only:
        # ======================================================================
        # STAGE A: Frozen Backbone, Train Heads
        # ======================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STARTING STAGE A: Head Adaptation (Backbone Frozen in eval mode)")
        logger.info("=" * 80)

        model.freeze_stage_a()

        stg_a_cfg = cfg.get("training", {}).get("stage_a", {})
        epochs_a = 1 if args.smoke_test else stg_a_cfg.get("epochs", 5)
        lr_a = float(stg_a_cfg.get("lr", 1e-3))
        min_lr_a = float(stg_a_cfg.get("min_lr", 1e-6))
        wd_a = float(stg_a_cfg.get("weight_decay", 1e-4))
        patience_a = stg_a_cfg.get("patience", 3)

        trainable_params_a = [p for p in model.parameters() if p.requires_grad]
        opt_a = torch.optim.AdamW(trainable_params_a, lr=lr_a, weight_decay=wd_a)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs_a, eta_min=min_lr_a)

        best_val_lift_a = -1e9
        best_val_auc_a = -1e9
        patience_counter_a = 0

        skip_stage_a = False
        if args.resume and stage_a_ckpt.exists() and not args.smoke_test:
            logger.info(f"Resume requested and {stage_a_ckpt.name} found. Skipping Stage A.")
            skip_stage_a = True

        if not skip_stage_a:
            for ep in range(1, epochs_a + 1):
                t_ep = time.time()
                loss_train = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=opt_a,
                    criterion=criterion,
                    scaler=scaler,
                    device=device,
                    cfg=cfg,
                    epoch=ep,
                    stage="A",
                    max_steps=2 if args.smoke_test else None,
                )
                sched_a.step()

                # Val evaluation
                v_logits, v_targets, v_masks, v_pids = collect_predictions(
                    model=model, loader=val_loader, device=device, cfg=cfg,
                    max_steps=2 if args.smoke_test else None,
                )
                _, val_summary, _ = evaluate_predictions(
                    logits=v_logits,
                    targets=v_targets,
                    masks=v_masks,
                    patient_ids=v_pids,
                    n_bootstraps=bootstrap_val,
                    min_positives_lift=cfg.get("evaluation", {}).get("min_positives_lift", 20),
                )

                ep_time = time.time() - t_ep
                cur_lift = val_summary["macro_auprc_lift"]
                cur_auc = val_summary["macro_auroc"]
                cur_pr = val_summary["macro_auprc"]

                logger.info(
                    f"Stage A | Ep {ep:02d}/{epochs_a:02d} ({ep_time:.1f}s) -> "
                    f"Train Loss: {loss_train:.4f} | Val AUROC: {cur_auc:.4f} | "
                    f"Val AUPRC: {cur_pr:.4f} | Val Lift: {cur_lift:.4f}"
                )

                # Model selection by AUPRC lift with AUROC tie-break
                is_better = (cur_lift > best_val_lift_a + 1e-4) or (
                    abs(cur_lift - best_val_lift_a) <= 1e-4 and cur_auc > best_val_auc_a
                )
                if is_better:
                    best_val_lift_a = cur_lift
                    best_val_auc_a = cur_auc
                    patience_counter_a = 0
                    torch.save(
                        {
                            "epoch": ep,
                            "stage": "A",
                            "model_state": model.state_dict(),
                            "optimizer_state": opt_a.state_dict(),
                            "val_summary": val_summary,
                            "config": cfg,
                        },
                        stage_a_ckpt,
                    )
                    logger.info(f"  --> Saved new best Stage A checkpoint: {stage_a_ckpt.name} (Lift: {cur_lift:.4f})")
                else:
                    patience_counter_a += 1
                    logger.info(f"  No improvement for {patience_counter_a}/{patience_a} epochs.")
                    if patience_counter_a >= patience_a and not args.smoke_test:
                        logger.info(f"Early stopping triggered in Stage A at epoch {ep}.")
                        break

        # ======================================================================
        # STAGE B: Selective Fine-Tuning
        # ======================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STARTING STAGE B: Selective Fine-Tuning (Features[5:] Unfrozen)")
        logger.info("=" * 80)

        # Load best Stage A weights
        if stage_a_ckpt.exists():
            ckpt_a = torch.load(stage_a_ckpt, map_location=device)
            model.load_state_dict(ckpt_a["model_state"])
            logger.info(f"Loaded best Stage A weights from {stage_a_ckpt.name} (Ep {ckpt_a['epoch']}).")

        stg_b_cfg = cfg.get("training", {}).get("stage_b", {})
        epochs_b = 1 if args.smoke_test else stg_b_cfg.get("epochs", 20)
        unfreeze_from = stg_b_cfg.get("unfreeze_from", 5)
        bb_lr = float(stg_b_cfg.get("backbone_lr", 1e-5))
        head_lr = float(stg_b_cfg.get("head_lr", 1e-4))
        min_lr_b = float(stg_b_cfg.get("min_lr", 1e-7))
        wd_b = float(stg_b_cfg.get("weight_decay", 1e-4))
        patience_b = stg_b_cfg.get("patience", 5)

        model.unfreeze_stage_b(unfreeze_from=unfreeze_from)

        # Differential learning rates
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = (
            list(model.msfb.parameters())
            + list(model.se.parameters())
            + list(model.cbam.parameters())
            + list(model.head.parameters())
        )

        param_groups = [
            {"params": bb_params, "lr": bb_lr, "weight_decay": wd_b},
            {"params": head_params, "lr": head_lr, "weight_decay": wd_b},
        ]
        opt_b = torch.optim.AdamW(param_groups)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=epochs_b, eta_min=min_lr_b)

        start_ep_b = 1
        best_val_lift_b = -1e9
        best_val_auc_b = -1e9
        patience_counter_b = 0

        # Resume Stage B if requested and available
        if args.resume and stage_b_ckpt.exists() and not args.smoke_test:
            ckpt_b = torch.load(stage_b_ckpt, map_location=device)
            model.load_state_dict(ckpt_b["model_state"])
            if "optimizer_state" in ckpt_b:
                try:
                    opt_b.load_state_dict(ckpt_b["optimizer_state"])
                except Exception as e:
                    logger.warning(f"Could not restore Stage B optimizer state: {e}")
            start_ep_b = ckpt_b.get("epoch", 0) + 1
            best_val_lift_b = ckpt_b.get("val_summary", {}).get("macro_auprc_lift", -1e9)
            best_val_auc_b = ckpt_b.get("val_summary", {}).get("macro_auroc", -1e9)
            logger.info(
                f"Resuming Stage B from Epoch {start_ep_b}/{epochs_b} "
                f"(prior best lift: {best_val_lift_b:.4f}, AUROC: {best_val_auc_b:.4f})."
            )
            for _ in range(1, start_ep_b):
                sched_b.step()

        for ep in range(start_ep_b, epochs_b + 1):
            t_ep = time.time()
            loss_train = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=opt_b,
                criterion=criterion,
                scaler=scaler,
                device=device,
                cfg=cfg,
                epoch=ep,
                stage="B",
                max_steps=2 if args.smoke_test else None,
            )
            sched_b.step()

            # Val evaluation
            v_logits, v_targets, v_masks, v_pids = collect_predictions(
                model=model, loader=val_loader, device=device, cfg=cfg,
                max_steps=2 if args.smoke_test else None,
            )
            _, val_summary, _ = evaluate_predictions(
                logits=v_logits,
                targets=v_targets,
                masks=v_masks,
                patient_ids=v_pids,
                n_bootstraps=bootstrap_val,
                min_positives_lift=cfg.get("evaluation", {}).get("min_positives_lift", 20),
            )

            ep_time = time.time() - t_ep
            cur_lift = val_summary["macro_auprc_lift"]
            cur_auc = val_summary["macro_auroc"]
            cur_pr = val_summary["macro_auprc"]

            logger.info(
                f"Stage B | Ep {ep:02d}/{epochs_b:02d} ({ep_time:.1f}s) -> "
                f"Train Loss: {loss_train:.4f} | Val AUROC: {cur_auc:.4f} | "
                f"Val AUPRC: {cur_pr:.4f} | Val Lift: {cur_lift:.4f}"
            )

            is_better = (cur_lift > best_val_lift_b + 1e-4) or (
                abs(cur_lift - best_val_lift_b) <= 1e-4 and cur_auc > best_val_auc_b
            )
            if is_better:
                best_val_lift_b = cur_lift
                best_val_auc_b = cur_auc
                patience_counter_b = 0
                torch.save(
                    {
                        "epoch": ep,
                        "stage": "B",
                        "model_state": model.state_dict(),
                        "optimizer_state": opt_b.state_dict(),
                        "val_summary": val_summary,
                        "config": cfg,
                    },
                    stage_b_ckpt,
                )
                logger.info(f"  --> Saved new best Stage B checkpoint: {stage_b_ckpt.name} (Lift: {cur_lift:.4f})")
            else:
                patience_counter_b += 1
                logger.info(f"  No improvement for {patience_counter_b}/{patience_b} epochs.")
                if patience_counter_b >= patience_b and not args.smoke_test:
                    logger.info(f"Early stopping triggered in Stage B at epoch {ep}.")
                    break

    # ==========================================================================
    # FINAL POST-PROCESSING & EVALUATION (ON VAL AND TEST)
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION & CALIBRATION PROTOCOL")
    logger.info("=" * 80)

    target_ckpt = stage_b_ckpt if stage_b_ckpt.exists() else stage_a_ckpt
    if target_ckpt.exists():
        ckpt_data = torch.load(target_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state"])
        logger.info(f"Loaded best checkpoint: {target_ckpt.name} (Stage {ckpt_data.get('stage')}, Ep {ckpt_data.get('epoch')})")

    # 1. Collect full validation predictions
    logger.info("Collecting validation predictions...")
    v_logits, v_targets, v_masks, v_pids = collect_predictions(
        model=model, loader=val_loader, device=device, cfg=cfg,
        max_steps=2 if args.smoke_test else None,
    )

    # 2. Fit temperature scaling on VAL logits
    logger.info("Fitting per-class temperature scaling on VAL logits via L-BFGS...")
    scaler, temp_dict = fit_temperature_scaling(
        val_logits=torch.from_numpy(v_logits).to(device),
        val_targets=torch.from_numpy(v_targets).to(device),
        val_masks=torch.from_numpy(v_masks).to(device),
    )
    temp_json_path = reports_dir / "temperatures.json"
    with open(temp_json_path, "w") as f:
        json.dump(temp_dict, f, indent=2)
    logger.info(f"Saved calibrated temperatures to {temp_json_path.name}: {temp_dict}")

    # 3. Tune thresholds on calibrated VAL probabilities
    logger.info("Tuning per-class decision thresholds on calibrated VAL probabilities...")
    with torch.no_grad():
        scaled_v_logits = scaler(torch.from_numpy(v_logits).to(device))
        val_calib_probs = torch.sigmoid(scaled_v_logits).cpu().numpy()

    thresholds_dict = tune_thresholds_on_val(
        val_probs=val_calib_probs,
        val_targets=v_targets,
        val_masks=v_masks,
        min_positives=cfg.get("evaluation", {}).get("min_positives_lift", 20),
    )
    th_json_path = reports_dir / "thresholds.json"
    with open(th_json_path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    logger.info(f"Saved tuned thresholds to {th_json_path.name}: {thresholds_dict}")

    bootstrap_test = 0 if args.smoke_test else cfg.get("evaluation", {}).get("bootstrap_iters_test", 1000)

    # 4. Generate full VAL metrics report
    val_metrics_df, val_summary_full, _ = evaluate_predictions(
        logits=v_logits,
        targets=v_targets,
        masks=v_masks,
        patient_ids=v_pids,
        thresholds=thresholds_dict,
        scaler=scaler,
        n_bootstraps=bootstrap_test,
    )
    val_metrics_csv = reports_dir / "val_metrics.csv"
    val_metrics_df.to_csv(val_metrics_csv, index=False)
    logger.info(f"Saved validation metrics to {val_metrics_csv.name}")

    # 5. Evaluate on TEST (one single pass with frozen thresholds & temperatures)
    logger.info("Evaluating on TEST split with frozen thresholds and calibration temperatures...")
    t_logits, t_targets, t_masks, t_pids = collect_predictions(
        model=model, loader=test_loader, device=device, cfg=cfg,
        max_steps=2 if args.smoke_test else None,
    )
    test_metrics_df, test_summary_full, _ = evaluate_predictions(
        logits=t_logits,
        targets=t_targets,
        masks=t_masks,
        patient_ids=t_pids,
        thresholds=thresholds_dict,
        scaler=scaler,
        n_bootstraps=bootstrap_test,
    )
    test_metrics_csv = reports_dir / "test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_csv, index=False)
    logger.info(f"Saved test metrics to {test_metrics_csv.name}")

    # 6. Save markdown summary report
    test_summary_md = reports_dir / "test_summary.md"
    save_markdown_summary(test_metrics_df, test_summary_full, test_summary_md)

    logger.info("\n" + "=" * 80)
    logger.info("TEST EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Macro AUROC:      {test_summary_full['macro_auroc']:.4f}")
    logger.info(f"Macro AUPRC:      {test_summary_full['macro_auprc']:.4f}")
    logger.info(f"Macro AUPRC Lift: {test_summary_full['macro_auprc_lift']:.4f}")
    logger.info(f"Macro F1:         {test_summary_full['macro_f1']:.4f}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()

