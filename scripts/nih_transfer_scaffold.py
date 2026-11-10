"""
scripts/nih_transfer_scaffold.py
==================================
NIH ChestX-ray14 transfer learning scaffold.

Supports:
1. Loading a CheXpert B1v2 checkpoint and mapping overlapping heads to NIH
2. Masked loss for joint CheXpert + NIH training
3. Official NIH patient-level splits

Usage:
    python scripts/nih_transfer_scaffold.py \\
        --chexpert_checkpoint checkpoints/chexpert_b1_v2/stageB_best.pth \\
        --nih_data_dir /path/to/nih \\
        --mode sequential_finetune \\
        --output_dir outputs/nih_transfer

Label overlap note:
    CheXpert "Pleural Effusion" ↔ NIH "Effusion" (name differs)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("nih_transfer")

# =============================================================================
# Label definitions
# =============================================================================

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

NIH_14_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# CheXpert label → NIH label (exact name mapping for overlapping classes)
CHEXPERT_TO_NIH: Dict[str, str] = {
    "Cardiomegaly":    "Cardiomegaly",
    "Edema":           "Edema",
    "Consolidation":   "Consolidation",
    "Pneumonia":       "Pneumonia",
    "Atelectasis":     "Atelectasis",
    "Pneumothorax":    "Pneumothorax",
    "Pleural Effusion": "Effusion",   # name differs
}
NIH_TO_CHEXPERT: Dict[str, str] = {v: k for k, v in CHEXPERT_TO_NIH.items()}

# CheXpert-only labels (no NIH equivalent)
CHEXPERT_ONLY = [
    "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity",
    "Lung Lesion", "Pleural Other", "Fracture", "Support Devices",
]

# NIH-only labels (no CheXpert equivalent)
NIH_ONLY = [
    "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening",
]


# =============================================================================
# Transfer: CheXpert → NIH
# =============================================================================

def transfer_chexpert_to_nih(
    chexpert_checkpoint_path: str,
    output_num_classes: int = 14,  # NIH 14-class
    freeze_backbone: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a CheXpert B1v2 checkpoint and build a NIH-ready model.

    Strategy:
    - Load CheXpert model (14 classes)
    - For NIH, create a new 14-class model with the same backbone
    - Transfer overlapping classifier heads from CheXpert
    - Random-init heads for NIH-only classes
    - Optionally freeze backbone (Stage A)

    Returns:
        NIH-adapted CheXpertB1V2Model with 14 NIH classes
    """
    from src.models.chexpert_b1_v2 import CheXpertB1V2Model

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load CheXpert checkpoint
    log.info("Loading CheXpert checkpoint: %s", chexpert_checkpoint_path)
    ckpt = torch.load(PROJECT_ROOT / chexpert_checkpoint_path, map_location=dev)
    ckpt_cfg = ckpt.get("cfg", {}).get("model", {})

    # Build CheXpert model and load weights
    chexpert_model = CheXpertB1V2Model(ckpt_cfg)
    chexpert_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    # Build NIH model (same backbone architecture, different num_classes)
    nih_cfg = {**ckpt_cfg, "num_classes": output_num_classes}
    nih_model = CheXpertB1V2Model(nih_cfg)

    # Transfer backbone, MSFB, channel/spatial attention, DAG
    shared_modules = ["msfb", "channel_attn", "spatial_attn"]
    if hasattr(chexpert_model, "lateral_conv"):
        shared_modules += ["lateral_conv", "fuse_bn_relu"]

    state_dict_chex = chexpert_model.state_dict()
    state_dict_nih  = nih_model.state_dict()

    # Transfer backbone params
    for k in list(state_dict_chex.keys()):
        if k.startswith("backbone.") or any(k.startswith(m + ".") for m in shared_modules):
            if k in state_dict_nih and state_dict_nih[k].shape == state_dict_chex[k].shape:
                state_dict_nih[k] = state_dict_chex[k].clone()

    # Transfer overlapping classifier heads
    # CheXpert classifier[i] corresponds to CHEXPERT_14_LABELS[i]
    for nih_idx, nih_lbl in enumerate(NIH_14_LABELS):
        chex_lbl = NIH_TO_CHEXPERT.get(nih_lbl, None)
        if chex_lbl is None:
            log.info("NIH-only class (random init): %s", nih_lbl)
            continue
        chex_idx = CHEXPERT_14_LABELS.index(chex_lbl)
        # Transfer Linear(256, 1) weight and bias
        chex_w_key = f"classifiers.{chex_idx}.weight"
        chex_b_key = f"classifiers.{chex_idx}.bias"
        nih_w_key  = f"classifiers.{nih_idx}.weight"
        nih_b_key  = f"classifiers.{nih_idx}.bias"
        if chex_w_key in state_dict_chex and nih_w_key in state_dict_nih:
            state_dict_nih[nih_w_key] = state_dict_chex[chex_w_key].clone()
            state_dict_nih[nih_b_key] = state_dict_chex[chex_b_key].clone()
            log.info("Transferred head: CheXpert[%d]=%s → NIH[%d]=%s",
                     chex_idx, chex_lbl, nih_idx, nih_lbl)

    nih_model.load_state_dict(state_dict_nih, strict=True)

    if freeze_backbone:
        nih_model.freeze_backbone()
        log.info("NIH model: backbone frozen for Stage A.")

    nih_model = nih_model.to(dev)
    log.info("NIH-adapted model ready. Num classes: %d", output_num_classes)
    return nih_model


# =============================================================================
# Masked loss for joint training
# =============================================================================

def masked_loss_joint(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dataset_ids: torch.Tensor,
    chexpert_label_indices: List[int],
    nih_label_indices: List[int],
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked BCE loss for joint CheXpert + NIH training.

    Args:
        logits:                 [B, C] model output
        targets:                [B, C] ground-truth labels
        dataset_ids:            [B] tensor, 0 = CheXpert, 1 = NIH
        chexpert_label_indices: list of column indices for CheXpert labels
        nih_label_indices:      list of column indices for NIH labels
        pos_weight:             [C] or None

    Returns:
        scalar loss
    """
    B, C = logits.shape
    mask = torch.zeros(B, C, device=logits.device)

    # CheXpert samples → only CheXpert label columns
    chex_rows = (dataset_ids == 0)
    if chex_rows.any():
        mask[chex_rows, :][:, chexpert_label_indices] = 1.0

    # NIH samples → only NIH label columns
    nih_rows = (dataset_ids == 1)
    if nih_rows.any():
        mask[nih_rows, :][:, nih_label_indices] = 1.0

    per_elem = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    masked = per_elem * mask
    n_active = mask.sum().clamp(min=1.0)
    return masked.sum() / n_active


# =============================================================================
# NIH official splits
# =============================================================================

def load_nih_official_splits(
    nih_data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load official NIH ChestX-ray14 patient-level train/test splits.

    Official files:
        {nih_data_dir}/train_val_list.txt  — list of image filenames for train+val
        {nih_data_dir}/test_list.txt       — list of image filenames for test

    Args:
        nih_data_dir: path to the NIH ChestX-ray14 dataset directory
                      (must contain Data_Entry_2017.csv and split text files)

    Returns:
        (train_df, val_df, test_df) — DataFrames with patient-level separation

    Raises:
        FileNotFoundError: if split files or Data_Entry_2017.csv not found
    """
    nih_dir = Path(nih_data_dir)

    # Check required files
    csv_path = nih_dir / "Data_Entry_2017.csv"
    train_val_list = nih_dir / "train_val_list.txt"
    test_list = nih_dir / "test_list.txt"

    for f in [csv_path, train_val_list, test_list]:
        if not f.exists():
            raise FileNotFoundError(
                f"Required NIH file not found: {f}\n"
                f"Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
                f"Required files: Data_Entry_2017.csv, train_val_list.txt, test_list.txt"
            )

    # Load metadata
    df = pd.read_csv(csv_path)
    if "Image Index" not in df.columns:
        raise ValueError("Data_Entry_2017.csv must contain 'Image Index' column")

    # Load split file lists
    with open(train_val_list) as f:
        train_val_files = set(f.read().strip().split("\n"))
    with open(test_list) as f:
        test_files = set(f.read().strip().split("\n"))

    # Patient ID extraction: NIH image filenames are like "00000001_000.png"
    # Patient ID = first 8 digits
    df["patient_id"] = df["Image Index"].apply(
        lambda x: x.split("_")[0]
    )

    # Assign splits
    df_test = df[df["Image Index"].isin(test_files)].reset_index(drop=True)

    # Train/val from train_val_list (further split 90/10 by patient)
    df_trainval = df[df["Image Index"].isin(train_val_files)].reset_index(drop=True)
    all_trainval_pids = df_trainval["patient_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(all_trainval_pids)
    n_val = int(len(all_trainval_pids) * 0.10)
    val_pids = set(all_trainval_pids[:n_val])
    train_pids = set(all_trainval_pids[n_val:])

    df_train = df_trainval[df_trainval["patient_id"].isin(train_pids)].reset_index(drop=True)
    df_val   = df_trainval[df_trainval["patient_id"].isin(val_pids)].reset_index(drop=True)

    # Assert patient-level separation
    all_pids_train = set(df_train["patient_id"].unique())
    all_pids_val   = set(df_val["patient_id"].unique())
    all_pids_test  = set(df_test["patient_id"].unique())

    assert len(all_pids_train & all_pids_val)  == 0, "NIH: patient leakage train∩val"
    assert len(all_pids_train & all_pids_test) == 0, "NIH: patient leakage train∩test"
    assert len(all_pids_val   & all_pids_test) == 0, "NIH: patient leakage val∩test"

    log.info(
        "NIH splits — train: %d images (%d patients), val: %d images (%d patients), "
        "test: %d images (%d patients)",
        len(df_train), len(all_pids_train),
        len(df_val),   len(all_pids_val),
        len(df_test),  len(all_pids_test),
    )

    return df_train, df_val, df_test


# =============================================================================
# Main (scaffold runner)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NIH Transfer Scaffold")
    parser.add_argument("--chexpert_checkpoint", required=True,
                        help="Path to trained CheXpert B1v2 checkpoint")
    parser.add_argument("--nih_data_dir", default=None,
                        help="Path to NIH ChestX-ray14 dataset directory")
    parser.add_argument("--mode", default="sequential_finetune",
                        choices=["sequential_finetune", "joint_training"],
                        help="Transfer mode")
    parser.add_argument("--output_dir", default="outputs/nih_transfer")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build model and print info, don't train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build NIH-adapted model
    log.info("Mode: %s", args.mode)
    nih_model = transfer_chexpert_to_nih(
        args.chexpert_checkpoint,
        output_num_classes=len(NIH_14_LABELS),
        freeze_backbone=True,  # Stage A: freeze backbone
        device=device,
    )

    total_params = sum(p.numel() for p in nih_model.parameters())
    trainable_params = sum(p.numel() for p in nih_model.parameters() if p.requires_grad)
    log.info("NIH model: total=%.1fM, trainable=%.1fM",
             total_params/1e6, trainable_params/1e6)

    # Print label mapping summary
    print("\n=== Label Transfer Mapping ===")
    print(f"{'NIH Label':<25} {'CheXpert Source':<30} {'Type'}")
    print("-" * 70)
    for nih_lbl in NIH_14_LABELS:
        chex_lbl = NIH_TO_CHEXPERT.get(nih_lbl, None)
        if chex_lbl:
            print(f"{nih_lbl:<25} ← {chex_lbl:<30} [transferred]")
        else:
            print(f"{nih_lbl:<25} ← {'(random init)':<30} [new head]")

    if args.nih_data_dir:
        try:
            train_df, val_df, test_df = load_nih_official_splits(args.nih_data_dir)
            log.info("NIH splits loaded successfully.")
        except FileNotFoundError as e:
            log.error("%s", e)
    else:
        log.info("No --nih_data_dir provided — skipping data loading.")
        log.info("To start training, provide --nih_data_dir and implement the training loop.")

    if args.dry_run:
        log.info("Dry run complete.")
        return

    # Training scaffold
    log.info(
        "\n[SCAFFOLD] To train, implement the following loop:\n"
        "  Stage A: train NIH heads only (frozen backbone, 5 epochs)\n"
        "  Stage B: unfreeze features[5:], backbone_lr=1e-5, head_lr=1e-4, 20 epochs\n"
        "  Use masked_loss_joint() for joint CheXpert+NIH training\n"
        "  Use load_nih_official_splits() for NIH data\n"
        "  Checkpoint: %s",
        output_dir / "nih_stageB_best.pth"
    )

    # Save label mapping config
    mapping_path = output_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "chexpert_to_nih": CHEXPERT_TO_NIH,
            "nih_to_chexpert": NIH_TO_CHEXPERT,
            "chexpert_only": CHEXPERT_ONLY,
            "nih_only": NIH_ONLY,
            "chexpert_14_labels": CHEXPERT_14_LABELS,
            "nih_14_labels": NIH_14_LABELS,
        }, f, indent=2)
    log.info("Saved label mapping: %s", mapping_path)


if __name__ == "__main__":
    main()

