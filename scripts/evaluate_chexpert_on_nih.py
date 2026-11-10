"""
scripts/evaluate_chexpert_on_nih.py
===================================
Evaluates CheXpertB1V2Model directly on the NIH ChestX-ray14 dataset.
Measures out-of-domain cross-dataset generalization across the 7 overlapping classes:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion (NIH: "Effusion")
- Pneumonia
- Pneumothorax
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_nih")

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# 7 Overlapping classes between CheXpert and NIH
OVERLAPPING_CLASSES = {
    # CheXpert label -> NIH label
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
}


class NIHEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, resolution: int = 224):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Precompute targets for overlapping classes
        self.targets = np.zeros((len(self.df), len(OVERLAPPING_CLASSES)), dtype=np.float32)
        for row_idx, row in self.df.iterrows():
            findings = str(row["Finding Labels"]).split("|")
            for col_idx, (_, nih_label) in enumerate(OVERLAPPING_CLASSES.items()):
                if nih_label in findings:
                    self.targets[row_idx, col_idx] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = self.images_dir / img_name
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_t = self.transform(img)
        except Exception:
            img_t = torch.zeros(3, self.resolution, self.resolution)
        return img_t, torch.from_numpy(self.targets[idx])


def main():
    parser = argparse.ArgumentParser(description="Evaluate CheXpert model on NIH dataset")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train_val", "all"],
        help="Which NIH split to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of samples for fast smoke testing",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Running on device: %s", device)

    # 1. Load Checkpoint
    ckpt_path = PROJECT_ROOT / args.checkpoint
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})

    model = CheXpertB1V2Model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Find CheXpert head indices corresponding to the 7 overlapping classes
    chex_indices = [
        CHEXPERT_14_LABELS.index(chex_lbl)
        for chex_lbl in OVERLAPPING_CLASSES.keys()
    ]

    # 2. Load NIH Data
    meta_path = PROJECT_ROOT / "data/nih_cxr14/metadata/Data_Entry_2017.csv"
    images_dir = PROJECT_ROOT / "data/nih_cxr14/images/images-224"
    log.info("Reading NIH metadata: %s", meta_path)
    df = pd.read_csv(meta_path)

    if args.split == "test":
        split_file = PROJECT_ROOT / "data/nih_cxr14/metadata/test_list_NIH.txt"
        with open(split_file) as f:
            valid_set = set(line.strip() for line in f if line.strip())
        df = df[df["Image Index"].isin(valid_set)].reset_index(drop=True)
        log.info("Filtered to official NIH test split: %d images", len(df))
    elif args.split == "train_val":
        split_file = PROJECT_ROOT / "data/nih_cxr14/metadata/train_val_list_NIH.txt"
        with open(split_file) as f:
            valid_set = set(line.strip() for line in f if line.strip())
        df = df[df["Image Index"].isin(valid_set)].reset_index(drop=True)
        log.info("Filtered to official NIH train/val split: %d images", len(df))

    if args.max_samples and len(df) > args.max_samples:
        df = df.iloc[: args.max_samples].reset_index(drop=True)
        log.info("Sample capped at %d images", len(df))

    dataset = NIHEvalDataset(
        df,
        images_dir,
        resolution=model_cfg.get("input_resolution", 224),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Inference
    log.info("Starting inference on %d NIH images...", len(dataset))
    t0 = time.time()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for b_idx, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(imgs)
                # Extract only the 7 overlapping class logits
                sub_logits = logits[:, chex_indices]
                probs = torch.sigmoid(sub_logits).float().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(loader):
                elapsed = time.time() - t0
                log.info("Processed batch %d/%d (%.1f img/s)", b_idx + 1, len(loader), (b_idx + 1) * args.batch_size / elapsed)

    probs_all = np.concatenate(all_probs, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)

    # 4. Metrics Computation
    results = {}
    valid_aurocs = []
    valid_auprcs = []

    class_names = list(OVERLAPPING_CLASSES.keys())
    for i, chex_name in enumerate(class_names):
        nih_name = OVERLAPPING_CLASSES[chex_name]
        t = targets_all[:, i]
        p = probs_all[:, i]
        n_pos = int(t.sum())

        if len(np.unique(t)) < 2:
            auc = float("nan")
            pr = float("nan")
        else:
            auc = float(roc_auc_score(t, p))
            pr = float(average_precision_score(t, p))
            valid_aurocs.append(auc)
            valid_auprcs.append(pr)

        # Optimal threshold via Youden's J
        if len(np.unique(t)) >= 2:
            fpr, tpr, threshs = roc_curve(t, p)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = float(threshs[best_idx])
            pred = (p >= best_thresh).astype(int)
            f1 = float(f1_score(t, pred, zero_division=0))
        else:
            best_thresh = 0.5
            f1 = float("nan")

        results[chex_name] = {
            "nih_label": nih_name,
            "auroc": auc,
            "auprc": pr,
            "optimal_threshold": best_thresh,
            "f1": f1,
            "positives": n_pos,
            "total": len(t),
        }

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")
    results["_macro"] = {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
    }

    # 5. Output Table
    print("\n" + "=" * 78)
    print(f" ZERO-SHOT CROSS-DATASET EVALUATION ON NIH ({args.split.upper()} SET: {len(dataset):,} IMAGES)")
    print(f" Model Checkpoint: {args.checkpoint}")
    print("=" * 78)
    print(f"{'CheXpert Pathology':<20} {'NIH Label':<16} {'AUROC':>8} {'AUPRC':>8} {'Best Thr':>9} {'F1':>7} {'#Pos':>7}")
    print("-" * 78)
    for chex_name in class_names:
        r = results[chex_name]
        print(
            f"{chex_name:<20} {r['nih_label']:<16} {r['auroc']:>8.4f} {r['auprc']:>8.4f} "
            f"{r['optimal_threshold']:>9.4f} {r['f1']:>7.4f} {r['positives']:>7d}"
        )
    print("-" * 78)
    print(f"{'OVERLAPPING MACRO':<37} {macro_auroc:>8.4f} {macro_auprc:>8.4f}")
    print("=" * 78 + "\n")

    # 6. Save JSON
    out_dir = PROJECT_ROOT / "outputs/nih_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"nih_{args.split}_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved metrics to: %s", out_file)


if __name__ == "__main__":
    main()

