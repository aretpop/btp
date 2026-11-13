"""
src/data/dataset.py
===================
CheXpert V4 Dataset and DataLoader construction.

- Preprocessing: Resize(240), RGB replication, ImageNet normalization.
- Train augmentation: RandomResizedCrop(240, scale=(0.8, 1.0), ratio=(0.9, 1.1)) + RandomHorizontalFlip(p=0.5).
- Val/Test: Resize(240) only.
- Batch Construction: Plain DataLoader, batch_size=32, shuffle=True, drop_last=True.
- No weighted/tiered sampler unless explicitly requested behind sampler: tiered.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from src.data.labels import CHEXPERT_14_LABELS, convert_labels
from src.data.splits import extract_patient_id

logger = logging.getLogger(__name__)

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_v4_transforms(
    input_size: int = 240,
    split: str = "train",
    hflip: bool = True,
) -> T.Compose:
    """Build torchvision transform pipeline per Section 2 specification."""
    if split == "train":
        t_list = [
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        ]
        if hflip:
            t_list.append(T.RandomHorizontalFlip(p=0.5))
        t_list.extend([
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
        return T.Compose(t_list)
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])


class CheXpertV4Dataset(Dataset):
    """
    CheXpert V4 Dataset loading images from local cache and returning raw label tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        input_size: int = 240,
        hflip: bool = True,
        view_filter: str = "none",
    ):
        self.split = split
        self.input_size = input_size
        self.view_filter = view_filter.lower()

        df_filtered = df.copy()
        if self.view_filter != "none":
            col = "local_filename" if "local_filename" in df_filtered.columns else "path_to_image"
            if self.view_filter == "frontal":
                df_filtered = df_filtered[df_filtered[col].str.contains("frontal", case=False, na=False)]
            elif self.view_filter == "lateral":
                df_filtered = df_filtered[df_filtered[col].str.contains("lateral", case=False, na=False)]

        self.df = df_filtered.reset_index(drop=True)
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "chexpert_cache"
        self.cache_dir = Path(cache_dir)
        self.transform = get_v4_transforms(input_size=input_size, split=split, hflip=hflip)

        if "patient_id" not in self.df.columns:
            self.df["patient_id"] = self.df["path_to_image"].apply(extract_patient_id)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fid = str(row["file_id"])
        pid = str(row.get("patient_id", "unknown"))

        # Find cached image on disk
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = self.cache_dir / f"{fid}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is not None and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to read image at {img_path}: {e}")
                img = Image.new("RGB", (self.input_size, self.input_size), 0)
        else:
            img = Image.new("RGB", (self.input_size, self.input_size), 0)

        image_tensor = self.transform(img)

        # Extract raw labels [14] (values: 1.0, 0.0, -1.0, or NaN)
        raw_vals = []
        for lbl in CHEXPERT_14_LABELS:
            v = row.get(lbl, float("nan"))
            raw_vals.append(float(v) if not pd.isna(v) else float("nan"))
        raw_tensor = torch.tensor(raw_vals, dtype=torch.float32)

        return {
            "image": image_tensor,
            "raw_labels": raw_tensor,
            "patient_id": pid,
            "file_id": fid,
        }


def collate_v4(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    raw_labels = torch.stack([item["raw_labels"] for item in batch], dim=0)
    patient_ids = [item["patient_id"] for item in batch]
    file_ids = [item["file_id"] for item in batch]
    return {
        "images": images,
        "raw_labels": raw_labels,
        "patient_ids": patient_ids,
        "file_ids": file_ids,
    }


def build_v4_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct DataLoaders per Section 3 specification.
    """
    input_size = cfg.get("input_size", cfg.get("input_resolution", 240))
    batch_size = cfg.get("batch_size", 32)
    eval_batch_size = cfg.get("eval_batch_size", 64)
    num_workers = cfg.get("num_workers", 8)
    hflip = cfg.get("hflip", True)
    view_filter = cfg.get("view_filter", "none")
    cache_dir = cfg.get("cache_dir", "data/chexpert_cache")
    sampler_type = cfg.get("sampler", {}).get("mode", "random") if isinstance(cfg.get("sampler"), dict) else str(cfg.get("sampler", "random"))

    if sampler_type == "tiered":
        logger.warning(
            "DOUBLE CORRECTION ALERT: Combining 'sampler: tiered' with weighted loss (USLT) "
            "leads to compounding over-correction on tail classes!"
        )

    train_ds = CheXpertV4Dataset(
        train_df, cache_dir=cache_dir, split="train",
        input_size=input_size, hflip=hflip, view_filter=view_filter
    )
    val_ds = CheXpertV4Dataset(
        val_df, cache_dir=cache_dir, split="val",
        input_size=input_size, hflip=False, view_filter=view_filter
    )
    test_ds = CheXpertV4Dataset(
        test_df, cache_dir=cache_dir, split="test",
        input_size=input_size, hflip=False, view_filter=view_filter
    )

    # Seeded generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # Safe worker settings (Windows handles 4-8 workers cleanly if pin_memory is True)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        generator=g,
        collate_fn=collate_v4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_v4,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_v4,
    )

    logger.info(
        f"DataLoaders built: train={len(train_loader)} batches ({len(train_ds)} items, bs={batch_size}), "
        f"val={len(val_loader)} batches ({len(val_ds)} items, bs={eval_batch_size}), "
        f"test={len(test_loader)} batches ({len(test_ds)} items, bs={eval_batch_size})."
    )
    return train_loader, val_loader, test_loader

