"""
src/data/splits.py
==================
Patient-level train/val/test splitting (70/10/20) and class balance reporting.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.labels import CHEXPERT_14_LABELS

logger = logging.getLogger(__name__)


def extract_patient_id(path_str: str) -> str:
    """Extract patient ID string (e.g. 'patient04528') from local or redivis path."""
    m = re.search(r"patient\d+", str(path_str))
    if m:
        return m.group(0)
    # Fallback to parent directory if patient format differs
    parts = str(path_str).replace("\\", "/").split("/")
    for p in parts:
        if "patient" in p.lower():
            return p
    return "unknown"


def extract_view_info(path_str: str, filename_str: str = "") -> Tuple[str, str]:
    """
    Extract view position (frontal vs lateral) and projection (AP vs PA) if present.
    """
    combined = (str(path_str) + " " + str(filename_str)).lower()
    
    # View: frontal vs lateral
    if "lateral" in combined:
        view = "lateral"
    elif "frontal" in combined:
        view = "frontal"
    else:
        view = "unknown"

    # Projection: AP vs PA
    if "_ap" in combined or " ap" in combined or "/ap" in combined:
        proj = "AP"
    elif "_pa" in combined or " pa" in combined or "/pa" in combined:
        proj = "PA"
    elif "lateral" in combined or "_ll" in combined:
        proj = "LL"
    else:
        proj = "unknown"

    return view, proj


def make_patient_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
    official_valid_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create strictly disjoint patient-level splits.
    Guarantees zero patient overlap between train, val, and test.
    """
    df = df.copy()
    if "patient_id" not in df.columns:
        df["patient_id"] = df["path_to_image"].apply(extract_patient_id)

    # Check official validation patients to keep out of train
    official_valid_patients = set()
    if official_valid_csv is not None and Path(official_valid_csv).exists():
        val_df = pd.read_csv(official_valid_csv)
        col = "path_to_image" if "path_to_image" in val_df.columns else val_df.columns[0]
        official_valid_patients = set(val_df[col].apply(extract_patient_id))
        logger.info(f"Loaded {len(official_valid_patients)} official validation patients.")
    else:
        logger.warning("Official CheXpert validation CSV not found or not provided.")

    # All unique patients in registry
    all_patients = sorted(df["patient_id"].unique())
    logger.info(f"Total unique patients in registry: {len(all_patients):,}")

    # Separate any official valid patients if they overlap
    overlap = set(all_patients).intersection(official_valid_patients)
    if len(overlap) > 0:
        logger.info(f"Excluding {len(overlap)} official validation patients from train split.")
        eligible_patients = [p for p in all_patients if p not in overlap]
    else:
        eligible_patients = all_patients

    # Deterministic patient shuffle
    rng = np.random.RandomState(seed)
    shuffled_patients = np.array(eligible_patients)
    rng.shuffle(shuffled_patients)

    n_total = len(shuffled_patients)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    # Test gets the remainder to ensure exact coverage
    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    # Assert zero overlap
    assert len(train_patients.intersection(val_patients)) == 0, "Train-Val patient leakage!"
    assert len(train_patients.intersection(test_patients)) == 0, "Train-Test patient leakage!"
    assert len(val_patients.intersection(test_patients)) == 0, "Val-Test patient leakage!"
    assert len(train_patients) + len(val_patients) + len(test_patients) == n_total

    train_df = df[df["patient_id"].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].copy().reset_index(drop=True)

    official_df = df[df["patient_id"].isin(overlap)].copy().reset_index(drop=True) if len(overlap) > 0 else pd.DataFrame()

    logger.info(
        f"Split complete (seed={seed}): "
        f"Train: {len(train_df):,} images ({len(train_patients):,} patients), "
        f"Val: {len(val_df):,} images ({len(val_patients):,} patients), "
        f"Test: {len(test_df):,} images ({len(test_patients):,} patients)."
    )
    return train_df, val_df, test_df, official_df


def generate_class_balance_report(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Generate detailed class balance report per Section 2 specifications.
    """
    n_images = len(split_df)
    n_patients = split_df["patient_id"].nunique() if "patient_id" in split_df.columns else split_df["path_to_image"].apply(extract_patient_id).nunique()

    # View stats
    views = split_df["local_filename"].str.lower() if "local_filename" in split_df.columns else split_df["path_to_image"].str.lower()
    n_frontal = int(views.str.contains("frontal").sum())
    n_lateral = int(views.str.contains("lateral").sum())

    records = []
    for cls_name in CHEXPERT_14_LABELS:
        if cls_name in split_df.columns:
            s = split_df[cls_name]
            n_pos = int((s == 1.0).sum())
            n_neg_explicit = int((s == 0.0).sum())
            n_unc = int((s == -1.0).sum())
            n_blank = int(s.isna().sum())
            n_neg_blank0 = n_neg_explicit + n_blank
            prev = n_pos / max(1, n_images)
            unc_pos_ratio = n_unc / max(1, n_pos)
        else:
            n_pos = n_neg_explicit = n_unc = n_blank = n_neg_blank0 = 0
            prev = unc_pos_ratio = 0.0

        records.append({
            "class_name": cls_name,
            "n_pos": n_pos,
            "n_neg_explicit": n_neg_explicit,
            "n_uncertain": n_unc,
            "n_blank": n_blank,
            "n_neg_blank0": n_neg_blank0,
            "prevalence": round(prev, 6),
            "uncertain_pos_ratio": round(unc_pos_ratio, 4),
            "total_images": n_images,
            "unique_patients": n_patients,
            "frontal_views": n_frontal,
            "lateral_views": n_lateral,
            "split": split_name,
        })

    report_df = pd.DataFrame(records)
    return report_df

