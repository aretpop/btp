"""
scripts/build_report_index.py
==============================
Queries Redivis CheXpert+ for all frontal training images that have
non-null section_findings AND section_impression.

Saves: data/chexpert_subset/train_report_index.csv
Columns: file_id, local_filename, path_to_image, section_findings, section_impression,
         + 11 label columns (same as train_index.csv)

Usage:
    python scripts/build_report_index.py
"""
import os, sys, time, logging
os.environ.setdefault("REDIVIS_API_TOKEN", "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")
from pathlib import Path
import pandas as pd
import redivis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "chexpert_subset" / "train_report_index.csv"
TRAIN_INDEX = PROJECT_ROOT / "data" / "chexpert_subset" / "train_index.csv"

CHEXPERT_LABELS = [
    "No Finding", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Support Devices",
]

def main():
    log.info("Connecting to Redivis CheXpert+...")
    user    = redivis.user("aimi")
    dataset = user.dataset("chexpert_plus:5yyj:v1_0")

    log.info("Querying frontal train images with non-null reports...")
    t0 = time.time()
    df_reports = dataset.query(
        "SELECT path_to_image, section_findings, section_impression "
        "FROM `df_chexpert_plus_240401` "
        "WHERE split = 'train' "
        "  AND frontal_lateral = 'Frontal' "
        "  AND section_findings IS NOT NULL "
        "  AND section_impression IS NOT NULL"
    ).to_pandas_dataframe()
    log.info(f"Redivis query done in {time.time()-t0:.1f}s — {len(df_reports):,} rows with reports")

    # Merge with train_index.csv (which has file_id, labels etc)
    log.info("Loading existing train_index.csv...")
    df_idx = pd.read_csv(TRAIN_INDEX)

    # Normalize path_to_image for join (train_index has 'train/...' paths too)
    df_reports["path_to_image"] = df_reports["path_to_image"].str.replace("\\", "/")
    df_idx["path_to_image"]     = df_idx["path_to_image"].str.replace("\\", "/")

    log.info(f"train_index rows: {len(df_idx):,}")
    df_merged = df_idx.merge(df_reports, on="path_to_image", how="inner")
    log.info(f"After join (has report + label): {len(df_merged):,} rows")

    # Filter frontal only (local_filename contains 'frontal')
    df_frontal = df_merged[df_merged["local_filename"].str.contains("frontal", na=False)].reset_index(drop=True)
    log.info(f"Frontal only: {len(df_frontal):,} rows")

    # Clean up report text
    df_frontal["section_findings"]   = df_frontal["section_findings"].str.strip()
    df_frontal["section_impression"] = df_frontal["section_impression"].str.strip()

    # Drop rows where report is empty string after strip
    df_frontal = df_frontal[
        (df_frontal["section_findings"].str.len() > 10) &
        (df_frontal["section_impression"].str.len() > 5)
    ].reset_index(drop=True)
    log.info(f"After text length filter: {len(df_frontal):,} rows")

    # Save
    df_frontal.to_csv(OUT_PATH, index=False)
    log.info(f"Saved → {OUT_PATH}")
    log.info(f"Coverage: {len(df_frontal):,} / ~190,869 frontal train images have usable reports")

    # Print sample
    row = df_frontal.iloc[0]
    log.info(f"\nSample:\n  path: {row['path_to_image']}\n"
             f"  findings: {row['section_findings'][:120]}...\n"
             f"  impression: {row['section_impression'][:80]}...")

if __name__ == "__main__":
    main()
