"""
scripts/create_v4_splits_and_reports.py
=======================================
Generate patient-level splits (70/10/20, seed=42) and detailed class balance reports.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import logging

from src.data.splits import (
    make_patient_splits,
    generate_class_balance_report,
    extract_patient_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v4_splits")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "chexpert_subset" / "chexpert_materialized_index.csv"
OFFICIAL_VALID_PATH = PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv"

SPLITS_DIR = PROJECT_ROOT / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info(f"Loading registry from {INDEX_PATH}...")
    df = pd.read_csv(INDEX_PATH)
    logger.info(f"Loaded {len(df):,} total records.")

    # Generate patient-level splits
    train_df, val_df, test_df, official_df = make_patient_splits(
        df,
        train_frac=0.70,
        val_frac=0.10,
        test_frac=0.20,
        seed=42,
        official_valid_csv=OFFICIAL_VALID_PATH,
    )

    # Save splits
    train_path = SPLITS_DIR / "v4_seed42_train.csv"
    val_path = SPLITS_DIR / "v4_seed42_val.csv"
    test_path = SPLITS_DIR / "v4_seed42_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved splits to {SPLITS_DIR}:")
    logger.info(f"  Train: {len(train_df):,} rows -> {train_path.name}")
    logger.info(f"  Val:   {len(val_df):,} rows -> {val_path.name}")
    logger.info(f"  Test:  {len(test_df):,} rows -> {test_path.name}")

    # Official valid patients
    if OFFICIAL_VALID_PATH.exists():
        val_meta = pd.read_csv(OFFICIAL_VALID_PATH)
        val_meta["patient_id"] = val_meta["path_to_image"].apply(extract_patient_id)
        val_meta.to_csv(SPLITS_DIR / "official_valid_patients.csv", index=False)
        logger.info(f"Saved official valid meta ({len(val_meta)} records) to splits/official_valid_patients.csv")

    # Generate and save class balance reports
    for s_df, s_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        report = generate_class_balance_report(s_df, s_name)
        report_path = REPORTS_DIR / f"class_balance_{s_name}.csv"
        report.to_csv(report_path, index=False)
        logger.info(f"Saved class balance report for {s_name} to {report_path.name}")

    logger.info("Splits and reports generation completed successfully.")


if __name__ == "__main__":
    main()
