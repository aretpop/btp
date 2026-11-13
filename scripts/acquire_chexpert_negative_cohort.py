"""
scripts/acquire_chexpert_negative_cohort.py
===========================================
Acquire confirmed true negative (0.0) cases from Redivis for CheXpert+
to balance the true-negative class distribution.

Prioritizes rarest confirmed negatives first (Atelectasis, Lung Lesion,
Pneumonia, Pleural Other, Support Devices, Fracture, etc.), downloads
full images into data/chexpert_cache, and updates the materialized index.
"""

import os
import io
import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "chexpert_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = DATA_DIR / "chexpert_subset" / "train_index.csv"
REPORT_INDEX_CSV = DATA_DIR / "chexpert_subset" / "train_report_index.csv"
MATERIALIZED_INDEX_CSV = DATA_DIR / "chexpert_subset" / "chexpert_materialized_index.csv"

DEFAULT_API_TOKEN = os.environ.get("REDIVIS_API_TOKEN", "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")

# Rarest negatives first so each download maximizes coverage of under-represented negative classes
TARGET_NEG_PRIORITY = [
    "Pleural Other",
    "Atelectasis",
    "Support Devices",
    "Lung Lesion",
    "Pneumonia",
    "Lung Opacity",
    "Fracture",
    "Edema",
    "Consolidation",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
]

ALL_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("acquire_negatives")


def get_redivis_table(token: str):
    os.environ["REDIVIS_API_TOKEN"] = token
    import redivis
    user = redivis.user("aimi")
    dataset = user.dataset("chexpert_plus:5yyj:v1_0")
    return dataset.table("png_train:s6cj")


def load_report_mapping():
    reports = {}
    if REPORT_INDEX_CSV.exists():
        log.info(f"Loading clinical reports from {REPORT_INDEX_CSV.name}...")
        df_rep = pd.read_csv(REPORT_INDEX_CSV)
        for _, row in df_rep.iterrows():
            fid = row["file_id"]
            findings = str(row.get("section_findings", "")) if pd.notna(row.get("section_findings")) else ""
            impression = str(row.get("section_impression", "")) if pd.notna(row.get("section_impression")) else ""
            reports[fid] = (findings, impression)
    log.info(f"Loaded reports for {len(reports):,} studies")
    return reports


def download_single_image(table, file_id: str, file_name: str, cache_dir: Path, retries: int = 3) -> bool:
    out_path = cache_dir / f"{file_id}.jpg"
    if out_path.exists():
        return True

    for attempt in range(retries):
        try:
            f = table.file(file_name)
            with f.open("rb") as fh:
                data = fh.read()
            
            # Validate and convert to JPEG
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(out_path, "JPEG", quality=92)
            return True
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"Download failed for {file_id} ({file_name}): {e}")
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def main():
    parser = argparse.ArgumentParser(description="Acquire Targeted CheXpert True Negative Cohort")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max 0.0 samples to acquire per class")
    parser.add_argument("--max-total", type=int, default=None, help="Max total new images to download")
    parser.add_argument("--rare-only", action="store_true", help="Download only the rare negative classes (<1000 0.0s)")
    parser.add_argument("--token", type=str, default=DEFAULT_API_TOKEN, help="Redivis API token")
    args = parser.parse_args()

    log.info("Loading full train metadata...")
    full_df = pd.read_csv(INDEX_CSV)
    
    log.info("Loading current materialized index...")
    mat_df = pd.read_csv(MATERIALIZED_INDEX_CSV)
    cached_stems = set(mat_df["file_id"])
    log.info(f"Current local materialized index has {len(mat_df):,} images.")

    report_map = load_report_mapping()

    # Determine which classes to acquire
    classes_to_process = (
        ["Pleural Other", "Atelectasis", "Support Devices", "Lung Lesion", "Pneumonia", "Lung Opacity", "Fracture"]
        if args.rare_only else TARGET_NEG_PRIORITY
    )

    log.info("Identifying uncached confirmed 0.0 samples...")
    selected_fids = []
    selected_set = set()

    for cls_name in classes_to_process:
        if cls_name not in full_df.columns:
            continue
        
        # Candidates: confirmed 0.0, not already cached, not already selected in earlier pass
        mask = (full_df[cls_name] == 0.0) & (~full_df["file_id"].isin(cached_stems)) & (~full_df["file_id"].isin(selected_set))
        candidates = full_df[mask]
        
        if args.max_per_class is not None:
            candidates = candidates.head(args.max_per_class)
            
        fids = candidates["file_id"].tolist()
        for fid in fids:
            if fid not in selected_set:
                selected_set.add(fid)
                selected_fids.append(fid)

        log.info(f"  {cls_name:28s}: selected {len(fids):5,d} uncached 0.0s (total unique so far: {len(selected_fids):,d})")

    if args.max_total is not None and len(selected_fids) > args.max_total:
        selected_fids = selected_fids[:args.max_total]
        log.info(f"Trimmed total download target to {len(selected_fids):,} images per --max-total.")

    log.info(f"Final download list: {len(selected_fids):,} unique images to download.")

    if len(selected_fids) == 0:
        log.info("All requested 0.0 negatives are already cached locally!")
        return

    # Prepare download tuples (fid, local_filename)
    sub_df = full_df[full_df["file_id"].isin(selected_fids)].copy()
    items_to_download = list(zip(sub_df["file_id"], sub_df["local_filename"]))

    log.info("Connecting to Redivis table 'png_train:s6cj'...")
    table = get_redivis_table(args.token)

    log.info(f"Starting parallel download with {args.workers} workers...")
    t0 = time.time()
    success_fids = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_single_image, table, fid, fn, CACHE_DIR): fid
            for fid, fn in items_to_download
        }

        for idx, future in enumerate(as_completed(futures)):
            fid = futures[future]
            if future.result():
                success_fids.add(fid)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(items_to_download):
                elapsed = time.time() - t0
                speed = (idx + 1) / max(elapsed, 1e-5)
                log.info(f"  Progress: {idx + 1}/{len(items_to_download)} ({len(success_fids):,} saved, {speed:.1f} img/s)")

    log.info(f"Download complete: {len(success_fids):,} images successfully saved to {CACHE_DIR.name}.")

    if len(success_fids) == 0:
        log.warning("No new images were downloaded. Materialized index remains unchanged.")
        return

    # Append new records to materialized index
    log.info("Updating materialized index...")
    new_rows = full_df[full_df["file_id"].isin(success_fids)].copy()
    new_records = []

    for _, row in new_rows.iterrows():
        fid = row["file_id"]
        pos_diseases = [d for d in ALL_14_LABELS if row.get(d) == 1.0]
        neg_diseases = [d for d in ALL_14_LABELS if row.get(d) == 0.0]
        rep_findings, rep_impression = report_map.get(fid, ("", ""))

        rec = {
            "file_id": fid,
            "local_filename": row["local_filename"],
            "path_to_image": row.get("path_to_image", ""),
            "local_path": str(CACHE_DIR / f"{fid}.jpg"),
            "confirmed_positives": ";".join(pos_diseases),
            "num_confirmed": len(pos_diseases),
            "section_findings": rep_findings,
            "section_impression": rep_impression,
            "download_status": "negative_acquisition"
        }
        for lbl in ALL_14_LABELS:
            rec[lbl] = row.get(lbl, np.nan)
        new_records.append(rec)

    new_df = pd.DataFrame(new_records)
    updated_mat_df = pd.concat([mat_df, new_df], ignore_index=True)
    # Deduplicate just in case
    updated_mat_df = updated_mat_df.drop_duplicates(subset=["file_id"])
    updated_mat_df.to_csv(MATERIALIZED_INDEX_CSV, index=False)
    log.info(f"Successfully updated {MATERIALIZED_INDEX_CSV.name}: {len(updated_mat_df):,} total images.")


if __name__ == "__main__":
    main()

