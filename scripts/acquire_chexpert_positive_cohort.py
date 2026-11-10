"""
scripts/acquire_chexpert_positive_cohort.py
===========================================
Multi-Label Positive-Class Acquisition with Global Deduplication for CheXpert+

Acquires confirmed positive (1.0) chest X-ray cases from Redivis across 12 target diseases
with full global deduplication, report text pairing (for downstream decoder training),
resumable progress tracking, and validation checks.

Target Distribution:
  Pneumonia                  : 700
  Pleural Other              : 1,117
  Fracture                   : 2,648
  Lung Lesion                : 2,584
  Consolidation              : 3,515
  Enlarged Cardiomediastinum : 3,413
  Pneumothorax               : 5,785
  Atelectasis                : 7,821
  Cardiomegaly               : 9,213
  Edema                      : 11,420
  Pleural Effusion           : 15,000
  Lung Opacity               : 16,081
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

# 12 Target Pathologies in Priority Order (rarest first to maximize multi-label reuse)
TARGET_DISTRIBUTION = {
    "Pneumonia": 700,
    "Pleural Other": 1117,
    "Fracture": 2648,
    "Lung Lesion": 2584,
    "Consolidation": 3515,
    "Enlarged Cardiomediastinum": 3413,
    "Pneumothorax": 5785,
    "Atelectasis": 7821,
    "Cardiomegaly": 9213,
    "Edema": 11420,
    "Pleural Effusion": 15000,
    "Lung Opacity": 16081
}

ALL_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("acquire_chexpert")


def get_redivis_table(token: str):
    os.environ["REDIVIS_API_TOKEN"] = token
    import redivis
    user = redivis.user("aimi")
    dataset = user.dataset("chexpert_plus:5yyj:v1_0")
    return dataset.table("png_train:s6cj")


def load_report_mapping():
    """Load text reports (findings + impression) indexed by file_id."""
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
    """Download image, validate integrity, and save as JPEG."""
    out_path = cache_dir / f"{file_id}.jpg"
    if out_path.exists():
        return True

    for attempt in range(retries):
        try:
            f = table.file(file_name)
            with f.open("rb") as fh:
                data = fh.read()
            
            # Validate image integrity
            img = Image.open(io.BytesIO(data))
            img.verify()
            
            # Re-open and save as optimized JPEG
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(out_path, "JPEG", quality=92)
            return True
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"Download failed for {file_id} ({file_name}): {e}")
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def build_or_load_materialized_index(full_df: pd.DataFrame, report_map: dict):
    """
    Construct or update the global deduplicated metadata index.
    Checks existing files in CACHE_DIR and merges with train_index.csv annotations and reports.
    """
    log.info("Scanning existing cached images in local repository...")
    t0 = time.time()
    cached_files = os.listdir(CACHE_DIR)
    cached_stems = set(os.path.splitext(f)[0] for f in cached_files)
    log.info(f"Found {len(cached_stems):,} local images on disk (scan took {time.time()-t0:.2f}s).")

    records = []
    current_counts = {d: 0 for d in TARGET_DISTRIBUTION}

    # Match existing cached images to metadata
    matched_df = full_df[full_df["file_id"].isin(cached_stems)].copy()

    for _, row in matched_df.iterrows():
        fid = row["file_id"]
        pos_diseases = [d for d in TARGET_DISTRIBUTION if row.get(d) == 1.0]
        for d in pos_diseases:
            current_counts[d] += 1

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
            "download_status": "existing"
        }
        for lbl in ALL_14_LABELS:
            rec[lbl] = row.get(lbl, np.nan)
        records.append(rec)

    index_df = pd.DataFrame(records)
    # Persist the baseline index
    index_df.to_csv(MATERIALIZED_INDEX_CSV, index=False)
    log.info(f"Initialized global materialized registry with {len(index_df):,} existing records at {MATERIALIZED_INDEX_CSV.name}.")
    return index_df, cached_stems, current_counts


def print_status_table(current_counts: dict, targets: dict):
    """Format and print real-time status table."""
    print("\n" + "=" * 80)
    print(f"{'Disease':28s} | {'Current Positives':18s} | {'Target':8s} | {'Remaining':10s} | {'Status':8s}")
    print("-" * 80)
    all_met = True
    for d, tgt in targets.items():
        cur = current_counts[d]
        rem = max(0, tgt - cur)
        status = "[DONE]" if rem == 0 else f"{rem:,} left"
        if rem > 0:
            all_met = False
        print(f"{d:28s} | {cur:18,d} | {tgt:8,d} | {rem:10,d} | {status:8s}")
    print("=" * 80 + "\n", flush=True)
    return all_met


def main():
    parser = argparse.ArgumentParser(description="Acquire Targeted CheXpert Positive Cohort")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    parser.add_argument("--chunk-size", type=int, default=25, help="Batch chunk size for downloads")
    parser.add_argument("--max-to-download", type=int, default=None, help="Optional safety limit on new images to download")
    parser.add_argument("--token", type=str, default=DEFAULT_API_TOKEN, help="Redivis API token")
    args = parser.parse_args()

    log.info(f"Loading CheXpert dataset metadata from {INDEX_CSV.name}...")
    full_df = pd.read_csv(INDEX_CSV)

    # Filter to frontal views
    full_df = full_df[full_df["local_filename"].str.contains("frontal", na=False)].reset_index(drop=True)
    log.info(f"Total eligible frontal scans: {len(full_df):,}")

    report_map = load_report_mapping()

    # Reconstruct current state & save baseline materialized index
    index_df, global_cached_stems, current_counts = build_or_load_materialized_index(full_df, report_map)

    # Initial Status Check
    print_status_table(current_counts, TARGET_DISTRIBUTION)

    # Connect to Redivis
    log.info("Connecting to Redivis table 'png_train:s6cj'...")
    table = get_redivis_table(args.token)

    total_downloaded_this_run = 0
    multi_label_contribution_count = 0
    t_start = time.time()

    # Acquisition Loop over priority diseases
    for disease, target in TARGET_DISTRIBUTION.items():
        remaining = target - current_counts[disease]
        if remaining <= 0:
            log.info(f"Target for {disease} ({target:,}) is already satisfied ({current_counts[disease]:,}). Skipping.")
            continue

        log.info(f"\n>>> ACQUIRING TARGET CLASS: {disease} (Needed: {remaining:,} to reach {target:,})")

        # Candidate selection: confirmed positive for this disease, frontal, not yet in global cache
        candidates = full_df[(full_df[disease] == 1.0) & (~full_df["file_id"].isin(global_cached_stems))].copy()
        
        if len(candidates) == 0:
            log.warning(f"No additional candidate samples available on Redivis for {disease}!")
            continue

        candidates = candidates.head(remaining)
        candidate_items = list(zip(candidates["file_id"], candidates["local_filename"]))
        log.info(f"Identified {len(candidate_items):,} candidates for {disease}. Downloading in chunks of {args.chunk_size}...")

        # Process in small chunks so we can stop immediately when target is reached
        for chunk_start in range(0, len(candidate_items), args.chunk_size):
            if current_counts[disease] >= target:
                log.info(f"Target for {disease} ({target:,}) reached! Stopping candidate downloads.")
                break

            if args.max_to_download and total_downloaded_this_run >= args.max_to_download:
                log.info(f"Reached max safety download limit ({args.max_to_download}). Stopping.")
                break

            chunk = candidate_items[chunk_start:chunk_start + args.chunk_size]
            batch_new_records = []

            with ThreadPoolExecutor(max_workers=min(args.workers, len(chunk))) as executor:
                futures = {
                    executor.submit(download_single_image, table, fid, fn, CACHE_DIR): (fid, fn)
                    for fid, fn in chunk
                }

                for future in as_completed(futures):
                    fid, fn = futures[future]
                    success = future.result()

                    if success:
                        total_downloaded_this_run += 1
                        global_cached_stems.add(fid)

                        # Extract complete multi-label vector from full_df
                        row_meta = full_df[full_df["file_id"] == fid].iloc[0]
                        pos_diseases = [d for d in TARGET_DISTRIBUTION if row_meta.get(d) == 1.0]

                        # Global deduplication update: increment positive count for EVERY confirmed disease in this image
                        for pos_d in pos_diseases:
                            current_counts[pos_d] += 1
                        
                        if len(pos_diseases) > 1:
                            multi_label_contribution_count += 1

                        rep_findings, rep_impression = report_map.get(fid, ("", ""))

                        rec = {
                            "file_id": fid,
                            "local_filename": fn,
                            "path_to_image": row_meta.get("path_to_image", ""),
                            "local_path": str(CACHE_DIR / f"{fid}.jpg"),
                            "confirmed_positives": ";".join(pos_diseases),
                            "num_confirmed": len(pos_diseases),
                            "section_findings": rep_findings,
                            "section_impression": rep_impression,
                            "download_status": "downloaded"
                        }
                        for lbl in ALL_14_LABELS:
                            rec[lbl] = row_meta.get(lbl, np.nan)
                        batch_new_records.append(rec)

            # Persist batch updates to disk immediately
            if batch_new_records:
                new_df = pd.DataFrame(batch_new_records)
                index_df = pd.concat([index_df, new_df], ignore_index=True)
                index_df.to_csv(MATERIALIZED_INDEX_CSV, index=False)

            elapsed = time.time() - t_start
            speed = total_downloaded_this_run / max(elapsed, 1e-5)
            log.info(f"Chunk progress ({disease}): {min(chunk_start + args.chunk_size, len(candidate_items))}/{len(candidate_items)} "
                     f"| Disease count: {current_counts[disease]}/{target} | Overall speed: {speed:.1f} img/s")

        # Print status table after completing this disease
        all_satisfied = print_status_table(current_counts, TARGET_DISTRIBUTION)

        if args.max_to_download and total_downloaded_this_run >= args.max_to_download:
            break

        if all_satisfied:
            log.info("ALL 12 DISEASE TARGETS HAVE BEEN FULLY SATISFIED!")
            break

    # Final Summary
    elapsed_total = time.time() - t_start
    log.info("\n" + "=" * 80)
    log.info("ACQUISITION RUN COMPLETE")
    log.info(f"  Total unique images downloaded this run : {total_downloaded_this_run:,}")
    log.info(f"  Total unique images available locally   : {len(global_cached_stems):,}")
    log.info(f"  Multi-label co-occurrence contributions : {multi_label_contribution_count:,}")
    log.info(f"  Total runtime                           : {elapsed_total/60:.2f} minutes")
    log.info(f"  Centralized metadata index saved to     : {MATERIALIZED_INDEX_CSV}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
