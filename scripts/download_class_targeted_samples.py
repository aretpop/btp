"""
scripts/download_class_targeted_samples.py
=========================================
Targeted Class-Specific Image Downloader from Redivis for CheXpert+.

Allows downloading additional positive samples for specific under-represented
classes (e.g. Pneumonia, Pleural Other, Fracture, Lung Lesion) to augment the local cache.

Usage:
  # Download all remaining confirmed positive (1.0) Pneumonia cases:
  python scripts/download_class_targeted_samples.py --disease Pneumonia

  # Download 500 cases of Pneumonia including uncertain (-1.0) cases:
  python scripts/download_class_targeted_samples.py --disease Pneumonia --include-uncertain --max-samples 500

  # Download multiple rare classes at once:
  python scripts/download_class_targeted_samples.py --disease "Pneumonia,Pleural Other,Fracture" --max-samples 300
"""

import os
import io
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "chexpert_subset" / "train_index.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "chexpert_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_API_TOKEN = os.environ.get("REDIVIS_API_TOKEN", "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")

def get_redivis_table(token: str):
    os.environ["REDIVIS_API_TOKEN"] = token
    import redivis
    user = redivis.user("aimi")
    dataset = user.dataset("chexpert_plus:5yyj:v1_0")
    return dataset.table("png_train:s6cj")

def download_single(table, file_id: str, file_name: str, cache_dir: Path) -> bool:
    """Download a single image from Redivis in a single fast read and save as JPEG."""
    out_path = cache_dir / f"{file_id}.jpg"
    if out_path.exists():
        return True

    try:
        f = table.file(file_name)
        with f.open("rb") as fh:
            data = fh.read()
        
        # Load and verify image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(out_path, "JPEG", quality=92)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {file_name}: {e}", flush=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Targeted Class Downloader for CheXpert+")
    parser.add_argument("--disease", type=str, required=True, help="Target disease(s), e.g. Pneumonia or 'Pneumonia,Fracture'")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to download per disease (None = all)")
    parser.add_argument("--include-uncertain", action="store_true", help="Include uncertain (-1.0) labels in addition to 1.0")
    parser.add_argument("--frontal-only", action="store_true", default=True, help="Download only frontal AP/PA views")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    parser.add_argument("--token", type=str, default=DEFAULT_API_TOKEN, help="Redivis API token")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        print(f"Error: Index file not found at {INDEX_PATH}")
        return

    print(f"[INDEX] Loading {INDEX_PATH.name}...")
    df = pd.read_csv(INDEX_PATH)

    # Scan existing cache
    print(f"[CACHE] Scanning existing cache in {CACHE_DIR.name}...", flush=True)
    existing_files = os.listdir(CACHE_DIR)
    cached_stems = set(os.path.splitext(f)[0] for f in existing_files)
    print(f"[CACHE] Currently cached: {len(cached_stems):,} images", flush=True)

    diseases = [d.strip() for d in args.disease.split(",")]
    
    # Initialize Redivis
    print(f"[REDIVIS] Connecting to table 'png_train:s6cj'...")
    table = get_redivis_table(args.token)

    for disease in diseases:
        if disease not in df.columns:
            print(f"[WARNING] Class '{disease}' not found in metadata columns. Skipping.")
            continue

        valid_vals = [1.0, -1.0] if args.include_uncertain else [1.0]
        subset = df[df[disease].isin(valid_vals)].copy()

        if args.frontal_only and "local_filename" in subset.columns:
            subset = subset[subset["local_filename"].str.contains("frontal", na=False)]

        total_pos = len(subset)
        uncached = subset[~subset["file_id"].isin(cached_stems)].copy()
        
        print("\n" + "=" * 70)
        print(f"TARGET CLASS: {disease}")
        print(f"  Total occurrences in dataset: {total_pos:,} (values: {valid_vals})")
        print(f"  Already in local cache:       {total_pos - len(uncached):,}")
        print(f"  Available for download:       {len(uncached):,}")
        print("=" * 70)

        if len(uncached) == 0:
            print(f"  All available {disease} samples are already downloaded locally!")
            continue

        if args.max_samples is not None:
            uncached = uncached.head(args.max_samples)
            print(f"  Limiting download to: {len(uncached):,} samples")

        items_to_download = list(zip(uncached["file_id"], uncached["local_filename"]))
        
        print(f"  Starting download of {len(items_to_download)} images with {args.workers} workers...")
        success_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_single, table, fid, fn, CACHE_DIR): fid
                for fid, fn in items_to_download
            }

            for idx, future in enumerate(as_completed(futures)):
                if future.result():
                    success_count += 1
                    cached_stems.add(futures[future])

                if (idx + 1) % 50 == 0 or (idx + 1) == len(items_to_download):
                    elapsed = time.time() - t0
                    speed = (idx + 1) / max(elapsed, 1e-5)
                    print(f"    Progress: {idx + 1}/{len(items_to_download)} ({success_count} saved, {speed:.1f} img/s)")

        print(f"  Finished {disease}: Successfully downloaded {success_count} new images.")

    print("\n[COMPLETE] All requested downloads finished.")

if __name__ == "__main__":
    main()
