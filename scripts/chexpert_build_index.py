"""
Phase 1: Build the training index
- Download findings_fixed.json (labels for ALL images)
- Get PNG_train file list with file_ids
- Join → save train_index.csv + valid_index.csv
"""
import os
os.environ["REDIVIS_API_TOKEN"] = "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk"

import redivis
import pandas as pd
import json
from pathlib import Path

pd.set_option('display.max_columns', None)

user = redivis.user("aimi")
dataset = user.dataset("chexpert_plus:5yyj:v1_0")

OUTPUT_DIR = Path("data/chexpert_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# ================================================================
# [1] Download + parse findings_fixed.json (ALL disease labels)
# ================================================================
labels_path = OUTPUT_DIR / "findings_fixed.json"

if not labels_path.exists():
    print("[1] Downloading findings_fixed.json (~84MB)...")
    table_labels = dataset.table("chexpert_labels:pmec")
    files = {f.name: f for f in table_labels.list_files()}
    f = files['findings_fixed.json']
    f.download(str(labels_path))
    print(f"    Saved to {labels_path}")
else:
    print(f"[1] Already downloaded: {labels_path}")

print("    Parsing JSONL...")
records = []
with open(labels_path, 'r') as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df_labels = pd.DataFrame(records)
print(f"    Total label records: {len(df_labels):,}")
print(f"    Train: {df_labels['path_to_image'].str.startswith('train/').sum():,}")
print(f"    Valid: {df_labels['path_to_image'].str.startswith('valid/').sum():,}")
print(f"    Test:  {df_labels['path_to_image'].str.startswith('test/').sum():,}")

# ================================================================
# [2] Get PNG_train file list (file_id -> filename mapping)
#     We only need file_id + file_name — no image bytes yet
# ================================================================
train_filelist_path = OUTPUT_DIR / "png_train_files.csv"

if not train_filelist_path.exists():
    print("\n[2] Fetching PNG_train file list (file_id + filename)...")
    print("    This may take a few minutes for 220k+ files...")
    table_train = dataset.table("PNG_train")
    df_train_files = table_train.to_pandas_dataframe()  # just metadata, no images
    df_train_files.columns = [c.lower() for c in df_train_files.columns]
    print(f"    PNG_train count: {len(df_train_files):,}")
    print(f"    Columns: {list(df_train_files.columns)}")
    df_train_files.to_csv(train_filelist_path, index=False)
    print(f"    Saved to {train_filelist_path}")
else:
    print(f"\n[2] Already downloaded: {train_filelist_path}")
    df_train_files = pd.read_csv(train_filelist_path)
    print(f"    PNG_train count: {len(df_train_files):,}")

# ================================================================
# [3] Also get PNG_valid file list
# ================================================================
valid_filelist_path = OUTPUT_DIR / "png_valid_files.csv"
if not valid_filelist_path.exists():
    print("\n[3] Fetching PNG_valid file list...")
    table_valid = dataset.table("PNG_valid:41v9")
    df_valid_files = table_valid.to_pandas_dataframe()
    df_valid_files.columns = [c.lower() for c in df_valid_files.columns]
    df_valid_files.to_csv(valid_filelist_path, index=False)
    print(f"    PNG_valid count: {len(df_valid_files):,}")
else:
    df_valid_files = pd.read_csv(valid_filelist_path)
    print(f"\n[3] PNG_valid already saved: {len(df_valid_files)} files")

# ================================================================
# [4] Build train_index.csv: join labels + file_ids
# ================================================================
print("\n[4] Building train index (labels + file_id)...")

def build_index(df_files, df_labels, split):
    """Join file metadata with disease labels on path_to_image."""
    # file_name in redivis: patient64643/study1/view1_frontal.png
    # path_to_image in labels: valid/patient64643/study1/view1_frontal.jpg
    df_files = df_files.copy()
    df_files['path_key'] = df_files['file_name'].astype(str).str.replace('.png', '.jpg', regex=False)
    df_files['path_key'] = split + '/' + df_files['path_key']

    df_split_labels = df_labels[df_labels['path_to_image'].str.startswith(split + '/')].copy()

    merged = df_split_labels.merge(
        df_files[['file_id', 'file_name', 'path_key']],
        left_on='path_to_image',
        right_on='path_key',
        how='inner'
    )
    # Add local filename (png)
    merged['local_filename'] = merged['file_name']
    merged = merged.drop(columns=['path_key'])
    return merged

df_train_index = build_index(df_train_files, df_labels, 'train')
df_valid_index = build_index(df_valid_files, df_labels, 'valid')

print(f"    Train index rows: {len(df_train_index):,}")
print(f"    Valid index rows: {len(df_valid_index):,}")

# Check label coverage
print(f"\n    Train label distribution (sample):")
for col in CHEXPERT_LABELS[:5]:
    if col in df_train_index.columns:
        pos = (df_train_index[col] == 1.0).sum()
        neg = (df_train_index[col] == 0.0).sum()
        unc = (df_train_index[col] == -1.0).sum()
        total = len(df_train_index)
        print(f"      {col:30s}: +{pos:6d} ({pos/total*100:.1f}%)  -{neg:6d}  ±{unc:6d}")

# Save indexes
train_idx_path = OUTPUT_DIR / "train_index.csv"
valid_idx_path = OUTPUT_DIR / "valid_index.csv"

df_train_index.to_csv(train_idx_path, index=False)
df_valid_index.to_csv(valid_idx_path, index=False)
print(f"\n    Saved: {train_idx_path}")
print(f"    Saved: {valid_idx_path}")
print(f"\n✅ Index building complete!")
print(f"   Train: {len(df_train_index):,} images  |  Valid: {len(df_valid_index):,} images")

# View types
if 'local_filename' in df_train_index.columns:
    frontal_mask = df_train_index['local_filename'].str.contains('frontal')
    print(f"   Frontal: {frontal_mask.sum():,}  |  Lateral: {(~frontal_mask).sum():,}")
