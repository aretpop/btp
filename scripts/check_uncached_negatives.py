import os
from pathlib import Path
import pandas as pd

full_df = pd.read_csv('data/chexpert_subset/train_index.csv')
mat_df = pd.read_csv('data/chexpert_subset/chexpert_materialized_index.csv')
cached_stems = set(mat_df['file_id'])

labels = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# For each label, count how many 0.0s are not yet cached
print("Uncached confirmed 0.0 negatives per class:")
uncached_fids_by_class = {}
all_uncached_zero_fids = set()

for lbl in labels:
    sub = full_df[(full_df[lbl] == 0.0) & (~full_df['file_id'].isin(cached_stems))]
    uncached_fids_by_class[lbl] = set(sub['file_id'])
    all_uncached_zero_fids.update(sub['file_id'])
    print(f"  {lbl:28s}: {len(sub):6,d} uncached 0.0s (out of {int((full_df[lbl]==0.0).sum()):,d} total)")

print("-" * 60)
print(f"Total unique images with AT LEAST ONE confirmed 0.0 not yet cached: {len(all_uncached_zero_fids):,d}")

# Specifically for the rare/tail negatives:
rare_zero_fids = set()
for lbl in ['Atelectasis', 'Lung Lesion', 'Pneumonia', 'Pleural Other', 'Support Devices', 'Lung Opacity']:
    rare_zero_fids.update(uncached_fids_by_class[lbl])
print(f"Total unique images with 0.0 for rare classes (Atelectasis, Lung Lesion, Pneumonia, Pleural Other, Support Devices, Lung Opacity): {len(rare_zero_fids):,d}")

