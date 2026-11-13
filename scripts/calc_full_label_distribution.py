import pandas as pd
import numpy as np

mat = pd.read_csv('data/chexpert_subset/chexpert_materialized_index.csv')
labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

print(f"{'Class':28} | {'Pos (1.0)':9} | {'Neg (0.0)':9} | {'Unc (-1.0)':10} | {'NaN (Blank)':11} | {'Total':9}")
print("-" * 85)
for lbl in labels:
    pos = int((mat[lbl] == 1.0).sum())
    neg = int((mat[lbl] == 0.0).sum())
    unc = int((mat[lbl] == -1.0).sum())
    nan = int(mat[lbl].isna().sum())
    tot = len(mat)
    print(f"{lbl:28} | {pos:9,d} | {neg:9,d} | {unc:10,d} | {nan:11,d} | {tot:9,d}")

