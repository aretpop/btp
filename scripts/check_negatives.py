import pandas as pd

full_df = pd.read_csv('data/chexpert_subset/train_index.csv')
mat_df = pd.read_csv('data/chexpert_subset/chexpert_materialized_index.csv')

labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

print(f"{'Class':28} | {'Local Cached 0.0':16} | {'Full Redivis 0.0':16} | {'Full Redivis 1.0':16}")
print("-" * 84)
for lbl in labels:
    if lbl in full_df.columns:
        loc_zero = int((mat_df[lbl] == 0.0).sum()) if lbl in mat_df.columns else 0
        full_zero = int((full_df[lbl] == 0.0).sum())
        full_pos = int((full_df[lbl] == 1.0).sum())
        print(f"{lbl:28} | {loc_zero:16,d} | {full_zero:16,d} | {full_pos:16,d}")
    else:
        print(f"{lbl:28} | {'N/A':16} | {'N/A':16} | {'N/A':16}")

