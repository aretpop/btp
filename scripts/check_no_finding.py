import pandas as pd

full_df = pd.read_csv('data/chexpert_subset/train_index.csv')
no_finding = full_df[full_df['No Finding'] == 1.0]
print(f"Total 'No Finding' == 1.0 rows in train_index: {len(no_finding):,d}")

labels = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

print("In rows where No Finding == 1.0:")
for lbl in labels:
    vals = no_finding[lbl].value_counts(dropna=False).to_dict()
    print(f"  {lbl:28s}: {vals}")

