"""
scripts/smoke_test_model.py
Smoke tests for CheXpertB1V2Model and CheXpertLocalDatasetV2.
"""
import sys
sys.path.insert(0, 'c:\\Users\\savi2\\OneDrive\\Desktop\\btp')
import torch
from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from src.datasets.chexpert_local_v2 import CheXpertLocalDatasetV2, make_patient_splits, TieredSamplerV2

# Test 1: Model at 224x224
cfg = {"input_resolution": 224, "fuse_block5": True, "attn_pool_mode": "norm_avg", "feat_dim": 256, "num_classes": 14}
model = CheXpertB1V2Model(cfg)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    logits, attn_reg = model(x)
assert logits.shape == (2, 14), f"Expected (2,14), got {logits.shape}"
print(f"[PASS] 224x224 logits shape: {logits.shape}")

# Test 2: Model at 320x320
cfg2 = {"input_resolution": 320, "fuse_block5": True, "attn_pool_mode": "norm_avg", "feat_dim": 256, "num_classes": 14}
model2 = CheXpertB1V2Model(cfg2)
x2 = torch.randn(2, 3, 320, 320)
with torch.no_grad():
    logits2, _ = model2(x2)
assert logits2.shape == (2, 14), f"Expected (2,14), got {logits2.shape}"
print(f"[PASS] 320x320 logits shape: {logits2.shape}")

# Test 3: Attention pool modes
for mode in ["sum", "norm_avg", "softmax"]:
    cfg3 = {"input_resolution": 224, "fuse_block5": True, "attn_pool_mode": mode, "feat_dim": 256}
    m = CheXpertB1V2Model(cfg3)
    with torch.no_grad():
        out, _ = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 14)
    print(f"[PASS] attn_pool_mode={mode} works")

# Test 4: fuse_block5=False
cfg4 = {"input_resolution": 224, "fuse_block5": False, "attn_pool_mode": "norm_avg"}
m4 = CheXpertB1V2Model(cfg4)
with torch.no_grad():
    out4, _ = m4(torch.randn(2, 3, 224, 224))
assert out4.shape == (2, 14)
print(f"[PASS] fuse_block5=False works")

# Test 5: disable_dilation3
cfg5 = {"input_resolution": 224, "disable_dilation3": True}
m5 = CheXpertB1V2Model(cfg5)
with torch.no_grad():
    out5, _ = m5(torch.randn(2, 3, 224, 224))
assert out5.shape == (2, 14)
print(f"[PASS] disable_dilation3=True works")

# Test 6: Dataset split (small)
import pandas as pd
df = pd.read_csv('c:\\Users\\savi2\\OneDrive\\Desktop\\btp\\data\\chexpert_subset\\chexpert_materialized_index.csv')
train_df, val_df, test_df = make_patient_splits(df, val_frac=0.10, test_frac=0.10)
print(f"[PASS] Patient splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
# Verify no patient leakage
import re
get_pid = lambda df: set(df['local_filename'].apply(lambda x: re.search(r'(patient\d+)', str(x)).group(1) if re.search(r'(patient\d+)', str(x)) else ''))
train_pids = get_pid(train_df)
val_pids = get_pid(val_df)
test_pids = get_pid(test_df)
assert len(train_pids & val_pids) == 0, "PATIENT LEAKAGE: train/val"
assert len(train_pids & test_pids) == 0, "PATIENT LEAKAGE: train/test"
assert len(val_pids & test_pids) == 0, "PATIENT LEAKAGE: val/test"
print(f"[PASS] Zero patient leakage confirmed")

print("\nAll smoke tests passed!")
