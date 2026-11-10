"""
scripts/evaluate_ensemble_diversity.py
======================================
Comprehensive Ensemble Diversity & Oracle Upper-Bound Analysis
for EfficientNet-B0, B1, and B3 on the NIH ChestX-ray14 Sealed Test Set.

Calculates for each of the 14 thoracic diseases:
  1. Individual AUROC and PR-AUC (AUPRC).
  2. Individual F1-Score at fixed threshold (0.5) and calibrated optimal thresholds.
  3. Pairwise prediction-error correlation: r(e_i, e_j).
  4. Pairwise disagreement rate: D(M_i, M_j).
  5. Double-fault rate: DF(M_i, M_j).
  6. Oracle ensemble performance (Accuracy, Sensitivity, Specificity, F1, and Headroom Gap).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

# Workspace path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms,
    CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR, CLASSES, NUM_CLASSES
)
from train_phase2_attention import MultiAttentionXRayModel

# Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDS_CACHE = OUTPUTS_DIR / "predictions" / "ensemble_test_predictions.npz"
METRICS_DIR = OUTPUTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "b0": {
        "backbone": "efficientnet_b0",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "best_attention_efficientnet_b0.pth",
        "thresholds": PROJECT_ROOT / "checkpoints" / "optimal_thresholds_efficientnet_b0.json",
    },
    "b1": {
        "backbone": "efficientnet_b1",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "best_attention_efficientnet_b1.pth",
        "thresholds": PROJECT_ROOT / "checkpoints" / "optimal_thresholds_efficientnet_b1.json",
    },
    "b3": {
        "backbone": "efficientnet_b3",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "best_attention_efficientnet_b3.pth",
        "thresholds": PROJECT_ROOT / "checkpoints" / "optimal_thresholds_efficientnet_b3.json",
    },
}

def load_model(backbone_name: str, checkpoint_path: Path, device: torch.device):
    """Instantiate and load a trained MultiAttentionXRayModel."""
    print(f"Loading {checkpoint_path.name}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

def load_thresholds():
    """Load pre-calibrated Youden's J optimal thresholds for each model."""
    thresholds = {}
    for m_key, cfg in MODEL_CONFIGS.items():
        with open(cfg["thresholds"], "r") as f:
            thresholds[m_key] = json.load(f)
    return thresholds

def collect_test_predictions(device: torch.device, force_recompute: bool = False, max_samples: int = None):
    """
    Run inference across the sealed test set once and cache probability arrays.
    Returns: dict of { 'b0': probs, 'b1': probs, 'b3': probs, 'labels': labels }
    """
    if PREDS_CACHE.exists() and not force_recompute and max_samples is None:
        print(f"[CACHE] Loading cached ensemble predictions from {PREDS_CACHE}...")
        data = np.load(PREDS_CACHE)
        return {
            "b0": data["probs_b0"],
            "b1": data["probs_b1"],
            "b3": data["probs_b3"],
            "labels": data["labels"]
        }

    print("[DATA] Initializing sealed NIH test split (20%, seed=42)...")
    full_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))
    n_total = len(full_ds)
    n_test = int(n_total * 0.20)
    n_val = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(42)
    _, _, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    indices = test_idx.indices
    if max_samples is not None:
        indices = indices[:max_samples]
        print(f"[TEST SUBSET] Evaluating on first {max_samples} test samples")

    test_sub = Subset(full_ds, indices)
    test_loader = DataLoader(test_sub, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[DATA] Test set size: {len(test_sub)} images ({len(test_loader)} batches)")

    # Load all 3 models onto device
    models = {
        m_key: load_model(cfg["backbone"], cfg["checkpoint"], device)
        for m_key, cfg in MODEL_CONFIGS.items()
    }

    probs_b0, probs_b1, probs_b3, all_labels = [], [], [], []
    t0 = time.time()

    print("[INFERENCE] Running unified forward pass for B0, B1, and B3...")
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            
            p_b0 = torch.sigmoid(models["b0"](images)).cpu().numpy()
            p_b1 = torch.sigmoid(models["b1"](images)).cpu().numpy()
            p_b3 = torch.sigmoid(models["b3"](images)).cpu().numpy()

            probs_b0.append(p_b0)
            probs_b1.append(p_b1)
            probs_b3.append(p_b3)
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(test_loader):
                processed = min((batch_idx + 1) * test_loader.batch_size, len(test_sub))
                elapsed = time.time() - t0
                speed = processed / max(elapsed, 1e-5)
                print(f"  Processed {processed}/{len(test_sub)} images ({processed/len(test_sub)*100:.1f}%) - {speed:.1f} img/s")

    arr_b0 = np.concatenate(probs_b0, axis=0)
    arr_b1 = np.concatenate(probs_b1, axis=0)
    arr_b3 = np.concatenate(probs_b3, axis=0)
    arr_labels = np.concatenate(all_labels, axis=0)

    if max_samples is None:
        print(f"[CACHE] Saving predictions to {PREDS_CACHE}...")
        np.savez_compressed(
            PREDS_CACHE,
            probs_b0=arr_b0,
            probs_b1=arr_b1,
            probs_b3=arr_b3,
            labels=arr_labels
        )

    return {
        "b0": arr_b0,
        "b1": arr_b1,
        "b3": arr_b3,
        "labels": arr_labels
    }

def pearson_correlation(x, y):
    """Compute Pearson correlation coefficient safely with constant-array protection."""
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(r) else float(r)

def compute_ensemble_metrics(predictions, thresholds):
    """
    Perform rigorous evaluation of:
    1. AUROC & PR-AUC
    2. F1 at common (0.5) and optimal thresholds
    3. Pairwise error correlation
    4. Pairwise disagreement rate
    5. Double-fault rate
    6. Oracle ensemble performance
    """
    y_true = predictions["labels"]
    p_b0 = predictions["b0"]
    p_b1 = predictions["b1"]
    p_b3 = predictions["b3"]
    n_samples, n_diseases = y_true.shape

    records = []

    pairs = [("b0", "b1"), ("b0", "b3"), ("b1", "b3")]

    for d, disease in enumerate(CLASSES):
        y_d = y_true[:, d]
        n_pos = int(y_d.sum())
        prevalence = n_pos / n_samples

        # 1. Individual AUROC & PR-AUC
        auroc_b0 = roc_auc_score(y_d, p_b0[:, d])
        auroc_b1 = roc_auc_score(y_d, p_b1[:, d])
        auroc_b3 = roc_auc_score(y_d, p_b3[:, d])

        prauc_b0 = average_precision_score(y_d, p_b0[:, d])
        prauc_b1 = average_precision_score(y_d, p_b1[:, d])
        prauc_b3 = average_precision_score(y_d, p_b3[:, d])

        # 2. Binary predictions at fixed (0.5) and optimal thresholds
        # Fixed 0.5
        bin_b0_50 = (p_b0[:, d] >= 0.5).astype(int)
        bin_b1_50 = (p_b1[:, d] >= 0.5).astype(int)
        bin_b3_50 = (p_b3[:, d] >= 0.5).astype(int)

        f1_b0_50 = f1_score(y_d, bin_b0_50, zero_division=0)
        f1_b1_50 = f1_score(y_d, bin_b1_50, zero_division=0)
        f1_b3_50 = f1_score(y_d, bin_b3_50, zero_division=0)

        # Calibrated optimal thresholds
        thr_b0 = thresholds["b0"][disease]
        thr_b1 = thresholds["b1"][disease]
        thr_b3 = thresholds["b3"][disease]

        bin_b0_opt = (p_b0[:, d] >= thr_b0).astype(int)
        bin_b1_opt = (p_b1[:, d] >= thr_b1).astype(int)
        bin_b3_opt = (p_b3[:, d] >= thr_b3).astype(int)

        f1_b0_opt = f1_score(y_d, bin_b0_opt, zero_division=0)
        f1_b1_opt = f1_score(y_d, bin_b1_opt, zero_division=0)
        f1_b3_opt = f1_score(y_d, bin_b3_opt, zero_division=0)

        acc_b0_opt = np.mean(bin_b0_opt == y_d)
        acc_b1_opt = np.mean(bin_b1_opt == y_d)
        acc_b3_opt = np.mean(bin_b3_opt == y_d)

        # 3. Error vectors (e = 1 if misclassified, 0 if correct) using optimal decisions
        e_b0 = (bin_b0_opt != y_d).astype(int)
        e_b1 = (bin_b1_opt != y_d).astype(int)
        e_b3 = (bin_b3_opt != y_d).astype(int)

        err_dict = {"b0": e_b0, "b1": e_b1, "b3": e_b3}
        bin_dict = {"b0": bin_b0_opt, "b1": bin_b1_opt, "b3": bin_b3_opt}

        # Pairwise error correlation: r(e_i, e_j)
        r_b0_b1 = pearson_correlation(e_b0, e_b1)
        r_b0_b3 = pearson_correlation(e_b0, e_b3)
        r_b1_b3 = pearson_correlation(e_b1, e_b3)

        # 4. Pairwise disagreement rate: D(M_i, M_j) = mean(y_i != y_j)
        d_b0_b1 = np.mean(bin_b0_opt != bin_b1_opt)
        d_b0_b3 = np.mean(bin_b0_opt != bin_b3_opt)
        d_b1_b3 = np.mean(bin_b1_opt != bin_b3_opt)

        # 5. Double-fault rate: DF(M_i, M_j) = mean(e_i == 1 and e_j == 1)
        df_b0_b1 = np.mean((e_b0 == 1) & (e_b1 == 1))
        df_b0_b3 = np.mean((e_b0 == 1) & (e_b3 == 1))
        df_b1_b3 = np.mean((e_b1 == 1) & (e_b3 == 1))

        # 6. Oracle ensemble performance:
        # Correct if ANY model is correct:
        # oracle_correct = (bin_b0 == y_d) | (bin_b1 == y_d) | (bin_b3 == y_d)
        # oracle_prediction = y_d if oracle_correct else 1 - y_d
        any_correct_opt = (bin_b0_opt == y_d) | (bin_b1_opt == y_d) | (bin_b3_opt == y_d)
        oracle_pred_opt = np.where(any_correct_opt, y_d, 1 - y_d)

        oracle_acc_opt = np.mean(oracle_pred_opt == y_d)
        oracle_f1_opt = f1_score(y_d, oracle_pred_opt, zero_division=0)
        
        # Sensitivity and Specificity of Oracle
        cm = confusion_matrix(y_d, oracle_pred_opt, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        oracle_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        oracle_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Also compute Oracle at fixed 0.5 for reference
        any_correct_50 = (bin_b0_50 == y_d) | (bin_b1_50 == y_d) | (bin_b3_50 == y_d)
        oracle_pred_50 = np.where(any_correct_50, y_d, 1 - y_d)
        oracle_acc_50 = np.mean(oracle_pred_50 == y_d)
        oracle_f1_50 = f1_score(y_d, oracle_pred_50, zero_division=0)

        # Headroom / Gap (Oracle vs Best Individual Model)
        best_f1_ind = max(f1_b0_opt, f1_b1_opt, f1_b3_opt)
        best_acc_ind = max(acc_b0_opt, acc_b1_opt, acc_b3_opt)
        f1_headroom = oracle_f1_opt - best_f1_ind
        acc_headroom = oracle_acc_opt - best_acc_ind

        records.append({
            "Disease": disease,
            "Prevalence": prevalence,
            "AUROC_B0": auroc_b0,
            "AUROC_B1": auroc_b1,
            "AUROC_B3": auroc_b3,
            "PRAUC_B0": prauc_b0,
            "PRAUC_B1": prauc_b1,
            "PRAUC_B3": prauc_b3,
            "F1_50_B0": f1_b0_50,
            "F1_50_B1": f1_b1_50,
            "F1_50_B3": f1_b3_50,
            "Thr_Opt_B0": thr_b0,
            "Thr_Opt_B1": thr_b1,
            "Thr_Opt_B3": thr_b3,
            "F1_Opt_B0": f1_b0_opt,
            "F1_Opt_B1": f1_b1_opt,
            "F1_Opt_B3": f1_b3_opt,
            "Acc_B0": acc_b0_opt,
            "Acc_B1": acc_b1_opt,
            "Acc_B3": acc_b3_opt,
            "ErrCorr_B0_B1": r_b0_b1,
            "ErrCorr_B0_B3": r_b0_b3,
            "ErrCorr_B1_B3": r_b1_b3,
            "Disagree_B0_B1": d_b0_b1,
            "Disagree_B0_B3": d_b0_b3,
            "Disagree_B1_B3": d_b1_b3,
            "DoubleFault_B0_B1": df_b0_b1,
            "DoubleFault_B0_B3": df_b0_b3,
            "DoubleFault_B1_B3": df_b1_b3,
            "Oracle_Acc": oracle_acc_opt,
            "Oracle_Sens": oracle_sens,
            "Oracle_Spec": oracle_spec,
            "Oracle_F1": oracle_f1_opt,
            "Best_Ind_F1": best_f1_ind,
            "F1_Headroom": f1_headroom,
            "Acc_Headroom": acc_headroom,
        })

    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Diversity & Oracle Upper Bound")
    parser.add_argument("--recompute", action="store_true", help="Force recomputing test set predictions")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples (for test debug)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    thresholds = load_thresholds()
    predictions = collect_test_predictions(device, force_recompute=args.recompute, max_samples=args.max_samples)

    print("\n[ANALYSIS] Calculating 6 ensemble diversity and performance metrics per disease...")
    df = compute_ensemble_metrics(predictions, thresholds)

    # Save to CSV and JSON
    csv_path = METRICS_DIR / "ensemble_diversity_metrics.csv"
    json_path = METRICS_DIR / "ensemble_diversity_metrics.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"[EXPORT] Saved metrics to {csv_path} and {json_path}")

    # Compute Macro Summaries
    print("\n" + "=" * 80)
    print(" SUMMARY: MACRO-AVERAGE ENSEMBLE METRICS (14 THORACIC PATHOLOGIES)")
    print("=" * 80)
    print(f"Macro AUROC:        B0: {df['AUROC_B0'].mean():.4f} | B1: {df['AUROC_B1'].mean():.4f} | B3: {df['AUROC_B3'].mean():.4f}")
    print(f"Macro PR-AUC:       B0: {df['PRAUC_B0'].mean():.4f} | B1: {df['PRAUC_B1'].mean():.4f} | B3: {df['PRAUC_B3'].mean():.4f}")
    print(f"Macro F1 (Fixed 0.5): B0: {df['F1_50_B0'].mean():.4f} | B1: {df['F1_50_B1'].mean():.4f} | B3: {df['F1_50_B3'].mean():.4f}")
    print(f"Macro F1 (Opt Thr):   B0: {df['F1_Opt_B0'].mean():.4f} | B1: {df['F1_Opt_B1'].mean():.4f} | B3: {df['F1_Opt_B3'].mean():.4f}")
    print("-" * 80)
    print(f"Pairwise Prediction-Error Correlation (r_err):")
    print(f"  B0 vs B1: {df['ErrCorr_B0_B1'].mean():.4f} | B0 vs B3: {df['ErrCorr_B0_B3'].mean():.4f} | B1 vs B3: {df['ErrCorr_B1_B3'].mean():.4f}")
    print(f"Pairwise Disagreement Rate: ")
    print(f"  B0 vs B1: {df['Disagree_B0_B1'].mean()*100:.2f}% | B0 vs B3: {df['Disagree_B0_B3'].mean()*100:.2f}% | B1 vs B3: {df['Disagree_B1_B3'].mean()*100:.2f}%")
    print(f"Pairwise Double-Fault Rate: ")
    print(f"  B0 vs B1: {df['DoubleFault_B0_B1'].mean()*100:.2f}% | B0 vs B3: {df['DoubleFault_B0_B3'].mean()*100:.2f}% | B1 vs B3: {df['DoubleFault_B1_B3'].mean()*100:.2f}%")
    print("-" * 80)
    print(f"ORACLE ENSEMBLE PERFORMANCE (Upper Bound for Stacking/Selector Head):")
    print(f"  Oracle Accuracy:    {df['Oracle_Acc'].mean()*100:.2f}% (vs Best Ind: {df[['Acc_B0','Acc_B1','Acc_B3']].max(axis=1).mean()*100:.2f}%) -> Headroom: +{df['Acc_Headroom'].mean()*100:.2f}%")
    print(f"  Oracle F1-Score:    {df['Oracle_F1'].mean():.4f} (vs Best Ind: {df['Best_Ind_F1'].mean():.4f}) -> Headroom: +{df['F1_Headroom'].mean():.4f}")
    print(f"  Oracle Sensitivity: {df['Oracle_Sens'].mean()*100:.2f}%")
    print(f"  Oracle Specificity: {df['Oracle_Spec'].mean()*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()

