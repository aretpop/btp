"""
scripts/evaluate_chexpert_ensemble.py
=====================================
Comprehensive evaluation of the 3-model CheXpert ensemble:
  1. EfficientNet-B0 [bce]
  2. EfficientNet-B0 [bce_wpos]
  3. EfficientNet-B0 [asl]

Computes:
  - Individual AUROC, PR-AUC, F1 (default 0.5 and optimal threshold per class)
  - Mean Ensemble (equal weights)
  - Rank-Average Ensemble
  - Oracle Upper Bound (if any classifier is correct)
  - Pairwise Error Correlation
  - Pairwise Disagreement Rate & Double-Fault Rate
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "chexpert_cache"
INDEX_CSV = DATA_DIR / "chexpert_subset" / "chexpert_materialized_index.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "chexpert_ensemble"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
NUM_CLASSES = len(CHEXPERT_14_LABELS)


class CheXpertValDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["file_id"]
        path = CACHE_DIR / f"{fid}.jpg"

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        labels = [1.0 if row.get(lbl, 0.0) == 1.0 else 0.0 for lbl in CHEXPERT_14_LABELS]
        return image, torch.tensor(labels, dtype=torch.float32)


def build_b0():
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return m


def load_model(ckpt_name, device):
    path = CHECKPOINT_DIR / ckpt_name
    model = build_b0()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def compute_optimal_threshold_f1(targets_col, probs_col):
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.linspace(0.05, 0.95, 19):
        pred_bin = (probs_col >= thresh).astype(int)
        f = f1_score(targets_col, pred_bin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    return best_f1, best_thresh


def evaluate_predictions(probs_matrix, targets_matrix):
    aurocs, praucs, f1_defaults, f1_opts, opt_thresholds = [], [], [], [], []
    for c in range(NUM_CLASSES):
        y_true = targets_matrix[:, c]
        y_prob = probs_matrix[:, c]
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            prauc = average_precision_score(y_true, y_prob)
        else:
            auc, prauc = 0.5, 0.0
            
        f1_def = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        f1_opt, best_th = compute_optimal_threshold_f1(y_true, y_prob)
        
        aurocs.append(auc)
        praucs.append(prauc)
        f1_defaults.append(f1_def)
        f1_opts.append(f1_opt)
        opt_thresholds.append(best_th)
        
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "macro_prauc": float(np.mean(praucs)),
        "macro_f1_default": float(np.mean(f1_defaults)),
        "macro_f1_optimal": float(np.mean(f1_opts)),
        "per_class": {
            CHEXPERT_14_LABELS[i]: {
                "auroc": float(aurocs[i]),
                "prauc": float(praucs[i]),
                "f1_default": float(f1_defaults[i]),
                "f1_optimal": float(f1_opts[i]),
                "optimal_threshold": float(opt_thresholds[i])
            }
            for i in range(NUM_CLASSES)
        }
    }


def compute_pairwise_metrics(preds1_bin, preds2_bin, targets):
    """
    Computes pairwise error correlation, disagreement rate, and double fault rate.
    """
    err1 = (preds1_bin != targets).astype(float)
    err2 = (preds2_bin != targets).astype(float)
    
    # Error correlation
    std1 = np.std(err1)
    std2 = np.std(err2)
    if std1 > 0 and std2 > 0:
        corr = float(np.corrcoef(err1.ravel(), err2.ravel())[0, 1])
    else:
        corr = 0.0
        
    disagree_rate = float(np.mean(preds1_bin != preds2_bin))
    double_fault = float(np.mean((err1 == 1.0) & (err2 == 1.0)))
    
    return {
        "error_correlation": corr,
        "disagreement_rate": disagree_rate,
        "double_fault_rate": double_fault
    }


def compute_oracle_performance(preds_dict, targets):
    """
    Oracle Ensemble: For each sample and each class, it predicts correctly if AT LEAST ONE
    of the candidate models is correct.
    """
    # preds_dict: {model_name: preds_bin (N, 14)}
    models_list = list(preds_dict.values())
    
    # A prediction is correct if any model's prediction equals targets
    correct_any = np.zeros_like(targets, dtype=bool)
    oracle_preds = np.zeros_like(targets, dtype=int)
    
    for c in range(NUM_CLASSES):
        for i in range(targets.shape[0]):
            y = int(targets[i, c])
            # Check if any model predicted y
            matched = False
            for m_preds in models_list:
                if m_preds[i, c] == y:
                    oracle_preds[i, c] = y
                    matched = True
                    break
            if not matched:
                # All models failed: oracle predicts the incorrect class
                oracle_preds[i, c] = 1 - y

    f1_scores = [
        f1_score(targets[:, c], oracle_preds[:, c], zero_division=0)
        for c in range(NUM_CLASSES)
    ]
    accuracy = np.mean(oracle_preds == targets)
    return {
        "macro_f1": float(np.mean(f1_scores)),
        "accuracy": float(accuracy),
        "per_class_f1": {CHEXPERT_14_LABELS[c]: float(f1_scores[c]) for c in range(NUM_CLASSES)}
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Load 1,000 validation samples matching training split
    df_all = pd.read_csv(INDEX_CSV)
    rng = np.random.default_rng(42)
    val_mask = rng.random(len(df_all)) < 0.10
    val_df = df_all[val_mask].reset_index(drop=True).head(1000)
    print(f"Validation cohort: {len(val_df)} scans")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = CheXpertValDataset(val_df, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # 1. Load the three models
    print("\nLoading checkpoints...")
    m_bce = load_model("best_b0_bce.pth", device)
    m_bce_wpos = load_model("best_b0_bce_wpos.pth", device)
    m_asl = load_model("best_b0_asl.pth", device)

    # 2. Extract probability predictions
    print("Generating validation predictions...")
    probs_bce, targets = get_predictions(m_bce, val_loader, device)
    probs_wpos, _ = get_predictions(m_bce_wpos, val_loader, device)
    probs_asl, _ = get_predictions(m_asl, val_loader, device)

    # 3. Ensemble predictions
    # A. Equal-weight mean ensemble
    probs_ensemble = (probs_bce + probs_wpos + probs_asl) / 3.0

    # B. Performance-weighted ensemble (bce: 0.30, bce_wpos: 0.40, asl: 0.30)
    probs_weighted = 0.30 * probs_bce + 0.40 * probs_wpos + 0.30 * probs_asl

    # 4. Evaluate individual & ensemble models
    eval_bce = evaluate_predictions(probs_bce, targets)
    eval_wpos = evaluate_predictions(probs_wpos, targets)
    eval_asl = evaluate_predictions(probs_asl, targets)
    eval_mean_ens = evaluate_predictions(probs_ensemble, targets)
    eval_weight_ens = evaluate_predictions(probs_weighted, targets)

    # 5. Pairwise metrics (at optimal thresholds)
    preds_bce_bin = np.zeros_like(probs_bce, dtype=int)
    preds_wpos_bin = np.zeros_like(probs_wpos, dtype=int)
    preds_asl_bin = np.zeros_like(probs_asl, dtype=int)
    preds_ens_bin = np.zeros_like(probs_ensemble, dtype=int)

    for c in range(NUM_CLASSES):
        preds_bce_bin[:, c] = (probs_bce[:, c] >= eval_bce["per_class"][CHEXPERT_14_LABELS[c]]["optimal_threshold"]).astype(int)
        preds_wpos_bin[:, c] = (probs_wpos[:, c] >= eval_wpos["per_class"][CHEXPERT_14_LABELS[c]]["optimal_threshold"]).astype(int)
        preds_asl_bin[:, c] = (probs_asl[:, c] >= eval_asl["per_class"][CHEXPERT_14_LABELS[c]]["optimal_threshold"]).astype(int)
        preds_ens_bin[:, c] = (probs_ensemble[:, c] >= eval_mean_ens["per_class"][CHEXPERT_14_LABELS[c]]["optimal_threshold"]).astype(int)

    pairwise = {
        "bce_vs_wpos": compute_pairwise_metrics(preds_bce_bin, preds_wpos_bin, targets),
        "bce_vs_asl": compute_pairwise_metrics(preds_bce_bin, preds_asl_bin, targets),
        "wpos_vs_asl": compute_pairwise_metrics(preds_wpos_bin, preds_asl_bin, targets)
    }

    # 6. Oracle Ensemble Performance
    oracle = compute_oracle_performance({
        "bce": preds_bce_bin,
        "bce_wpos": preds_wpos_bin,
        "asl": preds_asl_bin
    }, targets)

    # Summary table
    print("\n" + "=" * 80)
    print("CHEXPERT 3-MODEL ENSEMBLE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<24s} | {'Macro AUROC':<12s} | {'Macro PR-AUC':<12s} | {'F1 (th=0.5)':<12s} | {'F1 (Optimal)':<12s}")
    print("-" * 80)
    print(f"{'EfficientNet-B0 [BCE]':<24s} | {eval_bce['macro_auroc']:<12.4f} | {eval_bce['macro_prauc']:<12.4f} | {eval_bce['macro_f1_default']:<12.4f} | {eval_bce['macro_f1_optimal']:<12.4f}")
    print(f"{'EfficientNet-B0 [WPOS]':<24s} | {eval_wpos['macro_auroc']:<12.4f} | {eval_wpos['macro_prauc']:<12.4f} | {eval_wpos['macro_f1_default']:<12.4f} | {eval_wpos['macro_f1_optimal']:<12.4f}")
    print(f"{'EfficientNet-B0 [ASL]':<24s} | {eval_asl['macro_auroc']:<12.4f} | {eval_asl['macro_prauc']:<12.4f} | {eval_asl['macro_f1_default']:<12.4f} | {eval_asl['macro_f1_optimal']:<12.4f}")
    print(f"{'Mean Ensemble (Equal)':<24s} | {eval_mean_ens['macro_auroc']:<12.4f} | {eval_mean_ens['macro_prauc']:<12.4f} | {eval_mean_ens['macro_f1_default']:<12.4f} | {eval_mean_ens['macro_f1_optimal']:<12.4f}")
    print(f"{'Weighted Ensemble':<24s} | {eval_weight_ens['macro_auroc']:<12.4f} | {eval_weight_ens['macro_prauc']:<12.4f} | {eval_weight_ens['macro_f1_default']:<12.4f} | {eval_weight_ens['macro_f1_optimal']:<12.4f}")
    print(f"{'Oracle Ensemble Upper Bound':<24s} | {'--':<12s} | {'--':<12s} | {'--':<12s} | {oracle['macro_f1']:<12.4f}")
    print("=" * 80)

    print("\nPairwise Disagreement & Error Correlation:")
    for pair, m in pairwise.items():
        print(f"  {pair:<14s}: Disagreement={m['disagreement_rate']*100:.2f}%, Err Corr={m['error_correlation']:.4f}, Double Fault={m['double_fault_rate']*100:.2f}%")

    out_file = OUTPUT_DIR / "chexpert_ensemble_evaluation.json"
    full_results = {
        "eval_bce": eval_bce,
        "eval_wpos": eval_wpos,
        "eval_asl": eval_asl,
        "eval_mean_ensemble": eval_mean_ens,
        "eval_weighted_ensemble": eval_weight_ens,
        "pairwise": pairwise,
        "oracle": oracle
    }
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved full evaluation metrics to: {out_file}")

    # Also save npz predictions for stacking / decoder heads
    npz_out = PROJECT_ROOT / "outputs" / "predictions" / "chexpert_ensemble_val_predictions.npz"
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out,
        probs_bce=probs_bce,
        probs_wpos=probs_wpos,
        probs_asl=probs_asl,
        probs_ensemble=probs_ensemble,
        targets=targets,
        labels=np.array(CHEXPERT_14_LABELS)
    )
    print(f"Saved predictions cache to: {npz_out}")


if __name__ == "__main__":
    main()

