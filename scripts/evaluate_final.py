"""
=============================================================================
evaluate_final.py
=============================================================================
Final, scientifically rigorous evaluation of the best trained checkpoint.

Data protocol (70 / 10 / 20 split, seed=42):
  - 70% Training set   : used to train weights          
  - 10% Validation set : used to select best checkpoint AND find optimal thresholds
  - 20% TEST SET       : SEALED — opened ONLY here for final metrics

This script:
  1. Reconstructs the exact same 3-way split used during training
  2. Runs inference on the Validation set to find optimal thresholds via Youden's J
  3. Runs inference on the SEALED test set using the Validation thresholds
  4. Reports AUROC, F1, Sensitivity, Specificity on the test set
=============================================================================
"""

import sys, torch, json, argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve,
    classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms,
    CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR, CLASSES, NUM_CLASSES
)

def evaluate_subset(model, loader, device):
    """Run inference on a subset, return (probs, labels) as numpy arrays"""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            out = model(images.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(labels)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx+1)*128} images...")
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()

def main(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("=" * 70)
    print(" RIGOROUS THRESHOLD CALIBRATION & TEST EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset splits
    full_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))
    n_total = len(full_ds)
    n_test  = int(n_total * 0.20)
    n_val   = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(42)
    _train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    val_loader = DataLoader(Subset(full_ds, val_idx.indices), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(full_ds, test_idx.indices), batch_size=128, shuffle=False, num_workers=0)
    print(f"Split: {n_train} train / {n_val} val / {n_test} test (sealed)")

    # 2. Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    print(f"Loading {backbone_name} from {checkpoint_path.name} (Epoch {ckpt.get('epoch', 'N/A')})")
    
    if args.is_anatomy:
        from anatomy_attention.anatomy_attention_model import AnatomyDiseaseAttentionModel
        config = ckpt.get("config")
        model = AnatomyDiseaseAttentionModel(config)
    elif args.is_attention:
        from train_phase2_attention import MultiAttentionXRayModel
        # Load the base model first to act as the frozen feature extractor
        base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
        in_features = base_model.backbone.classifier[1].in_features
        model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    else:
        model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
        
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # 3. Validation Pass - Find Optimal Thresholds (No Data Leakage!)
    print("\n--- PHASE 1: Calibrating Thresholds on VALIDATION set ---")
    val_probs, val_labels = evaluate_subset(model, val_loader, device)
    
    optimal_thresholds = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, thresholds = roc_curve(val_labels[:, i], val_probs[:, i])
        J = tpr - fpr
        best_idx = J.argmax()
        optimal_thresholds[cls] = float(thresholds[best_idx])

    # Save the calibrated thresholds for deployment / predict_xray.py
    thresh_path = CHECKPOINT_DIR / f"optimal_thresholds_{backbone_name}.json"
    with open(thresh_path, "w") as f:
        json.dump({k: round(v, 4) for k, v in optimal_thresholds.items()}, f, indent=2)
    print(f"Thresholds calibrated and saved to {thresh_path.name}")

    # 4. Test Pass - Final Evaluation
    print("\n--- PHASE 2: Evaluating on SEALED TEST set ---")
    test_probs, test_labels = evaluate_subset(model, test_loader, device)

    per_class_metrics = {}
    for i, cls in enumerate(CLASSES):
        # AUROC is threshold-independent, but we still calculate it here
        auroc = roc_auc_score(test_labels[:, i], test_probs[:, i])
        
        # Apply the threshold derived from the validation set
        opt_thr = optimal_thresholds[cls]
        preds = (test_probs[:, i] >= opt_thr).astype(int)
        
        tp = ((preds == 1) & (test_labels[:, i] == 1)).sum()
        fp = ((preds == 1) & (test_labels[:, i] == 0)).sum()
        tn = ((preds == 0) & (test_labels[:, i] == 0)).sum()
        fn = ((preds == 0) & (test_labels[:, i] == 1)).sum()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = f1_score(test_labels[:, i], preds, zero_division=0)
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)

        per_class_metrics[cls] = {
            "auroc": auroc,
            "threshold": opt_thr,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "acc": acc
        }

    macro_auroc = np.mean([m["auroc"] for m in per_class_metrics.values()])
    macro_f1    = np.mean([m["f1"] for m in per_class_metrics.values()])
    macro_acc   = np.mean([m["acc"] for m in per_class_metrics.values()])

    # 5. Report
    sep = "=" * 78
    report_lines = [
        sep,
        f" FINAL TEST SET EVALUATION REPORT ({backbone_name})",
        sep,
        f" Macro-AUROC: {macro_auroc:.4f}",
        f" Macro-F1:    {macro_f1:.4f}",
        f" Macro-Acc:   {macro_acc:.4f}",
        sep,
        f" {'Class':<22} {'AUROC':>6} {'Thr':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'Acc':>6}",
        "-" * 78,
    ]

    for cls, m in per_class_metrics.items():
        report_lines.append(
            f" {cls:<22} {m['auroc']:6.4f} {m['threshold']:6.3f} "
            f"{m['sensitivity']:6.3f} {m['specificity']:6.3f} {m['f1']:6.3f} {m['acc']:6.3f}"
        )
    report_lines.append(sep)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_name = args.report_name if args.report_name else f"evaluation_report_{backbone_name}.txt"
    report_path = CHECKPOINT_DIR / out_name
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--is-attention", action="store_true", help="Set to True if evaluating a Phase 2 MultiAttention model")
    parser.add_argument("--is-anatomy", action="store_true", help="Set to True if evaluating an AnatomyAttention model")
    parser.add_argument("--report-name", type=str, default=None, help="Custom output report filename")
    args = parser.parse_args()
    main(args)
