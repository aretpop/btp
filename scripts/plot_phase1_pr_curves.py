"""
scripts/plot_phase1_pr_curves.py
=================================
Generates publication-quality Precision-Recall (PR) curves for Phase 1
(Vision Feature Extractor Baseline Classifier) on the NIH ChestX-ray14 dataset.

Features:
  1. Reconstructs exact 70/10/20 data split (seed=42) from Phase 1 training & final evaluation protocol.
  2. Runs high-throughput GPU inference and caches predictions (probabilities & ground truth) to disk.
  3. Computes per-class PR curves, Average Precision (AP / AUPRC), Micro-average, and Macro-average.
  4. Plots:
     - Combined multi-label PR curve plot with iso-F1 curves and AP legend.
     - 14-panel grid plot with class prevalence baselines and calibrated operating points.
  5. Exports detailed quantitative CSV and JSON metric summaries.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup path imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms,
    CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR, CLASSES, NUM_CLASSES
)

FIG_DIR = PROJECT_ROOT / "outputs" / "btp_report_figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDS_DIR = PROJECT_ROOT / "outputs" / "predictions"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase1_pr_curve")


def load_model(checkpoint_path: Path, device: torch.device):
    """Load the Phase 1 classifier model."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b1")
    logger.info(f"Checkpoint backbone: {backbone_name}, trained epoch: {ckpt.get('epoch', 'N/A')}")

    model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, backbone_name


def get_test_predictions(model, test_loader, device: torch.device, cache_file: Path, force_recompute: bool = False):
    """Run inference or load cached test predictions."""
    if cache_file.exists() and not force_recompute:
        logger.info(f"Loading cached predictions from {cache_file}")
        data = np.load(cache_file)
        return data["probs"], data["labels"]

    logger.info(f"Running inference on {len(test_loader.dataset)} test samples...")
    all_probs, all_labels = [], []
    total_batches = len(test_loader)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                processed = min((batch_idx + 1) * test_loader.batch_size, len(test_loader.dataset))
                logger.info(f"  Processed {processed}/{len(test_loader.dataset)} images ({processed/len(test_loader.dataset)*100:.1f}%)")

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    logger.info(f"Saving predictions cache to {cache_file}")
    np.savez_compressed(cache_file, probs=probs_arr, labels=labels_arr)
    return probs_arr, labels_arr


def plot_combined_pr_curve(pr_data: dict, out_path: Path, theme: str = "dark"):
    """Plot all 14 disease PR curves on a single high-resolution figure."""
    is_dark = (theme == "dark")
    bg_color = "#0f172a" if is_dark else "#ffffff"
    card_color = "#1e293b" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    subtext_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#334155" if is_dark else "#e2e8f0"
    spine_color = "#475569" if is_dark else "#cbd5e1"

    fig, ax = plt.subplots(figsize=(13, 9), facecolor=bg_color)
    ax.set_facecolor(card_color)

    # 14 distinct high-contrast colors
    palette = [
        "#38bdf8", "#34d399", "#f87171", "#fbbf24", "#a78bfa",
        "#f472b6", "#2dd4bf", "#fb923c", "#818cf8", "#4ade80",
        "#e879f9", "#38bdf8", "#facc15", "#fdba74"
    ]

    # Draw Iso-F1 curves
    f_scores = [0.2, 0.4, 0.6, 0.8]
    for f in f_scores:
        x = np.linspace(0.01, 1, 100)
        y = f * x / (2 * x - f)
        valid = (y >= 0) & (y <= 1)
        x_v, y_v = x[valid], y[valid]
        if len(x_v) > 0:
            ax.plot(x_v, y_v, color=subtext_color, linestyle="--", alpha=0.35, linewidth=1.0)
            ax.annotate(f"F1={f}", xy=(x_v[-1], y_v[-1]),
                        xytext=(x_v[-1] - 0.05, min(y_v[-1] + 0.015, 0.98)),
                        fontsize=8.5, color=subtext_color, alpha=0.7)

    # Plot Micro & Macro average PR curves
    ax.plot(
        pr_data["micro"]["recall"], pr_data["micro"]["precision"],
        color="#ffffff" if is_dark else "#000000",
        linewidth=3.0, linestyle="-",
        label=f"★ Micro-average (AUPRC = {pr_data['micro']['ap']:.4f})"
    )

    # Sort classes by AUPRC descending for the legend
    sorted_classes = sorted(CLASSES, key=lambda c: pr_data[c]["ap"], reverse=True)
    for idx, cls in enumerate(sorted_classes):
        c_data = pr_data[cls]
        color = palette[idx % len(palette)]
        ax.plot(
            c_data["recall"], c_data["precision"],
            color=color, linewidth=2.0, alpha=0.9,
            label=f"{cls:<18s} (AUPRC = {c_data['ap']:.4f}, Prev = {c_data['prevalence']*100:.1f}%)"
        )
        # Mark calibrated operating point if available
        if "op_recall" in c_data and "op_precision" in c_data:
            ax.scatter(
                c_data["op_recall"], c_data["op_precision"],
                color=color, s=55, zorder=5, edgecolors=bg_color, linewidth=1.2
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=13, fontweight="bold", color=text_color, labelpad=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=13, fontweight="bold", color=text_color, labelpad=10)

    macro_ap = np.mean([pr_data[c]["ap"] for c in CLASSES])
    ax.set_title(
        f"NIH ChestX-ray14 — Phase 1 Multi-Label Precision-Recall (PR) Curves\n"
        f"Test Set (N = 22,424) | Macro-AUPRC = {macro_ap:.4f} | Micro-AUPRC = {pr_data['micro']['ap']:.4f}",
        fontsize=14, fontweight="bold", color=text_color, pad=15
    )

    ax.grid(True, linestyle=":", alpha=0.4, color=grid_color)
    ax.tick_params(colors=subtext_color, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    leg = ax.legend(
        loc="upper right", frameon=True,
        facecolor=bg_color, edgecolor=spine_color,
        fontsize=9.2, labelcolor=text_color
    )
    for text in leg.get_texts():
        text.set_family("monospace")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=bg_color)
    plt.close()
    logger.info(f"Saved combined PR curve to {out_path}")


def plot_grid_pr_curves(pr_data: dict, out_path: Path, theme: str = "dark"):
    """Plot individual PR curve for each of the 14 pathologies in a 4x4 grid."""
    is_dark = (theme == "dark")
    bg_color = "#0f172a" if is_dark else "#ffffff"
    card_color = "#1e293b" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    subtext_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#334155" if is_dark else "#e2e8f0"
    spine_color = "#475569" if is_dark else "#cbd5e1"
    accent_color = "#38bdf8"

    fig, axes = plt.subplots(4, 4, figsize=(18, 16), facecolor=bg_color)
    axes_flat = axes.flatten()

    for idx, cls in enumerate(CLASSES):
        ax = axes_flat[idx]
        ax.set_facecolor(card_color)
        c_data = pr_data[cls]

        # PR Curve
        ax.plot(c_data["recall"], c_data["precision"], color=accent_color, linewidth=2.2, label=f"PR (AP={c_data['ap']:.3f})")

        # Baseline no-skill classifier (prevalence line)
        prev = c_data["prevalence"]
        ax.axhline(prev, color="#f87171", linestyle="--", linewidth=1.3, alpha=0.8, label=f"Prev ({prev*100:.1f}%)")

        # Calibrated Operating Point
        if "op_recall" in c_data and "op_precision" in c_data:
            ax.scatter(
                c_data["op_recall"], c_data["op_precision"],
                color="#fbbf24", s=65, zorder=5, edgecolors="#000", linewidth=1.5,
                label=f"Opt Thr ({c_data.get('threshold', 0.5):.2f})"
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f"{cls}", fontsize=12, fontweight="bold", color=text_color, pad=8)
        ax.set_xlabel("Recall", fontsize=9, color=subtext_color)
        ax.set_ylabel("Precision", fontsize=9, color=subtext_color)
        ax.grid(True, linestyle=":", alpha=0.35, color=grid_color)
        ax.tick_params(colors=subtext_color, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(spine_color)

        leg = ax.legend(loc="upper right", frameon=True, facecolor=bg_color, edgecolor=spine_color, fontsize=8, labelcolor=text_color)

    # 15th slot: Micro-average
    ax_micro = axes_flat[14]
    ax_micro.set_facecolor(card_color)
    ax_micro.plot(pr_data["micro"]["recall"], pr_data["micro"]["precision"], color="#34d399", linewidth=2.5, label=f"Micro PR (AP={pr_data['micro']['ap']:.3f})")
    ax_micro.set_xlim([0.0, 1.0])
    ax_micro.set_ylim([0.0, 1.05])
    ax_micro.set_title("★ Micro-Average", fontsize=12, fontweight="bold", color="#34d399", pad=8)
    ax_micro.set_xlabel("Recall", fontsize=9, color=subtext_color)
    ax_micro.set_ylabel("Precision", fontsize=9, color=subtext_color)
    ax_micro.grid(True, linestyle=":", alpha=0.35, color=grid_color)
    ax_micro.tick_params(colors=subtext_color, labelsize=8.5)
    for spine in ax_micro.spines.values():
        spine.set_color(spine_color)
    ax_micro.legend(loc="upper right", frameon=True, facecolor=bg_color, edgecolor=spine_color, fontsize=8.5, labelcolor=text_color)

    # 16th slot: Summary card
    ax_info = axes_flat[15]
    ax_info.set_facecolor(card_color)
    ax_info.axis("off")
    macro_ap = np.mean([pr_data[c]["ap"] for c in CLASSES])
    info_text = (
        f"NIH ChestX-ray14 Evaluation\n"
        f"------------------------------------\n"
        f"Phase:      Phase 1 Classification\n"
        f"Split:      20% Sealed Test Set\n"
        f"Samples:    N = 22,424 Images\n"
        f"Seed:       42 (Strict Protocol)\n"
        f"Classes:    14 Multi-Label Diseases\n\n"
        f"Macro-AUPRC: {macro_ap:.4f}\n"
        f"Micro-AUPRC: {pr_data['micro']['ap']:.4f}\n"
        f"------------------------------------\n"
        f"Top 3 by AUPRC:\n"
    )
    top_3 = sorted(CLASSES, key=lambda c: pr_data[c]["ap"], reverse=True)[:3]
    for r, c in enumerate(top_3, 1):
        info_text += f" {r}. {c}: {pr_data[c]['ap']:.4f}\n"

    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=10.5, family="monospace", color=text_color,
                 verticalalignment="center", linespacing=1.4)

    plt.suptitle("NIH ChestX-ray14 — Per-Pathology Precision-Recall Breakdown (Phase 1 Baseline)",
                 fontsize=16, fontweight="bold", color=text_color, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=bg_color)
    plt.close()
    logger.info(f"Saved grid PR curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PR curves for Phase 1 Classifier")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_classifier_efficientnet_b1.pth"),
                        help="Path to trained classification model checkpoint")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for test evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing predictions ignoring cache")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "light"], help="Plot theme (dark/light)")
    parser.add_argument("--out-dir", type=str, default=str(FIG_DIR), help="Output directory for plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    # 1. Load dataset with exact 70/10/20 seed 42 split
    logger.info("Initializing NIH ChestX-ray14 dataset split...")
    full_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))
    n_total = len(full_ds)
    n_test = int(n_total * 0.20)
    n_val = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(42)
    _train_idx, _val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    test_ds = Subset(full_ds, test_idx.indices)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    logger.info(f"Dataset split: {n_train} Train / {n_val} Val / {n_test} Test (using {len(test_ds)} test images)")

    # 2. Load model
    model, backbone_name = load_model(ckpt_path, device)

    # 3. Get test predictions
    cache_file = PREDS_DIR / f"phase1_{backbone_name}_test_predictions.npz"
    test_probs, test_labels = get_test_predictions(
        model, test_loader, device, cache_file, force_recompute=args.force_recompute
    )

    # 4. Load calibrated thresholds if available
    thresh_file = CHECKPOINT_DIR / f"optimal_thresholds_{backbone_name}.json"
    if not thresh_file.exists():
        thresh_file = CHECKPOINT_DIR / "optimal_thresholds.json"
    thresholds = {}
    if thresh_file.exists():
        with open(thresh_file, "r") as f:
            thresholds = json.load(f)
        logger.info(f"Loaded calibrated thresholds from {thresh_file.name}")

    # 5. Compute PR Curves and metrics
    logger.info("Computing Precision-Recall curves and Average Precision scores...")
    pr_data = {}
    metrics_records = []

    for idx, cls in enumerate(CLASSES):
        y_true = test_labels[:, idx]
        y_prob = test_probs[:, idx]

        precision, recall, threshs = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        prev = float(np.mean(y_true))

        entry = {
            "precision": precision,
            "recall": recall,
            "thresholds": threshs,
            "ap": float(ap),
            "prevalence": prev,
            "positives": int(np.sum(y_true)),
            "total": int(len(y_true))
        }

        # Calibrated threshold operating point
        opt_thresh = thresholds.get(cls, 0.5)
        preds = (y_prob >= opt_thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        op_prec = tp / max(tp + fp, 1)
        op_rec = tp / max(tp + fn, 1)
        op_f1 = f1_score(y_true, preds, zero_division=0)

        entry["threshold"] = opt_thresh
        entry["op_precision"] = float(op_prec)
        entry["op_recall"] = float(op_rec)
        entry["op_f1"] = float(op_f1)

        pr_data[cls] = entry

        metrics_records.append({
            "Class": cls,
            "AUPRC": round(ap, 4),
            "Prevalence": round(prev, 4),
            "Calibrated_Threshold": round(opt_thresh, 4),
            "Precision_at_Threshold": round(op_prec, 4),
            "Recall_at_Threshold": round(op_rec, 4),
            "F1_at_Threshold": round(op_f1, 4),
            "Positives": int(np.sum(y_true)),
            "Total": int(len(y_true))
        })

    # Micro-average PR curve
    micro_prec, micro_rec, _ = precision_recall_curve(test_labels.ravel(), test_probs.ravel())
    micro_ap = average_precision_score(test_labels.ravel(), test_probs.ravel())
    pr_data["micro"] = {
        "precision": micro_prec,
        "recall": micro_rec,
        "ap": float(micro_ap)
    }

    macro_ap = float(np.mean([m["AUPRC"] for m in metrics_records]))
    logger.info(f"Macro-AUPRC: {macro_ap:.4f} | Micro-AUPRC: {micro_ap:.4f}")

    # 6. Save metrics CSV & JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_records)
    csv_out = METRICS_DIR / f"phase1_{backbone_name}_pr_metrics.csv"
    metrics_df.to_csv(csv_out, index=False)
    logger.info(f"Saved PR metrics table to {csv_out}")

    json_out = METRICS_DIR / f"phase1_{backbone_name}_pr_metrics.json"
    with open(json_out, "w") as f:
        json.dump({
            "model": backbone_name,
            "checkpoint": str(ckpt_path),
            "macro_auprc": macro_ap,
            "micro_auprc": micro_ap,
            "per_class": metrics_records
        }, f, indent=2)
    logger.info(f"Saved PR metrics JSON to {json_out}")

    # 7. Render figures
    combined_fig = out_dir / f"pr_curve_phase1_{backbone_name}_combined.png"
    grid_fig = out_dir / f"pr_curve_phase1_{backbone_name}_grid.png"

    plot_combined_pr_curve(pr_data, combined_fig, theme=args.theme)
    plot_grid_pr_curves(pr_data, grid_fig, theme=args.theme)

    # Also render light theme if dark was default, so user has both
    if args.theme == "dark":
        combined_fig_light = out_dir / f"pr_curve_phase1_{backbone_name}_combined_light.png"
        grid_fig_light = out_dir / f"pr_curve_phase1_{backbone_name}_grid_light.png"
        plot_combined_pr_curve(pr_data, combined_fig_light, theme="light")
        plot_grid_pr_curves(pr_data, grid_fig_light, theme="light")

    print("\n" + "=" * 78)
    print(f" PHASE 1 CLASSIFIER PRECISION-RECALL (PR) METRICS REPORT ({backbone_name})")
    print("=" * 78)
    print(f" Macro-AUPRC : {macro_ap:.4f}")
    print(f" Micro-AUPRC : {micro_ap:.4f}")
    print("-" * 78)
    print(f" {'Class':<20} {'AUPRC':>7} {'Prev %':>7} {'Thr':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 78)
    for r in metrics_records:
        print(f" {r['Class']:<20} {r['AUPRC']:7.4f} {r['Prevalence']*100:6.2f}% {r['Calibrated_Threshold']:6.3f} {r['Precision_at_Threshold']:6.3f} {r['Recall_at_Threshold']:6.3f} {r['F1_at_Threshold']:6.3f}")
    print("=" * 78)
    print(f"Figures generated:")
    print(f" - {combined_fig}")
    print(f" - {grid_fig}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
