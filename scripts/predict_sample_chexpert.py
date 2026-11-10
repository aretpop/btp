"""
scripts/predict_sample_chexpert.py
==================================
Runs inference on specific CheXpert+ validation studies using the final
fine-tuned Stage B model (and Stage A), printing a clinical diagnostic table
and generating a multi-panel explainability figure.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transfer_chexpert import CHEXPERT_LABELS
from visualize_chexpert_explainability import (
    load_transfer_model, find_cached_path, compute_gradcam_and_attention,
    denormalize, overlay_heatmap, TRANSFORM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "sample_inferences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optimal Youden J thresholds calibrated from validation evaluation
THRESHOLDS_STAGE_B = {
    "No Finding":       0.885,
    "Cardiomegaly":     0.511,
    "Lung Opacity":     0.586,
    "Lung Lesion":      0.487,
    "Edema":            0.537,
    "Consolidation":    0.476,
    "Pneumonia":        0.500,
    "Pneumothorax":     0.478,
    "Pleural Effusion": 0.579,
    "Pleural Other":    0.351,
    "Support Devices":  0.715,
}


def run_sample_inference(sample_idx: int, device: torch.device):
    ckpt_b = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / "stageB_efficientnet_b1_attention.pth"
    model_b = load_transfer_model(ckpt_b, device)

    val_csv = PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv"
    df = pd.read_csv(val_csv)
    df = df[df["local_filename"].str.contains("frontal", na=False)].reset_index(drop=True)
    cache_dir = PROJECT_ROOT / "data" / "chexpert_val_cache"

    row = df.iloc[sample_idx]
    file_id = row["file_id"]
    local_fn = row["local_filename"]
    img_path = find_cached_path(cache_dir, file_id)

    pil = Image.open(img_path).convert("RGB")
    x = TRANSFORM(pil).unsqueeze(0).to(device)
    img_np = denormalize(x)

    with torch.no_grad():
        logits, disease_maps = model_b(x, return_attention=True)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    gt_labels = {}
    for lbl in CHEXPERT_LABELS:
        v = row.get(lbl, 0.0)
        gt_labels[lbl] = 1 if (v == 1.0) else 0

    print("=" * 75)
    print(f"CLINICAL INFERENCE REPORT: Study #{sample_idx}")
    print(f"Image File : {local_fn}")
    print(f"Patient ID : {local_fn.split('/')[0]}")
    print("=" * 75)
    print(f"{'Pathology':<24s} | {'Probability':<11s} | {'Threshold':<9s} | {'Prediction':<11s} | {'Ground Truth'}")
    print("-" * 75)

    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        name = CHEXPERT_LABELS[idx]
        p = probs[idx]
        th = THRESHOLDS_STAGE_B.get(name, 0.500)
        call = "POSITIVE" if p >= th else "NEGATIVE"
        gt = "POSITIVE" if gt_labels[name] == 1 else "NEGATIVE"
        match = " [CORRECT]" if (call == gt) else " [MISMATCH]"
        print(f"{name:<24s} | {p*100:6.2f}%    | {th*100:5.1f}%   | {call:<11s} | {gt:<11s}{match}")
    print("=" * 75)

    # Generate figure showing top 2 predicted pathologies + No Finding
    top_two = [sorted_indices[0], sorted_indices[1]]
    fig = plt.figure(figsize=(19, 6.2), facecolor="#0f172a")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.22)

    # Col 0: Original Radiograph
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_np, cmap="gray")
    gt_list = [l for l, v in gt_labels.items() if v == 1]
    ax0.set_title(f"Chest X-Ray\nPatient: {local_fn.split('/')[0]}\nGT: {', '.join(gt_list) if gt_list else 'No Finding'}",
                  fontsize=10.5, fontweight="bold", color="white")
    ax0.axis("off")

    # Col 1: Top pathology attention
    p1_idx = top_two[0]
    p1_name = CHEXPERT_LABELS[p1_idx]
    _, att1_up, cam1_up, _ = compute_gradcam_and_attention(model_b, x, p1_idx)
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(overlay_heatmap(img_np, att1_up, alpha=0.50, colormap="inferno"))
    ax1.set_title(f"Top Finding: {p1_name}\n14×14 Regional Attention (P = {probs[p1_idx]:.3f})",
                  fontsize=10.5, fontweight="bold", color="#38bdf8")
    ax1.axis("off")

    # Col 2: Top pathology Grad-CAM
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(overlay_heatmap(img_np, cam1_up, alpha=0.50, colormap="jet"))
    ax2.set_title(f"{p1_name} Feature Grad-CAM\nActivation Agreement",
                  fontsize=10.5, fontweight="bold", color="#fbbf24")
    ax2.axis("off")

    # Col 3: Probability Profile
    ax3 = fig.add_subplot(gs[3], facecolor="#1e293b")
    top_6 = sorted_indices[:6][::-1]
    y_pos = np.arange(len(top_6))
    colors = []
    for i in top_6:
        n = CHEXPERT_LABELS[i]
        if gt_labels[n] == 1:
            colors.append("#22c55e" if probs[i] >= THRESHOLDS_STAGE_B.get(n, 0.5) else "#38bdf8")
        else:
            colors.append("#ef4444" if probs[i] >= THRESHOLDS_STAGE_B.get(n, 0.5) else "#64748b")

    bars = ax3.barh(y_pos, probs[top_6], color=colors, height=0.62)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([CHEXPERT_LABELS[i] for i in top_6], fontsize=9.5, color="#f8fafc", fontweight="bold")
    ax3.set_xlim(0, 1.05)
    ax3.axvline(0.5, color="#f59e0b", linestyle="--", linewidth=1.1, alpha=0.8)
    ax3.set_xlabel("Predicted Probability", fontsize=9.5, color="#cbd5e1", fontweight="bold")
    ax3.set_title("Top-6 Probabilities\n(Green = True Pos, Slate = True Neg)", fontsize=10.5, fontweight="bold", color="white")
    ax3.tick_params(colors="#cbd5e1", labelsize=9)
    for s in ax3.spines.values():
        s.set_color("#475569")
    for bar, p_val in zip(bars, probs[top_6]):
        ax3.text(min(p_val + 0.02, 0.94), bar.get_y() + bar.get_height() / 2,
                 f"{p_val:.2f}", va="center", ha="left", fontsize=8.5, color="white", fontweight="bold")

    out_name = f"sample_{sample_idx}_{local_fn.split('/')[0]}.png"
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization -> {out_path}\n")
    return out_path, probs, gt_labels, local_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Validation index (0-201)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_sample_inference(args.index, device)


if __name__ == "__main__":
    main()
