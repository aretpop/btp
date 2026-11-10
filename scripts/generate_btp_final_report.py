"""
scripts/generate_btp_final_report.py
====================================
Computes comprehensive thesis-ready evaluation metrics (AUROC, AUPRC, F1,
Sensitivity, Specificity, Accuracy, Youden's J optimal thresholds) and
generates publication-quality ROC curves, Precision-Recall curves, and
cross-dataset comparison charts across NIH ChestX-ray14 and CheXpert+.

Outputs:
  - outputs/metrics/chexpert_transfer_metrics.csv
  - outputs/metrics/chexpert_transfer_metrics.json
  - outputs/btp_report_figures/roc_curves_chexpert.png
  - outputs/btp_report_figures/pr_curves_chexpert.png
  - outputs/btp_report_figures/nih_vs_chexpert_auroc_comparison.png
  - outputs/btp_report_figures/per_class_clinical_metrics.png
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_phase1_classifier import ChestXRayModel, NUM_CLASSES as NIH_CLASSES, CLASSES as NIH_LABELS
from transfer_chexpert import CheXpertTransferModel, CHEXPERT_LABELS, NUM_CHEXPERT
from visualize_chexpert_explainability import load_transfer_model, find_cached_path, TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
FIG_DIR      = PROJECT_ROOT / "outputs" / "btp_report_figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def get_val_predictions(model: CheXpertTransferModel, df: pd.DataFrame, cache_dir: Path, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []

    for _, row in df.iterrows():
        img_path = find_cached_path(cache_dir, row["file_id"])
        pil = Image.open(img_path).convert("RGB")
        x = TRANSFORM(pil).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        y = []
        for lbl in CHEXPERT_LABELS:
            val = row.get(lbl, 0.0)
            if pd.isna(val) or val is None or val == -1.0:
                val = 0.0
            y.append(float(val))
        all_probs.append(probs)
        all_labels.append(y)

    return np.array(all_probs), np.array(all_labels)


def compute_detailed_metrics(probs: np.ndarray, labels: np.ndarray, stage_label: str):
    rows = []
    curve_data = {}

    for i, name in enumerate(CHEXPERT_LABELS):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        n_pos = int(y_true.sum())
        n_neg = int((1 - y_true).sum())

        if n_pos == 0 or n_neg == 0:
            rows.append({
                "Stage": stage_label,
                "Pathology": name,
                "Positives": n_pos,
                "Negatives": n_neg,
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "Optimal_Threshold": np.nan,
                "F1_Score": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "Accuracy": np.nan,
            })
            continue

        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

        # Find threshold maximizing Youden's J (Sensitivity + Specificity - 1)
        j_scores = tpr - fpr
        best_j_idx = int(np.argmax(j_scores))
        opt_thresh = float(roc_thresh[best_j_idx])

        # Also check F1 across candidate thresholds for balanced clinical threshold
        f1_candidates = (2 * prec * rec) / (prec + rec + 1e-8)
        best_f1_idx = int(np.argmax(f1_candidates))
        if best_f1_idx < len(pr_thresh):
            opt_thresh_f1 = float(pr_thresh[best_f1_idx])
        else:
            opt_thresh_f1 = opt_thresh

        y_pred = (y_prob >= opt_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        acc  = float((tp + tn) / max(tp + tn + fp + fn, 1))
        f1   = float(f1_score(y_true, (y_prob >= opt_thresh_f1).astype(int), zero_division=0))

        rows.append({
            "Stage": stage_label,
            "Pathology": name,
            "Positives": n_pos,
            "Negatives": n_neg,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "Optimal_Threshold": round(opt_thresh, 3),
            "F1_Score": round(f1, 4),
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "Accuracy": round(acc, 4),
        })

        curve_data[name] = {
            "fpr": fpr, "tpr": tpr, "auroc": auroc,
            "prec": prec, "rec": rec, "auprc": auprc,
            "prevalence": n_pos / (n_pos + n_neg),
        }

    return pd.DataFrame(rows), curve_data


def plot_roc_curves(curves_a: dict, curves_b: dict):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8), facecolor="#0f172a")

    palette = [
        "#38bdf8", "#4ade80", "#f43f5e", "#fbbf24", "#a855f7",
        "#2dd4bf", "#fb923c", "#e879f9", "#818cf8", "#94a3b8"
    ]

    for ax, curves, title in [
        (axes[0], curves_a, "Stage A (Frozen NIH Backbone) — Validation ROC Curves\nMacro-AUROC = 0.8487"),
        (axes[1], curves_b, "Stage B (Unfrozen Top-3 Blocks) — Validation ROC Curves\nMacro-AUROC = 0.8436 (CheXpert-4 = 0.8619)")
    ]:
        ax.set_facecolor("#1e293b")
        sorted_names = sorted(curves.keys(), key=lambda k: curves[k]["auroc"], reverse=True)
        for idx, name in enumerate(sorted_names):
            c = curves[name]
            color = palette[idx % len(palette)]
            ax.plot(
                c["fpr"], c["tpr"], color=color, linewidth=2.2,
                label=f"{name:<18s} (AUC = {c['auroc']:.4f})"
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="Random Chance (0.5000)")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, color="#e2e8f0", fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, color="#e2e8f0", fontweight="bold")
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="white", pad=12)
        ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        leg = ax.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=8.8)
        for text in leg.get_texts():
            text.set_color("#f8fafc")
            text.set_family("monospace")

    plt.suptitle(
        "CheXpert+ Multi-Label Receiver Operating Characteristic (ROC) Curves across 10 Evaluable Pathologies",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / "roc_curves_chexpert.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved ROC curves -> {out_path}")
    return out_path


def plot_pr_curves(curves_a: dict, curves_b: dict):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8), facecolor="#0f172a")

    palette = [
        "#38bdf8", "#4ade80", "#f43f5e", "#fbbf24", "#a855f7",
        "#2dd4bf", "#fb923c", "#e879f9", "#818cf8", "#94a3b8"
    ]

    for ax, curves, title in [
        (axes[0], curves_a, "Stage A — Precision-Recall Curves"),
        (axes[1], curves_b, "Stage B — Precision-Recall Curves")
    ]:
        ax.set_facecolor("#1e293b")
        sorted_names = sorted(curves.keys(), key=lambda k: curves[k]["auprc"], reverse=True)
        mean_ap = np.mean([curves[k]["auprc"] for k in sorted_names])
        for idx, name in enumerate(sorted_names):
            c = curves[name]
            color = palette[idx % len(palette)]
            ax.plot(
                c["rec"], c["prec"], color=color, linewidth=2.2,
                label=f"{name:<18s} (AP = {c['auprc']:.4f})"
            )
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.03])
        ax.set_xlabel("Recall (Sensitivity)", fontsize=11, color="#e2e8f0", fontweight="bold")
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, color="#e2e8f0", fontweight="bold")
        ax.set_title(f"{title}\nMean AUPRC = {mean_ap:.4f}", fontsize=12.5, fontweight="bold", color="white", pad=12)
        ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        leg = ax.legend(loc="lower left", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=8.8)
        for text in leg.get_texts():
            text.set_color("#f8fafc")
            text.set_family("monospace")

    plt.suptitle(
        "CheXpert+ Multi-Label Precision-Recall (PR) Curves across 10 Evaluable Pathologies",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / "pr_curves_chexpert.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved PR curves -> {out_path}")
    return out_path


def plot_nih_vs_chexpert_progression():
    """
    Plots overall Macro-AUROC progression across all phases of the BTP project
    and overlapping disease comparisons between NIH ChestX-ray14 and CheXpert+.
    """
    fig = plt.figure(figsize=(19, 7.5), facecolor="#0f172a")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.25)

    # Panel 0: Overall Project Progression Bar Chart
    ax0 = fig.add_subplot(gs[0], facecolor="#1e293b")
    experiments = [
        "NIH Phase 1\n(GAP Baseline)",
        "NIH Phase 3 M1\n(7×7 Attention)",
        "NIH Phase 3 M2B\n(14×14 + B5 Fusion)",
        "CheXpert+ 14-Cls\n(Stratified Smoke)",
        "CheXpert+ Stage A\n(11-Cls Uniform)",
        "CheXpert+ Stage B\n(11-Cls Unfrozen)",
        "CheXpert-4 Official\n(Stage B Subset)"
    ]
    scores = [0.8424, 0.8354, 0.8336, 0.7308, 0.8487, 0.8436, 0.8619]
    colors = ["#64748b", "#38bdf8", "#0284c7", "#f43f5e", "#22c55e", "#10b981", "#f59e0b"]

    y_pos = np.arange(len(experiments))[::-1]
    bars = ax0.barh(y_pos, scores, color=colors, height=0.62)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(experiments, fontsize=9.5, color="#f8fafc", fontweight="bold")
    ax0.set_xlim(0.60, 0.93)
    ax0.set_xlabel("Validation Macro-AUROC", fontsize=11, color="#e2e8f0", fontweight="bold")
    ax0.set_title("BTP End-to-End Model Progression (NIH → CheXpert+)", fontsize=12.5, fontweight="bold", color="white", pad=12)
    ax0.grid(True, axis="x", linestyle=":", alpha=0.3, color="#94a3b8")
    ax0.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax0.spines.values():
        spine.set_color("#475569")

    for bar, s in zip(bars, scores):
        ax0.text(
            s + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{s:.4f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="white"
        )

    # Panel 1: Overlapping Pathologies (NIH M2B vs CheXpert Stage A vs CheXpert Stage B)
    ax1 = fig.add_subplot(gs[1], facecolor="#1e293b")
    shared_diseases = ["Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"]
    nih_m2b_scores  = [0.8754, 0.7499, 0.8632, 0.8480, 0.8495]
    chex_a_scores   = [0.8118, 0.8832, 0.8867, 0.8462, 0.9451]
    chex_b_scores   = [0.8429, 0.8944, 0.8858, 0.8243, 0.9465]

    x = np.arange(len(shared_diseases))
    width = 0.25

    b1 = ax1.bar(x - width, nih_m2b_scores, width, label="NIH Phase 3 M2B (Source)", color="#38bdf8")
    b2 = ax1.bar(x,         chex_a_scores,  width, label="CheXpert+ Stage A (Frozen)", color="#a855f7")
    b3 = ax1.bar(x + width, chex_b_scores,  width, label="CheXpert+ Stage B (Unfrozen)", color="#22c55e")

    ax1.set_xticks(x)
    ax1.set_xticklabels(shared_diseases, fontsize=10, color="#f8fafc", fontweight="bold")
    ax1.set_ylim(0.65, 1.00)
    ax1.set_ylabel("Validation AUROC", fontsize=11, color="#e2e8f0", fontweight="bold")
    ax1.set_title("Cross-Dataset Transfer Performance on Overlapping Pathologies", fontsize=12.5, fontweight="bold", color="white", pad=12)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax1.spines.values():
        spine.set_color("#475569")

    for bars_group in [b1, b2, b3]:
        for bar in bars_group:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, h + 0.006,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0", fontweight="bold"
            )

    leg = ax1.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=9.5)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.suptitle(
        "Cross-Dataset Generalization & Transfer Learning Benchmark (NIH ChestX-ray14 → Stanford CheXpert+)",
        fontsize=14.5, fontweight="bold", color="white", y=0.98
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.12, right=0.97)
    out_path = FIG_DIR / "nih_vs_chexpert_auroc_comparison.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved NIH vs CheXpert comparison -> {out_path}")
    return out_path


def plot_per_class_clinical_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """
    Plots Sensitivity, Specificity, F1, and AUROC per class for Stage A and Stage B.
    """
    valid_b = df_b.dropna(subset=["AUROC"]).sort_values("AUROC", ascending=True)
    valid_a = df_a.set_index("Pathology").loc[valid_b["Pathology"]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), facecolor="#0f172a")

    # Left: Stage A vs Stage B AUROC per class
    ax0 = axes[0]
    ax0.set_facecolor("#1e293b")
    y = np.arange(len(valid_b))
    h = 0.36
    ax0.barh(y - h/2, valid_a["AUROC"], height=h, color="#38bdf8", label="Stage A (Frozen Backbone)")
    ax0.barh(y + h/2, valid_b["AUROC"], height=h, color="#4ade80", label="Stage B (Unfrozen Top-3 Blocks)")
    ax0.set_yticks(y)
    ax0.set_yticklabels(valid_b["Pathology"], fontsize=10, color="#f8fafc", fontweight="bold")
    ax0.set_xlim(0.65, 1.0)
    ax0.set_xlabel("Validation AUROC", fontsize=11, color="#e2e8f0", fontweight="bold")
    ax0.set_title("Per-Class AUROC: Stage A vs. Stage B", fontsize=12.5, fontweight="bold", color="white", pad=12)
    ax0.grid(True, axis="x", linestyle=":", alpha=0.3, color="#94a3b8")
    ax0.tick_params(colors="#cbd5e1")
    for spine in ax0.spines.values():
        spine.set_color("#475569")
    leg0 = ax0.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=9.5)
    for t in leg0.get_texts():
        t.set_color("white")

    # Right: Stage B Sensitivity, Specificity, and F1 Score
    ax1 = axes[1]
    ax1.set_facecolor("#1e293b")
    w = 0.26
    ax1.barh(y - w, valid_b["Sensitivity"], height=w, color="#f43f5e", label="Sensitivity (Recall)")
    ax1.barh(y,     valid_b["Specificity"], height=w, color="#38bdf8", label="Specificity (TNR)")
    ax1.barh(y + w, valid_b["F1_Score"],    height=w, color="#fbbf24", label="Optimal F1-Score")
    ax1.set_yticks(y)
    ax1.set_yticklabels(valid_b["Pathology"], fontsize=10, color="#f8fafc", fontweight="bold")
    ax1.set_xlim(0.0, 1.05)
    ax1.set_xlabel("Score at Optimal Operating Threshold", fontsize=11, color="#e2e8f0", fontweight="bold")
    ax1.set_title("Stage B Clinical Operating Point (Sensitivity / Specificity / F1)", fontsize=12.5, fontweight="bold", color="white", pad=12)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.tick_params(colors="#cbd5e1")
    for spine in ax1.spines.values():
        spine.set_color("#475569")
    leg1 = ax1.legend(loc="lower right", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=9.5)
    for t in leg1.get_texts():
        t.set_color("white")

    plt.suptitle(
        "Comprehensive Clinical Evaluation Profile across CheXpert+ Validation Set (202 Frontal Studies)",
        fontsize=14.5, fontweight="bold", color="white", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / "per_class_clinical_metrics.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved clinical metrics chart -> {out_path}")
    return out_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Stage A and Stage B checkpoints on {device}...")

    ckpt_a = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / "stageA_efficientnet_b1_attention.pth"
    ckpt_b = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / "stageB_efficientnet_b1_attention.pth"

    model_a = load_transfer_model(ckpt_a, device)
    model_b = load_transfer_model(ckpt_b, device)

    val_csv = PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv"
    df = pd.read_csv(val_csv)
    df = df[df["local_filename"].str.contains("frontal", na=False)].reset_index(drop=True)
    cache_dir = PROJECT_ROOT / "data" / "chexpert_val_cache"

    probs_a, labels_a = get_val_predictions(model_a, df, cache_dir, device)
    probs_b, labels_b = get_val_predictions(model_b, df, cache_dir, device)

    df_metrics_a, curves_a = compute_detailed_metrics(probs_a, labels_a, "Stage_A_Frozen")
    df_metrics_b, curves_b = compute_detailed_metrics(probs_b, labels_b, "Stage_B_Unfrozen")

    df_combined = pd.concat([df_metrics_a, df_metrics_b], ignore_index=True)
    csv_path = METRICS_DIR / "chexpert_transfer_metrics.csv"
    json_path = METRICS_DIR / "chexpert_transfer_metrics.json"
    df_combined.to_csv(csv_path, index=False)

    summary = {
        "stage_a_macro_auroc": round(float(df_metrics_a["AUROC"].dropna().mean()), 4),
        "stage_a_macro_auprc": round(float(df_metrics_a["AUPRC"].dropna().mean()), 4),
        "stage_a_macro_f1":    round(float(df_metrics_a["F1_Score"].dropna().mean()), 4),
        "stage_a_macro_sens":  round(float(df_metrics_a["Sensitivity"].dropna().mean()), 4),
        "stage_a_macro_spec":  round(float(df_metrics_a["Specificity"].dropna().mean()), 4),
        "stage_b_macro_auroc": round(float(df_metrics_b["AUROC"].dropna().mean()), 4),
        "stage_b_macro_auprc": round(float(df_metrics_b["AUPRC"].dropna().mean()), 4),
        "stage_b_macro_f1":    round(float(df_metrics_b["F1_Score"].dropna().mean()), 4),
        "stage_b_macro_sens":  round(float(df_metrics_b["Sensitivity"].dropna().mean()), 4),
        "stage_b_macro_spec":  round(float(df_metrics_b["Specificity"].dropna().mean()), 4),
        "stage_a_table": df_metrics_a.to_dict(orient="records"),
        "stage_b_table": df_metrics_b.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics CSV -> {csv_path}")
    print(f"Saved metrics JSON -> {json_path}")

    plot_roc_curves(curves_a, curves_b)
    plot_pr_curves(curves_a, curves_b)
    plot_nih_vs_chexpert_progression()
    plot_per_class_clinical_metrics(df_metrics_a, df_metrics_b)


if __name__ == "__main__":
    main()
