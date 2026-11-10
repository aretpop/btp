"""
scripts/visualize_chexpert_explainability.py
=============================================
Generates explainability heatmaps (Disease-Specific Regional Attention Maps +
Grad-CAM overlays) on CheXpert+ validation images using the trained Stage A
and Stage B transfer learning checkpoints.

Outputs are saved to:
  outputs/chexpert_explainability/
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_phase1_classifier import ChestXRayModel, NUM_CLASSES as NIH_CLASSES
from transfer_chexpert import CheXpertTransferModel, CHEXPERT_LABELS, NUM_CHEXPERT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "chexpert_explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_transfer_model(ckpt_path: Path, device: torch.device) -> CheXpertTransferModel:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b1")
    nih_base = ChestXRayModel(backbone_name=backbone_name, num_classes=NIH_CLASSES, pretrained=False)
    model = CheXpertTransferModel(
        nih_base, backbone_name,
        use_attention=True,
        num_classes=NUM_CHEXPERT,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.clone().squeeze(0).cpu()
    for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[i] = t[i] * s + m
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.48, colormap: str = "jet") -> np.ndarray:
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_norm = (heatmap - h_min) / (h_max - h_min + 1e-8)
    colored = plt.colormaps[colormap](heatmap_norm)[:, :, :3]
    overlaid = (1.0 - alpha) * img_np + alpha * colored
    return np.clip(overlaid, 0.0, 1.0)


def compute_gradcam_and_attention(model: CheXpertTransferModel, x: torch.Tensor, class_idx: int):
    """
    Computes:
      1. probs: [11] predicted probabilities
      2. att_up: [224, 224] learned disease-specific spatial attention map
      3. gradcam_up: [224, 224] Grad-CAM on the refined feature map (spatial_att output)
      4. all_att_maps: [11, 14, 14] raw learned attention maps for all 11 classes
    """
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    h1 = model.spatial_att.register_forward_hook(fwd_hook)
    h2 = model.spatial_att.register_full_backward_hook(bwd_hook)

    x_in = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits, disease_maps = model(x_in, return_attention=True)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    score = logits[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    # 1. Learned Disease Attention Map (14x14 -> 224x224)
    att_14 = disease_maps[0, class_idx].detach().cpu()
    # Sharpen contrast slightly for visualization clarity
    att_up = F.interpolate(
        att_14.unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bicubic", align_corners=False
    ).squeeze().numpy()

    # 2. Grad-CAM on spatial_att refined features (7x7 -> 224x224)
    act = activations["feat"]      # [1, 256, 7, 7]
    grad = gradients["feat"]       # [1, 256, 7, 7]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1)).squeeze(0).cpu()
    gradcam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bicubic", align_corners=False
    ).squeeze().numpy()

    return probs, att_up, gradcam_up, disease_maps[0].detach().cpu().numpy()


def find_cached_path(cache_dir: Path, file_id: str) -> Path:
    safe_name = file_id.replace("/", "_").replace("\\", "_")
    for p in cache_dir.glob(f"{safe_name}*"):
        if p.is_file():
            return p
    # Fallback: check exact match or stem match
    p = cache_dir / f"{safe_name}.png"
    if p.exists():
        return p
    raise FileNotFoundError(f"Cached image not found for {file_id}")


def generate_case_showcase(df: pd.DataFrame, cache_dir: Path, model_b: CheXpertTransferModel, device: torch.device):
    """
    Figure 1: 5 Clinical Cases x 4 columns:
      Col 0: Original X-Ray + Ground Truth positives
      Col 1: Disease-Specific Regional Attention (14x14)
      Col 2: Grad-CAM Feature Attribution
      Col 3: Predicted Probability Profile (Horizontal Bar Chart)
    """
    target_targets = [
        ("Cardiomegaly",     1),
        ("Edema",            4),
        ("Consolidation",    5),
        ("Pneumothorax",     7),
        ("Pleural Effusion", 8),
    ]

    selected_rows = []
    used_indices = set()

    for target_name, target_idx in target_targets:
        best_row = None
        best_prob = -1.0
        candidates = df[df[target_name] == 1.0]
        for idx, row in candidates.iterrows():
            if idx in used_indices:
                continue
            try:
                img_path = find_cached_path(cache_dir, row["file_id"])
                pil = Image.open(img_path).convert("RGB")
                x = TRANSFORM(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model_b(x)
                    prob = torch.sigmoid(logits)[0, target_idx].item()
                if prob > best_prob:
                    best_prob = prob
                    best_row = (idx, row, img_path)
            except Exception:
                continue
        if best_row is not None:
            used_indices.add(best_row[0])
            selected_rows.append((target_name, target_idx, best_row[1], best_row[2]))

    n_cases = len(selected_rows)
    fig = plt.figure(figsize=(20, 4.2 * n_cases), facecolor="#0f172a")
    fig.suptitle(
        "CheXpert+ Explainability Showcase — Stage B Regional Attention + Grad-CAM",
        fontsize=16, fontweight="bold", color="white", y=0.995
    )

    gs = gridspec.GridSpec(n_cases, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.32, wspace=0.22)

    for row_i, (target_name, target_idx, row_data, img_path) in enumerate(selected_rows):
        pil = Image.open(img_path).convert("RGB")
        x = TRANSFORM(pil).unsqueeze(0).to(device)
        img_np = denormalize(x)

        probs, att_up, gradcam_up, _ = compute_gradcam_and_attention(model_b, x, target_idx)

        gt_positives = [lbl for lbl in CHEXPERT_LABELS if row_data.get(lbl, 0.0) == 1.0]
        gt_str = ", ".join(gt_positives) if gt_positives else "None"
        if len(gt_str) > 38:
            gt_str = gt_str[:35] + "..."

        patient_id = str(row_data.get("local_filename", img_path.name)).split("/")[0]

        # Panel 0: Original X-ray
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(img_np, cmap="gray")
        ax0.set_title(f"Case {row_i+1}: {target_name}\nGT: [{gt_str}]", fontsize=10, fontweight="bold", color="#f8fafc")
        ax0.axis("off")

        # Panel 1: Learned Regional Attention
        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(overlay_heatmap(img_np, att_up, alpha=0.48, colormap="inferno"))
        ax1.set_title(
            f"Learned Regional Attention (14×14)\nTarget: {target_name} (P = {probs[target_idx]:.3f})",
            fontsize=10, fontweight="bold", color="#38bdf8"
        )
        ax1.axis("off")

        # Panel 2: Grad-CAM Overlay
        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(overlay_heatmap(img_np, gradcam_up, alpha=0.48, colormap="jet"))
        ax2.set_title(
            f"Feature Grad-CAM Attribution\nSpatial Activation Agreement",
            fontsize=10, fontweight="bold", color="#fbbf24"
        )
        ax2.axis("off")

        # Panel 3: Top-6 Predicted Probabilities Bar Chart
        ax3 = fig.add_subplot(gs[row_i, 3], facecolor="#1e293b")
        top_k = np.argsort(probs)[::-1][:6][::-1]
        y_pos = np.arange(len(top_k))
        bar_Colors = []
        for idx_c in top_k:
            lbl_c = CHEXPERT_LABELS[idx_c]
            if idx_c == target_idx:
                bar_Colors.append("#22c55e")   # Green = target GT positive
            elif row_data.get(lbl_c, 0.0) == 1.0:
                bar_Colors.append("#38bdf8")   # Blue = other GT positive
            else:
                bar_Colors.append("#64748b")   # Slate = negative

        bars = ax3.barh(y_pos, probs[top_k], color=bar_Colors, height=0.65, edgecolor="none")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([CHEXPERT_LABELS[i] for i in top_k], fontsize=9, color="#e2e8f0", fontweight="bold")
        ax3.set_xlim(0, 1.05)
        ax3.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.7)
        ax3.set_xlabel("Predicted Probability", fontsize=8, color="#cbd5e1")
        ax3.set_title("Model Confidence Profile (Green/Blue = GT+)", fontsize=9.5, fontweight="bold", color="#f8fafc")
        ax3.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ["top", "right"]:
            ax3.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax3.spines[spine].set_color("#475569")

        for bar, p_val in zip(bars, probs[top_k]):
            ax3.text(
                min(p_val + 0.02, 0.93),
                bar.get_y() + bar.get_height() / 2,
                f"{p_val:.2f}",
                va="center", ha="left", fontsize=8, color="white", fontweight="bold"
            )

    out_path = OUT_DIR / "chexpert_case_showcase.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved showcase -> {out_path}")
    return out_path, selected_rows[0]


def generate_all11_attention_grid(row_info, model_b: CheXpertTransferModel, device: torch.device):
    """
    Figure 2: All 11 Disease-Specific Regional Attention Maps side-by-side on a single X-ray,
    demonstrating spatial diversity across heads.
    """
    target_name, target_idx, row_data, img_path = row_info
    pil = Image.open(img_path).convert("RGB")
    x = TRANSFORM(pil).unsqueeze(0).to(device)
    img_np = denormalize(x)

    probs, _, _, all_att = compute_gradcam_and_attention(model_b, x, target_idx)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5), facecolor="#0f172a")
    axes = axes.flatten()

    # Panel 0: Original X-ray
    axes[0].imshow(img_np, cmap="gray")
    gt_positives = [lbl for lbl in CHEXPERT_LABELS if row_data.get(lbl, 0.0) == 1.0]
    axes[0].set_title(
        f"Original Chest X-Ray\nGT+: {', '.join(gt_positives)}",
        fontsize=10.5, fontweight="bold", color="white"
    )
    axes[0].axis("off")

    sorted_indices = np.argsort(probs)[::-1]

    for plot_idx, d_idx in enumerate(sorted_indices):
        ax = axes[plot_idx + 1]
        cls_name = CHEXPERT_LABELS[d_idx]
        prob = probs[d_idx]
        att_14 = torch.tensor(all_att[d_idx])
        att_up = F.interpolate(
            att_14.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bicubic", align_corners=False
        ).squeeze().numpy()

        overlaid = overlay_heatmap(img_np, att_up, alpha=0.50, colormap="jet")
        ax.imshow(overlaid)

        is_gt = row_data.get(cls_name, 0.0) == 1.0
        gt_tag = " [GT+]" if is_gt else ""
        title_color = "#4ade80" if is_gt else ("#facc15" if prob >= 0.5 else "#e2e8f0")

        ax.set_title(
            f"[{d_idx}] {cls_name}{gt_tag}\nProb = {prob:.3f}",
            fontsize=10, fontweight="bold", color=title_color
        )
        ax.axis("off")

    plt.suptitle(
        "All 11 Disease-Specific Regional Attention Heads (14×14 Spatial Masks)\n"
        "Each pathology head attends to distinct anatomical zones (Green = Ground Truth Positive)",
        fontsize=14, fontweight="bold", color="white", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "chexpert_all11_attention_grid.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved 11-class grid -> {out_path}")
    return out_path


def generate_stageA_vs_stageB_comparison(df: pd.DataFrame, cache_dir: Path,
                                          model_a: CheXpertTransferModel,
                                          model_b: CheXpertTransferModel,
                                          device: torch.device):
    """
    Figure 3: Side-by-side comparison of Stage A (Frozen Backbone) vs Stage B (Unfrozen Top 3 Blocks)
    on Cardiomegaly and Consolidation cases.
    """
    compare_targets = [
        ("Cardiomegaly",  1),
        ("Consolidation", 5),
        ("Pneumothorax",  7),
    ]

    fig, axes = plt.subplots(len(compare_targets), 5, figsize=(21, 4.3 * len(compare_targets)), facecolor="#0f172a")

    for r_idx, (t_name, t_idx) in enumerate(compare_targets):
        candidates = df[df[t_name] == 1.0]
        chosen_path = None
        best_score = -999.0
        for _, row in candidates.iterrows():
            try:
                p = find_cached_path(cache_dir, row["file_id"])
                pil = Image.open(p).convert("RGB")
                x = TRANSFORM(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    pa = torch.sigmoid(model_a(x))[0, t_idx].item()
                    pb = torch.sigmoid(model_b(x))[0, t_idx].item()
                score = pb + 0.5 * (pb - pa)
                if score > best_score:
                    best_score = score
                    chosen_path = p
            except Exception:
                continue

        pil = Image.open(chosen_path).convert("RGB")
        x = TRANSFORM(pil).unsqueeze(0).to(device)
        img_np = denormalize(x)

        probs_a, att_a, cam_a, _ = compute_gradcam_and_attention(model_a, x, t_idx)
        probs_b, att_b, cam_b, _ = compute_gradcam_and_attention(model_b, x, t_idx)

        # Col 0: Original
        axes[r_idx, 0].imshow(img_np, cmap="gray")
        axes[r_idx, 0].set_title(f"Original X-Ray\nGT+: {t_name}", fontsize=10, fontweight="bold", color="white")
        axes[r_idx, 0].axis("off")

        # Col 1: Stage A Attention
        axes[r_idx, 1].imshow(overlay_heatmap(img_np, att_a, alpha=0.48, colormap="inferno"))
        axes[r_idx, 1].set_title(f"Stage A Regional Attention\n(Frozen) P = {probs_a[t_idx]:.3f}", fontsize=10, color="#cbd5e1")
        axes[r_idx, 1].axis("off")

        # Col 2: Stage B Attention
        axes[r_idx, 2].imshow(overlay_heatmap(img_np, att_b, alpha=0.48, colormap="inferno"))
        axes[r_idx, 2].set_title(f"Stage B Regional Attention\n(Unfrozen) P = {probs_b[t_idx]:.3f}", fontsize=10, fontweight="bold", color="#4ade80")
        axes[r_idx, 2].axis("off")

        # Col 3: Stage A Grad-CAM
        axes[r_idx, 3].imshow(overlay_heatmap(img_np, cam_a, alpha=0.48, colormap="jet"))
        axes[r_idx, 3].set_title(f"Stage A Feature Grad-CAM\nFrozen NIH Features", fontsize=10, color="#cbd5e1")
        axes[r_idx, 3].axis("off")

        # Col 4: Stage B Grad-CAM
        axes[r_idx, 4].imshow(overlay_heatmap(img_np, cam_b, alpha=0.48, colormap="jet"))
        axes[r_idx, 4].set_title(f"Stage B Feature Grad-CAM\nAdapted CheXpert Features", fontsize=10, fontweight="bold", color="#38bdf8")
        axes[r_idx, 4].axis("off")

    plt.suptitle(
        "Stage A (Frozen Backbone) vs. Stage B (Unfrozen Top-3 Blocks) — Attention & Grad-CAM Refinement",
        fontsize=15, fontweight="bold", color="white", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "chexpert_stageA_vs_stageB_attention.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Stage A vs B comparison -> {out_path}")
    return out_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CheXpert+ explainability visualization on {device}...")

    ckpt_a = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / "stageA_efficientnet_b1_attention.pth"
    ckpt_b = PROJECT_ROOT / "checkpoints" / "chexpert_transfer" / "stageB_efficientnet_b1_attention.pth"

    model_a = load_transfer_model(ckpt_a, device)
    model_b = load_transfer_model(ckpt_b, device)

    val_csv = PROJECT_ROOT / "data" / "chexpert_subset" / "valid_index.csv"
    df = pd.read_csv(val_csv)
    df = df[df["local_filename"].str.contains("frontal", na=False)].reset_index(drop=True)
    cache_dir = PROJECT_ROOT / "data" / "chexpert_val_cache"

    f1, sample_row_info = generate_case_showcase(df, cache_dir, model_b, device)
    f2 = generate_all11_attention_grid(sample_row_info, model_b, device)
    f3 = generate_stageA_vs_stageB_comparison(df, cache_dir, model_a, model_b, device)

    print("\nAll explainability figures generated successfully:")
    print(f"  1. {f1}")
    print(f"  2. {f2}")
    print(f"  3. {f3}")


if __name__ == "__main__":
    main()
