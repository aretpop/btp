"""
scripts/generate_test_case_showcase.py
=======================================
Generates a side-by-side clinical case study for patient64661:
- Original chest X-ray
- Attention maps for the positive findings (Cardiomegaly, Edema, Support Devices, Lung Opacity, Pleural Effusion)
- Classifier confidence profile
- Original ground truth report from CheXpert+
- VLM generated report from InternVL2-1B
"""

import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))

from train_vlm_internvl2 import load_classifier, load_vlm, build_training_input, preprocess_internvl2
from chexpert_report_dataset import (
    CHEXPERT_LABELS, STAGE_B_THRESHOLDS,
    attn_maps_to_grid_pil, format_classifier_text, IMG_TRANSFORM
)

ARTIFACT_DIR = Path(r"C:\Users\savi2\.gemini\antigravity\brain\b14ade40-8de8-46de-8ce1-916d01c7aa2a")
DEVICE = torch.device("cuda")

IMG_PATH = PROJECT_ROOT / "data" / "chexpert_val_cache" / "41v9-09bj184m4.IIRTsUE6qin6o-FLkheOFw.png"

ORIGINAL_REPORT = """FINDINGS:
The transesophageal echo probe has been removed. A new enteric tube is present. There is otherwise unchanged positioning of supportive medical devices.
Mild pulmonary edema and cardiomegaly.
Left basilar opacity. No pneumothorax.
No acute bony abnormalities are noted.

IMPRESSION:
1. Residual mild pulmonary edema and left basilar opacity."""

def main():
    print("Loading Stage B classifier...")
    classifier = load_classifier(DEVICE)

    pil_img = Image.open(IMG_PATH).convert("RGB")
    x = IMG_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits, disease_maps = classifier(x, return_attention=True)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        attn = disease_maps[0].cpu() # [11, 14, 14]

    prob_text = format_classifier_text(probs, CHEXPERT_LABELS)
    attn_grid = attn_maps_to_grid_pil(attn, size=224)

    print("Generating attention visualizations...")
    # Generate high-res multi-panel figure
    # 6 panels: Original X-ray, Support Devices (95%), Lung Opacity (91%), Edema (79%), Cardiomegaly (73%), Confidence Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0f172a")
    axes = axes.flatten()

    img_np = np.array(pil_img.resize((224, 224))) / 255.0

    # 1. Original X-ray
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Frontal Radiograph\nPatient: patient64661 (Study 1)", fontsize=11, fontweight="bold", color="white")
    axes[0].axis("off")

    # Colormap overlays
    def get_overlay(att_map, cmap="inferno", alpha=0.5):
        att_up = F.interpolate(att_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze().numpy()
        att_norm = (att_up - att_up.min()) / (att_up.max() - att_up.min() + 1e-8)
        colored = plt.colormaps[cmap](att_norm)[:, :, :3]
        return np.clip((1 - alpha) * img_np + alpha * colored, 0, 1)

    # 2. Support Devices (idx 10)
    idx_sd = CHEXPERT_LABELS.index("Support Devices")
    axes[1].imshow(get_overlay(attn[idx_sd], cmap="magma"))
    axes[1].set_title(f"Support Devices Attention (14×14)\nProb: {probs[idx_sd]*100:.1f}% [Enteric tube / lines]", fontsize=11, fontweight="bold", color="#38bdf8")
    axes[1].axis("off")

    # 3. Lung Opacity (idx 2)
    idx_lo = CHEXPERT_LABELS.index("Lung Opacity")
    axes[2].imshow(get_overlay(attn[idx_lo], cmap="inferno"))
    axes[2].set_title(f"Lung Opacity Attention (14×14)\nProb: {probs[idx_lo]*100:.1f}% [Left basilar opacity]", fontsize=11, fontweight="bold", color="#facc15")
    axes[2].axis("off")

    # 4. Pulmonary Edema (idx 4)
    idx_ed = CHEXPERT_LABELS.index("Edema")
    axes[3].imshow(get_overlay(attn[idx_ed], cmap="hot"))
    axes[3].set_title(f"Pulmonary Edema Attention (14×14)\nProb: {probs[idx_ed]*100:.1f}% [Perihilar haziness]", fontsize=11, fontweight="bold", color="#f87171")
    axes[3].axis("off")

    # 5. Cardiomegaly (idx 1)
    idx_cm = CHEXPERT_LABELS.index("Cardiomegaly")
    axes[4].imshow(get_overlay(attn[idx_cm], cmap="cividis"))
    axes[4].set_title(f"Cardiomegaly Attention (14×14)\nProb: {probs[idx_cm]*100:.1f}% [Cardiac silhouette]", fontsize=11, fontweight="bold", color="#4ade80")
    axes[4].axis("off")

    # 6. Confidence Profile
    ax_bar = axes[5]
    ax_bar.set_facecolor("#1e293b")
    sorted_order = np.argsort(probs)
    y_pos = np.arange(len(CHEXPERT_LABELS))
    colors = []
    for idx in sorted_order:
        lbl = CHEXPERT_LABELS[idx]
        p = probs[idx]
        thresh = STAGE_B_THRESHOLDS.get(lbl, 0.5)
        colors.append("#22c55e" if p >= thresh else "#64748b")

    bars = ax_bar.barh(y_pos, probs[sorted_order], color=colors, height=0.65)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([CHEXPERT_LABELS[i] for i in sorted_order], fontsize=9, color="#e2e8f0", fontweight="bold")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.axvline(0.5, color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_bar.set_title("Classifier Confidence Profile\n(Green = Positive, Red line = 0.5)", fontsize=10.5, fontweight="bold", color="#f8fafc")
    ax_bar.tick_params(colors="#cbd5e1", labelsize=8)
    for bar, idx in zip(bars, sorted_order):
        p_val = probs[idx]
        ax_bar.text(min(p_val + 0.02, 0.92), bar.get_y() + bar.get_height()/2, f"{p_val*100:.1f}%", va="center", ha="left", fontsize=8, color="white", fontweight="bold")

    plt.suptitle("Clinical Case Study (patient64661) — Learned Regional Attention & Pathological Alignment", fontsize=14, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_img_path = ARTIFACT_DIR / "patient64661_case_study.png"
    plt.savefig(out_img_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved case study visualization -> {out_img_path}")

    # Now run InternVL2-1B inference
    print("\nLoading InternVL2-1B for report generation...")
    vlm, tok = load_vlm(DEVICE)
    vlm.eval()

    pv_xray = preprocess_internvl2(pil_img, DEVICE, torch.bfloat16)
    pv_attn = preprocess_internvl2(attn_grid, DEVICE, torch.bfloat16)
    pixel_values = torch.cat([pv_xray, pv_attn], dim=0)

    question = (
        "<image>\n<image>\n"
        f"{prob_text}\n\n"
        "Based on the chest X-ray and the disease attention maps above, "
        "generate a complete radiology report with FINDINGS and IMPRESSION sections."
    )

    chat_out = vlm.chat(
        tokenizer=tok,
        pixel_values=pixel_values,
        question=question,
        generation_config=dict(max_new_tokens=256, do_sample=False),
        num_patches_list=[1, 1]
    )
    if isinstance(chat_out, tuple):
        response = chat_out[0]
    else:
        response = str(chat_out)

    print("\n" + "="*60)
    print("ORIGINAL GROUND TRUTH REPORT (CheXpert+):")
    print("="*60)
    print(ORIGINAL_REPORT)
    print("\n" + "="*60)
    print("VLM GENERATED REPORT (Zero-Shot / Base InternVL2 + Classifier Injection):")
    print("="*60)
    print(response)
    print("="*60)

    # Save metadata summary
    with open(ARTIFACT_DIR / "patient64661_results.txt", "w", encoding="utf-8") as f:
        f.write(f"PATIENT: patient64661 / study1\n\n")
        f.write(f"PROMPT CLASSIFIER INJECTION:\n{prob_text}\n\n")
        f.write(f"ORIGINAL REPORT:\n{ORIGINAL_REPORT}\n\n")
        f.write(f"GENERATED REPORT:\n{response}\n")

if __name__ == "__main__":
    main()
