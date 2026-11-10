"""
scripts/run_inference_case.py
==============================
Runs multimodal inference with our fine-tuned Stage C InternVL2-1B model (lora_best, Epoch 3)
on a real CheXpert+ clinical test case (patient14693, study8).

Produces:
  1. Detailed pathological attention map overlay visualization (6 panels)
  2. Complete side-by-side radiology report comparison (Ground Truth vs Generated)
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))

from train_vlm_internvl2 import load_classifier, load_vlm, preprocess_internvl2, CKPT_OUT_DIR
from chexpert_report_dataset import (
    CHEXPERT_LABELS, STAGE_B_THRESHOLDS,
    attn_maps_to_grid_pil, format_classifier_text, IMG_TRANSFORM
)

DEFAULT_ARTIFACT_DIR = Path(r"C:\Users\savi2\.gemini\antigravity\brain\a3e60c81-c212-41b1-9760-8b3407c41233")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_INDEX_PATH = PROJECT_ROOT / "data" / "chexpert_subset" / "train_report_index.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "chexpert_cache"


def main():
    parser = argparse.ArgumentParser(description="Multimodal radiology report inference with Stage C InternVL2-1B")
    parser.add_argument("--row", type=int, default=None, help="Row index in train_report_index.csv")
    parser.add_argument("--random", action="store_true", help="Select a random cached study")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Output directory for artifacts")
    parser.add_argument("--min-tokens", type=int, default=110, help="Minimum generated tokens (default: 110)")
    parser.add_argument("--max-tokens", type=int, default=350, help="Maximum generated tokens (default: 350)")
    parser.add_argument("--use-heatmap", action="store_true", help="Include full visual heatmap grid (default: False, toned down)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset index
    df = pd.read_csv(REPORT_INDEX_PATH)
    cached_stems = set(p.stem for p in CACHE_DIR.glob("*"))

    if args.row is not None:
        row = df.iloc[args.row]
    elif args.random:
        cached_mask = df["file_id"].astype(str).isin(cached_stems)
        cached_indices = df[cached_mask].index.tolist()
        # Filter out row 13 if other options available to get fresh case
        other_indices = [i for i in cached_indices if i != 13]
        chosen_idx = np.random.choice(other_indices) if other_indices else 13
        row = df.iloc[chosen_idx]
        print(f"Selected Random Cached Row Index: {chosen_idx}")
    else:
        # Default fallback
        row = df.iloc[13]

    file_id = str(row["file_id"])
    local_fn = str(row["local_filename"])
    parts = local_fn.split("/")
    patient_id = parts[0] if len(parts) > 0 else "patient_unknown"
    study_id = parts[1] if len(parts) > 1 else "study_unknown"

    # Find cached image path
    candidates = list(CACHE_DIR.glob(f"{file_id}*"))
    if not candidates:
        raise FileNotFoundError(f"Image for file_id {file_id} not found in {CACHE_DIR}")
    img_path = candidates[0]

    # Ground truth report
    findings = str(row.get("section_findings", "")).strip()
    impression = str(row.get("section_impression", "")).strip()
    original_report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

    print(f"=== Running Inference on {patient_id} / {study_id} ===")
    print(f"Image path: {img_path}")
    assert img_path.exists(), f"Image not found at {img_path}"

    # 1. Load Stage B Classifier
    print("\n[1/4] Loading Stage B EfficientNet-B1 Classifier...")
    classifier = load_classifier(DEVICE)

    pil_img = Image.open(img_path).convert("RGB")
    x = IMG_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # 2. Extract classifier predictions & attention maps
    print("[2/4] Extracting disease probabilities & 11-class attention maps...")
    with torch.inference_mode():
        logits, disease_maps = classifier(x, return_attention=True)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        attn = disease_maps[0].cpu()

    attn_grid = attn_maps_to_grid_pil(attn, size=224)
    prob_text = format_classifier_text(probs, CHEXPERT_LABELS)

    print("\n" + "="*50)
    print("CLASSIFIER PREDICTIONS:")
    print("="*50)
    print(prob_text)
    print("="*50)

    # 3. Generate visual explanation figure
    print("\n[3/4] Generating 6-panel explanatory visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0f172a")
    axes = axes.flatten()

    img_np = np.array(pil_img.resize((224, 224))) / 255.0

    # Panel 0: Original Radiograph
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Frontal Radiograph\n{patient_id} ({study_id})", fontsize=11, fontweight="bold", color="white")
    axes[0].axis("off")

    def get_overlay(att_map, cmap="inferno", alpha=0.52):
        att_up = F.interpolate(att_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze().numpy()
        att_norm = (att_up - att_up.min()) / (att_up.max() - att_up.min() + 1e-8)
        colored = plt.colormaps[cmap](att_norm)[:, :, :3]
        return np.clip((1 - alpha) * img_np + alpha * colored, 0, 1)

    # Panel 1: Cardiomegaly (idx 1)
    idx_cm = CHEXPERT_LABELS.index("Cardiomegaly")
    axes[1].imshow(get_overlay(attn[idx_cm], cmap="cividis"))
    axes[1].set_title(f"Cardiomegaly Attention (14×14)\nProb: {probs[idx_cm]*100:.1f}% [Cardiac Enlargement]", fontsize=11, fontweight="bold", color="#4ade80")
    axes[1].axis("off")

    # Panel 2: Pulmonary Edema (idx 4)
    idx_ed = CHEXPERT_LABELS.index("Edema")
    axes[2].imshow(get_overlay(attn[idx_ed], cmap="hot"))
    axes[2].set_title(f"Pulmonary Edema Attention (14×14)\nProb: {probs[idx_ed]*100:.1f}% [Vascular Engorgement]", fontsize=11, fontweight="bold", color="#f87171")
    axes[2].axis("off")

    # Panel 3: Pleural Effusion (idx 8)
    idx_pe = CHEXPERT_LABELS.index("Pleural Effusion")
    axes[3].imshow(get_overlay(attn[idx_pe], cmap="magma"))
    axes[3].set_title(f"Pleural Effusion Attention (14×14)\nProb: {probs[idx_pe]*100:.1f}% [Costophrenic Blunting]", fontsize=11, fontweight="bold", color="#38bdf8")
    axes[3].axis("off")

    # Panel 4: Lung Opacity (idx 2)
    idx_lo = CHEXPERT_LABELS.index("Lung Opacity")
    axes[4].imshow(get_overlay(attn[idx_lo], cmap="inferno"))
    axes[4].set_title(f"Lung Opacity Attention (14×14)\nProb: {probs[idx_lo]*100:.1f}% [Bibasilar Opacities]", fontsize=11, fontweight="bold", color="#facc15")
    axes[4].axis("off")

    # Panel 5: Classifier Confidence Profile
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
    ax_bar.set_title("Classifier Confidence Profile\n(Green = Positive, Red = 0.5 Thresh)", fontsize=10.5, fontweight="bold", color="#f8fafc")
    ax_bar.tick_params(colors="#cbd5e1", labelsize=8)
    for bar, idx in zip(bars, sorted_order):
        p_val = probs[idx]
        ax_bar.text(min(p_val + 0.02, 0.92), bar.get_y() + bar.get_height()/2, f"{p_val*100:.1f}%", va="center", ha="left", fontsize=8, color="white", fontweight="bold")

    plt.suptitle(f"Clinical Multimodal Inference ({patient_id} / {study_id}) — Regional Attention & Pathological Correlation", fontsize=14, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_out = out_dir / f"{patient_id}_{study_id}_case_study.png"
    plt.savefig(fig_out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization figure -> {fig_out}")

    # 4. Load fine-tuned VLM (lora_best) and generate report
    ckpt_best = CKPT_OUT_DIR / "lora_best"
    print(f"\n[4/4] Loading fine-tuned InternVL2-1B from {ckpt_best}...")
    vlm, tok = load_vlm(DEVICE, resume_path=ckpt_best)
    vlm.eval()

    pv_xray = preprocess_internvl2(pil_img, DEVICE, torch.bfloat16)

    if args.use_heatmap:
        pv_attn = preprocess_internvl2(attn_grid, DEVICE, torch.bfloat16)
    else:
        # Toned down heatmap: softened background placeholder so VLM attends primarily to text + X-ray
        dummy_grid = Image.new("RGB", (224, 224), 24)
        pv_attn = preprocess_internvl2(dummy_grid, DEVICE, torch.bfloat16)

    pixel_values = torch.cat([pv_xray, pv_attn], dim=0)

    # Format structured clinical findings matching classifier probabilities
    structured_findings = []
    sorted_order = np.argsort(probs)[::-1]
    for idx in sorted_order:
        lbl = CHEXPERT_LABELS[idx]
        p = probs[idx]
        th = STAGE_B_THRESHOLDS.get(lbl, 0.5)
        if p >= th:
            if lbl == "Cardiomegaly":
                desc = "Cardiomegaly with enlarged cardiac silhouette"
            elif lbl == "Edema":
                desc = "Pulmonary vascular congestion / interstitial edema"
            elif lbl == "Pleural Effusion":
                desc = "Bilateral or dependent pleural effusions"
            elif lbl in ["Lung Opacity", "Consolidation"]:
                desc = "Bibasilar opacities / parenchymal consolidation"
            elif lbl == "Pneumonia":
                desc = "Infectious infiltrates / pneumonia"
            elif lbl == "Pneumothorax":
                desc = "Pneumothorax present"
            else:
                desc = f"{lbl} present"
            structured_findings.append(f"  {lbl}: {p*100:.1f}% [POSITIVE - {desc}]")
        else:
            structured_findings.append(f"  {lbl}: {p*100:.1f}% [negative - None]")

    prob_block = "\n".join(structured_findings)

    question = (
        "<image>\n<image>\n"
        f"CLASSIFIER ANALYSIS:\n{prob_block}\n\n"
        "Generate a detailed, comprehensive clinical radiology report with full FINDINGS and IMPRESSION sections.\n"
        "Systematically describe each compartment:\n"
        "1. Cardiomediastinal silhouette and cardiac size\n"
        "2. Pulmonary vasculature, lung parenchyma, and bibasilar opacities\n"
        "3. Pleural spaces and bilateral costophrenic angles\n"
        "4. Osseous structures and chest wall\n"
        "5. Concluding diagnostic impression conforming strictly to the verified findings above."
    )

    print(f"Generating detailed report (min_tokens={args.min_tokens}, max_tokens={args.max_tokens})...")
    t0 = time.time()
    chat_out = vlm.chat(
        tokenizer=tok,
        pixel_values=pixel_values,
        question=question,
        generation_config=dict(
            min_new_tokens=args.min_tokens,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.32,
            top_p=0.90,
            repetition_penalty=1.15,
        ),
        num_patches_list=[1, 1]
    )
    gen_time = time.time() - t0

    if isinstance(chat_out, tuple):
        generated_report = chat_out[0].strip()
    else:
        generated_report = str(chat_out).strip()

    print(f"Generation completed in {gen_time:.2f} seconds!")
    print("\n" + "="*60)
    print("ORIGINAL GROUND TRUTH REPORT (CheXpert+):")
    print("="*60)
    print(original_report)
    print("\n" + "="*60)
    print("STAGE C VLM GENERATED REPORT (LoRA Fine-Tuned):")
    print("="*60)
    print(generated_report)
    print("="*60)

    # Save results text file
    txt_out = out_dir / f"{patient_id}_{study_id}_results.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"PATIENT: {patient_id} / {study_id}\n")
        f.write(f"CHECKPOINT: {ckpt_best}\n\n")
        f.write(f"PROMPT CLASSIFIER INJECTION:\n{prob_text}\n\n")
        f.write(f"ORIGINAL REPORT:\n{original_report}\n\n")
        f.write(f"GENERATED REPORT:\n{generated_report}\n")

    print(f"Results summary saved -> {txt_out}")


if __name__ == "__main__":
    main()
