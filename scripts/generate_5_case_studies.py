"""
scripts/generate_5_case_studies.py
====================================
Runs inference on 5 representative test cases using CheXpertV3Model (uslfa_v3),
extracts spatial Disease Attention Generator (DAG) heatmaps, generates
side-by-side clinical explainability figures, and saves them to outputs/heatmaps_v3/
and the Antigravity artifact directory.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v3 import CheXpertV3Model

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

SELECTED_CASES = [
    {
        "target": "Cardiomegaly",
        "fid": "s6cj-f5e10m11h.aJ3NLsrqO7YY3-K_cBAvjw",
        "description": "Cardiomegaly Case (Enlarged Cardiac Silhouette & Mediastinal Contour)",
    },
    {
        "target": "Pleural Effusion",
        "fid": "s6cj-f5e10m11h.79w57jL9svsT91kYjgXHag",
        "description": "Pleural Effusion Case (Costophrenic Sulcus Blunting & Basilar Fluid)",
    },
    {
        "target": "Edema",
        "fid": "s6cj-f5e10m11h.zUNOB8N4Wc9DOZIhXXXdBQ",
        "description": "Pulmonary Edema Case (Bilateral Alveolar & Interstitial Infiltrates)",
    },
    {
        "target": "Pneumothorax",
        "fid": "s6cj-f5e10m11h.rPGWth28H4DKpB5o61s1cg",
        "description": "Pneumothorax Case (Visceral Pleural Line & Apical Air Space)",
    },
    {
        "target": "Consolidation",
        "fid": "s6cj-f5e10m11h.aeYAPQakM7qjoTFIADPAwQ",
        "description": "Consolidation Case (Dense Lobar Parenchymal Opacification)",
    },
]

def load_image(path: Path, res: int = 224):
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    orig = np.array(img.resize((res, res)))
    tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0), orig

@torch.no_grad()
def predict_and_extract_attn(model: CheXpertV3Model, x: torch.Tensor, device: torch.device):
    model.eval()
    x = x.to(device)

    # Partial forward to extract attention maps
    feats = model.backbone(x)
    f_mid = feats["mid"]
    f_deep = feats["deep"]

    deep = model.msfb(f_deep)
    deep = model.channel_attn(deep)
    deep = model.spatial_attn(deep)
    deep_up = F.interpolate(deep, size=(model.H_mid, model.W_mid), mode="bilinear", align_corners=False)

    if model.fuse_block5 and model.lateral_conv is not None:
        lateral = model.lateral_conv(f_mid)
        attn_input = model.fuse_bn_relu(deep_up + lateral)
    else:
        attn_input = f_mid

    d_maps = model.dag(attn_input)  # [1, 14, H_mid, W_mid]
    logits, _ = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    res = x.shape[-1]
    maps_up = F.interpolate(d_maps, size=(res, res), mode="bilinear", align_corners=False)
    maps_np = maps_up[0].cpu().numpy()  # [14, res, res]

    return probs, maps_np

def overlay_heatmap(orig: np.ndarray, attn: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heatmap = cm.jet(attn_norm)[:, :, :3]
    overlay = alpha * (orig / 255.0) + (1 - alpha) * heatmap
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and checkpoint
    config_path = PROJECT_ROOT / "configs" / "chexpert_b1_v3.yaml"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "chexpert_b1_v3" / "chexpert_b1_v3_uslfa_v3_stageB_best.pth"
    val_json_path = PROJECT_ROOT / "outputs" / "chexpert_b1_v3" / "eval_val_chexpert_b1_v3_uslfa_v3_stageB_best.json"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(val_json_path) as f:
        val_eval = json.load(f)
    thresholds = val_eval.get("thresholds", {})

    model_cfg = {k: v for k, v in cfg.items() if k not in ("data", "sampler", "loss", "training", "eval", "logging", "experiment")}
    model = CheXpertV3Model.load_checkpoint_compat(str(ckpt_path), model_cfg).to(device)
    model.eval()

    # Load dataset index for ground truth
    df = pd.read_csv("data/chexpert_subset/chexpert_materialized_index.csv")
    df = df.set_index("file_id")

    cache_dir = PROJECT_ROOT / "data" / "chexpert_cache"
    out_dir = PROJECT_ROOT / "outputs" / "heatmaps_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(r"C:\Users\savi2\.gemini\antigravity\brain\b14ade40-8de8-46de-8ce1-916d01c7aa2a")

    summary_records = []

    for i, case in enumerate(SELECTED_CASES, 1):
        fid = case["fid"]
        target = case["target"]
        desc = case["description"]

        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = cache_dir / f"{fid}{ext}"
            if p.exists():
                img_path = p
                break

        if not img_path:
            print(f"Warning: Image {fid} not found in cache!")
            continue

        x, orig = load_image(img_path, res=224)
        probs, maps_np = predict_and_extract_attn(model, x, device)

        # Retrieve ground truth
        row_data = df.loc[fid] if fid in df.index else None

        # Determine which classes to display:
        # Include target class + any class where prob >= threshold
        shown_indices = []
        target_idx = CHEXPERT_14_LABELS.index(target)
        shown_indices.append(target_idx)

        for c_idx, lbl in enumerate(CHEXPERT_14_LABELS):
            if c_idx == target_idx:
                continue
            thr = thresholds.get(lbl, 0.35)
            if probs[c_idx] >= thr and lbl != "No Finding":
                shown_indices.append(c_idx)

        # Sort shown indices by probability descending, keeping target first
        other_indices = sorted([idx for idx in shown_indices if idx != target_idx], key=lambda idx: probs[idx], reverse=True)
        shown_indices = [target_idx] + other_indices[:3]  # Max 4 attention panels for clean display

        n_panels = 1 + len(shown_indices)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4), dpi=150)
        if n_panels == 1:
            axes = [axes]

        # 1. Original Radiograph
        axes[0].imshow(orig)
        axes[0].set_title(f"Original Chest X-Ray\nSample #{i}: {target}", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        # 2. Attention Overlays
        for col_idx, c_idx in enumerate(shown_indices, 1):
            lbl = CHEXPERT_14_LABELS[c_idx]
            p_val = probs[c_idx]
            thr = thresholds.get(lbl, 0.35)
            gt_val = row_data[lbl] if row_data is not None and lbl in row_data else np.nan

            if pd.isna(gt_val):
                gt_str = "Unmentioned"
            elif gt_val == 1.0:
                gt_str = "Pos (1.0)"
            elif gt_val == -1.0:
                gt_str = "Uncertain (-1)"
            else:
                gt_str = "Neg (0.0)"

            status = "POSITIVE" if p_val >= thr else "NEGATIVE"
            status_color = "#d9534f" if status == "POSITIVE" else "#5bc0de"

            attn_map = maps_np[c_idx]
            overlay = overlay_heatmap(orig, attn_map, alpha=0.45)

            axes[col_idx].imshow(overlay)
            axes[col_idx].set_title(
                f"{lbl}\n$P={p_val:.2f}$ (thr={thr:.2f}) | {status}\nGT: {gt_str}",
                fontsize=9.5,
                fontweight="bold" if c_idx == target_idx else "normal",
                pad=8,
            )
            axes[col_idx].axis("off")

        plt.suptitle(desc, fontsize=12, fontweight="bold", y=1.03)
        plt.tight_layout()

        out_fname = f"case_study_{i}_{target.lower().replace(' ', '_')}.png"
        out_fpath = out_dir / out_fname
        plt.savefig(out_fpath, bbox_inches="tight", dpi=160)
        plt.close(fig)

        # Copy to artifact dir for direct rendering
        artifact_dest = artifact_dir / out_fname
        shutil.copyfile(out_fpath, artifact_dest)
        print(f"Saved: {out_fpath} and copied to {artifact_dest}")

        summary_records.append({
            "case_num": i,
            "target": target,
            "fid": fid,
            "file_name": out_fname,
            "prob": float(probs[target_idx]),
            "threshold": float(thresholds.get(target, 0.35)),
            "gt": str(row_data[target]) if row_data is not None else "N/A",
            "co_detected": [CHEXPERT_14_LABELS[idx] for idx in shown_indices if idx != target_idx],
        })

    with open(out_dir / "case_studies_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\nAll 5 case studies generated successfully!")

if __name__ == "__main__":
    main()

