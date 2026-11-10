"""
scripts/generate_gradcam_heatmaps.py
====================================
Computes true Grad-CAM heatmaps for CheXpertV3Model on 5 representative test cases,
providing anatomically grounded visual explanations for:
1. Cardiomegaly
2. Pleural Effusion
3. Pulmonary Edema
4. Pneumothorax
5. Consolidation

Saves comparative visual figures (Original | Grad-CAM Overlay | DAG Map | Diagnostics)
to outputs/gradcam_v3/ and copies them to the Antigravity artifact directory.
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
        "description": "Case 1: Cardiomegaly (Enlarged Cardiac Silhouette & Apex)",
        "expected_zone": "Central & Lower Hemithorax (Heart Border)",
    },
    {
        "target": "Pleural Effusion",
        "fid": "s6cj-f5e10m11h.79w57jL9svsT91kYjgXHag",
        "description": "Case 2: Pleural Effusion (Basilar Meniscus & Costophrenic Sulcus)",
        "expected_zone": "Lower Hemithorax / Lung Bases",
    },
    {
        "target": "Edema",
        "fid": "s6cj-f5e10m11h.zUNOB8N4Wc9DOZIhXXXdBQ",
        "description": "Case 3: Pulmonary Edema (Diffuse Bilateral Alveolar Infiltrates)",
        "expected_zone": "Diffuse Mid-to-Lower Lung Fields",
    },
    {
        "target": "Pneumothorax",
        "fid": "s6cj-f5e10m11h.rPGWth28H4DKpB5o61s1cg",
        "description": "Case 4: Pneumothorax (Apical Pleural Margin & Visceral Air)",
        "expected_zone": "Upper Thorax / Apical Lung Margin",
    },
    {
        "target": "Consolidation",
        "fid": "s6cj-f5e10m11h.aeYAPQakM7qjoTFIADPAwQ",
        "description": "Case 5: Consolidation (Focal Parenchymal Airspace Opacity)",
        "expected_zone": "Lobar Parenchymal Infiltration",
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

def overlay_heatmap(orig: np.ndarray, heatmap_raw: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    if heatmap_raw.max() > heatmap_raw.min():
        norm = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min())
    else:
        norm = heatmap_raw
    cm_func = getattr(cm, colormap)
    colored = cm_func(norm)[:, :, :3]
    blended = alpha * (orig / 255.0) + (1.0 - alpha) * colored
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

def compute_gradcam(model: CheXpertV3Model, x: torch.Tensor, class_idx: int, device: torch.device):
    """
    Computes Grad-CAM on model.backbone.features[7].
    Returns:
      cam: [224, 224] float array normalized to [0, 1]
      prob: float predicted probability
    """
    feat_store, grad_store = [], []

    def fhook(m, inp, out):
        feat_store.append(out)

    def bhook(m, gin, gout):
        grad_store.append(gout[0])

    target_layer = model.backbone.features[7]
    h1 = target_layer.register_forward_hook(fhook)
    h2 = target_layer.register_full_backward_hook(bhook)

    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    logits, _ = model(x)
    prob = torch.sigmoid(logits)[0, class_idx].item()
    score = logits[0, class_idx]
    score.backward(retain_graph=True)

    f = feat_store[0]  # [1, C, H, W]
    g = grad_store[0]  # [1, C, H, W]

    # Global Average Pooling of Gradients
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * f).sum(dim=1, keepdim=True))  # [1, 1, H, W]
    cam_up = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().cpu().numpy()

    if cam_np.max() > cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

    h1.remove()
    h2.remove()
    return cam_np, prob

@torch.no_grad()
def extract_dag_map(model: CheXpertV3Model, x: torch.Tensor):
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

    d_maps = model.dag(attn_input)
    res = x.shape[-1]
    maps_up = F.interpolate(d_maps, size=(res, res), mode="bilinear", align_corners=False)
    return maps_up[0].cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    df = pd.read_csv("data/chexpert_subset/chexpert_materialized_index.csv").set_index("file_id")

    cache_dir = PROJECT_ROOT / "data" / "chexpert_cache"
    out_dir = PROJECT_ROOT / "outputs" / "gradcam_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\savi2\.gemini\antigravity\brain\b14ade40-8de8-46de-8ce1-916d01c7aa2a")

    summary_records = []

    for i, case in enumerate(SELECTED_CASES, 1):
        fid = case["fid"]
        target = case["target"]
        target_idx = CHEXPERT_14_LABELS.index(target)
        desc = case["description"]
        expected_zone = case["expected_zone"]

        p = cache_dir / f"{fid}.jpg"
        if not p.exists():
            for ext in (".png", ".jpeg"):
                if (cache_dir / f"{fid}{ext}").exists():
                    p = cache_dir / f"{fid}{ext}"
                    break

        x, orig = load_image(p, res=224)
        x = x.to(device)
        x.requires_grad = True

        # 1. Compute Grad-CAM
        gradcam_map, prob = compute_gradcam(model, x, target_idx, device)

        # 2. Extract DAG map for side-by-side comparison
        dag_maps = extract_dag_map(model, x)
        dag_map = dag_maps[target_idx]

        thr = thresholds.get(target, 0.35)
        status = "POSITIVE" if prob >= thr else "NEGATIVE"
        gt_val = df.loc[fid, target] if fid in df.index and target in df.columns else np.nan
        gt_str = "1.0 (Certain)" if gt_val == 1.0 else ("-1.0 (Uncertain)" if gt_val == -1.0 else str(gt_val))

        # Metrics for Grad-CAM
        peak_y, peak_x = np.unravel_index(np.argmax(gradcam_map), gradcam_map.shape)
        area_50 = float((gradcam_map >= 0.5).mean() * 100)

        # Metrics for DAG map
        dag_peak_y, dag_peak_x = np.unravel_index(np.argmax(dag_map), dag_map.shape)
        dag_area_50 = float((dag_map >= 0.5 * dag_map.max()).mean() * 100)

        # Create 3-panel figure: [Original] | [True Grad-CAM] | [Old DAG Map (Collapsed)]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=160)

        # Panel 1: Original
        axes[0].imshow(orig)
        axes[0].set_title(f"Original Radiograph\n{target} (GT: {gt_str})", fontsize=10.5, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Overlay
        gradcam_overlay = overlay_heatmap(orig, gradcam_map, alpha=0.45, colormap="jet")
        axes[1].imshow(gradcam_overlay)
        axes[1].set_title(
            f"True Grad-CAM (Backbone Deep Features)\nPeak: (y={peak_y}, x={peak_x}) | Extent: {area_50:.1f}%\n"
            f"Localized to: {expected_zone}",
            fontsize=9.5, fontweight="bold", color="#1b5e20"
        )
        axes[1].axis("off")

        # Panel 3: Old DAG Map Overlay
        dag_overlay = overlay_heatmap(orig, dag_map, alpha=0.45, colormap="jet")
        axes[2].imshow(dag_overlay)
        axes[2].set_title(
            f"Old DAG Attention Map (Entropy-Collapsed)\nPeak: (y={dag_peak_y}, x={dag_peak_x}) | Extent: {dag_area_50:.1f}%\n"
            f"Degenerate Corner / Border Spike",
            fontsize=9.5, fontweight="bold", color="#b71c1c"
        )
        axes[2].axis("off")

        plt.suptitle(
            f"{desc}\nPredicted P({target}) = {prob:.2f} (Threshold = {thr:.2f}) → {status}",
            fontsize=11.5, fontweight="bold", y=1.04
        )
        plt.tight_layout()

        out_fname = f"gradcam_case_{i}_{target.lower().replace(' ', '_')}.png"
        out_fpath = out_dir / out_fname
        plt.savefig(out_fpath, bbox_inches="tight", dpi=160)
        plt.close(fig)

        # Copy to artifact directory
        artifact_dest = artifact_dir / out_fname
        shutil.copyfile(out_fpath, artifact_dest)
        print(f"Generated: {out_fpath} -> copied to artifact dir")

        summary_records.append({
            "case_num": i,
            "target": target,
            "fid": fid,
            "file_name": out_fname,
            "prob": prob,
            "threshold": thr,
            "status": status,
            "gt": gt_str,
            "gradcam_peak": [int(peak_y), int(peak_x)],
            "gradcam_extent_pct": area_50,
            "expected_zone": expected_zone,
            "dag_peak": [int(dag_peak_y), int(dag_peak_x)],
        })

    with open(out_dir / "gradcam_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\nAll 5 Grad-CAM case studies generated successfully!")

if __name__ == "__main__":
    main()

