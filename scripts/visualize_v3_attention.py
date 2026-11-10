"""
scripts/visualize_v3_attention.py
===================================
Render per-disease GradCAM-style and DAG attention maps for V3 model.

Key behaviour per prompt:
  - Only renders maps for labels predicted POSITIVE (prob > threshold)
  - Never interprets a map for a negative-predicted label
  - Saves side-by-side: original | DAG attention | predicted labels

Usage:
    python scripts/visualize_v3_attention.py \
        --config configs/chexpert_b1_v3.yaml \
        --checkpoint checkpoints/chexpert_b1_v3/xxx_stageB_best.pth \
        --image_path data/chexpert_cache/patient00001_study1_view1.jpg \
        --out_dir outputs/attn_v3 \
        --threshold 0.4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.chexpert_b1_v3 import CheXpertV3Model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path: str, res: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Returns (preprocessed tensor [1,3,H,W], original uint8 RGB array [H,W,3])."""
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
def get_attention_maps(
    model: CheXpertV3Model,
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass, return:
      probs    [14]  sigmoid probabilities
      attn_maps [14, H, W]  DAG spatial attention maps (upsized to input res)
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)

    # Partial forward to extract attention maps before pooling
    feats = model.backbone(x)
    f_mid  = feats["mid"]
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

    d_maps = model.dag(attn_input)   # [1, 14, H_mid, W_mid]

    # Full forward for logits
    logits, _ = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()   # [14]

    # Upsample maps to input resolution for visualization
    res = x.shape[-1]
    maps_up = F.interpolate(d_maps, size=(res, res), mode="bilinear", align_corners=False)
    maps_np = maps_up[0].cpu().numpy()   # [14, res, res]

    return probs, maps_np


def overlay_heatmap(orig: np.ndarray, attn: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay attention map on original image.
    orig: [H,W,3] uint8; attn: [H,W] float in [0,1].
    Returns: [H,W,3] uint8.
    """
    import matplotlib.cm as cm
    # Normalize to [0,1]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heatmap = cm.jet(attn)[:, :, :3]   # [H,W,3] float
    overlay = alpha * orig / 255.0 + (1 - alpha) * heatmap
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def visualize_positive_predictions(
    orig: np.ndarray,
    probs: np.ndarray,
    attn_maps: np.ndarray,
    threshold: float = 0.4,
    max_classes: int = 6,
    out_path: Optional[str] = None,
):
    """
    Render attention maps ONLY for predicted-positive labels (prob > threshold).
    Saves a grid image: [original] [map1] [map2] ... for each positive class.
    """
    import matplotlib.pyplot as plt

    positive_idx = [i for i in range(len(CHEXPERT_14_LABELS)) if probs[i] >= threshold]
    positive_idx = positive_idx[:max_classes]   # cap for readability

    if not positive_idx:
        log.info("No predicted-positive labels above threshold %.2f — skipping visualization", threshold)
        return

    n_cols = 1 + len(positive_idx)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Original image
    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # One panel per positive disease
    for col, idx in enumerate(positive_idx, start=1):
        lbl = CHEXPERT_14_LABELS[idx]
        prob = probs[idx]
        attn = attn_maps[idx]   # [H,W]
        overlay = overlay_heatmap(orig, attn, alpha=0.45)
        axes[col].imshow(overlay)
        axes[col].set_title(f"{lbl}\np={prob:.2f}", fontsize=8)
        axes[col].axis("off")

    plt.suptitle(f"Predicted positive (thr={threshold}): {len(positive_idx)} classes", fontsize=11)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", out_path)
    else:
        plt.show()
    plt.close(fig)

    # Log the predicted classes and their probabilities
    log.info("Positive predictions (thr=%.2f):", threshold)
    for idx in positive_idx:
        log.info("  %-35s  p=%.4f", CHEXPERT_14_LABELS[idx], probs[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--image_path",  required=True, help="Path to .jpg/.png X-ray")
    parser.add_argument("--out_dir",     default="outputs/attn_v3")
    parser.add_argument("--threshold",   type=float, default=0.4,
                        help="Min probability to show map (only positive predictions)")
    parser.add_argument("--thresholds_json", default=None,
                        help="Per-class thresholds JSON from evaluate_chexpert_b1_v3.py")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    res = cfg.get("input_resolution", 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = {k: v for k, v in cfg.items()
                 if k not in ("data", "sampler", "loss", "training", "eval", "logging", "experiment")}
    model = CheXpertV3Model.load_checkpoint_compat(args.checkpoint, model_cfg).to(device)

    # Load per-class thresholds if provided
    per_class_thr = None
    if args.thresholds_json:
        import json
        with open(args.thresholds_json) as f:
            data = json.load(f)
        per_class_thr = data.get("thresholds", {})

    x, orig = load_image(args.image_path, res=res)
    probs, attn_maps = get_attention_maps(model, x, device)

    # Use per-class thresholds if available, otherwise single threshold
    if per_class_thr:
        thresholds_arr = np.array([per_class_thr.get(l, args.threshold) for l in CHEXPERT_14_LABELS])
        positive_idx_all = [i for i in range(14) if probs[i] >= thresholds_arr[i]]
    else:
        positive_idx_all = [i for i in range(14) if probs[i] >= args.threshold]

    log.info("Image: %s", args.image_path)
    log.info("Positive predictions (%d):", len(positive_idx_all))
    for i in positive_idx_all:
        log.info("  %-35s  p=%.4f", CHEXPERT_14_LABELS[i], probs[i])

    stem = Path(args.image_path).stem
    out_path = str(Path(args.out_dir) / f"{stem}_attn_v3.png")

    visualize_positive_predictions(
        orig, probs, attn_maps,
        threshold=args.threshold,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

