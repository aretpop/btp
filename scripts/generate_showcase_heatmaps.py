"""
scripts/generate_showcase_heatmaps.py
=====================================
Generates publication-quality explainability heatmaps for key thoracic diseases:
Cardiomegaly, Pleural Effusion, Edema, Pneumothorax, and Atelectasis.
Produces 3-panel figures:
[Original CXR | Disease Attention (DAG) Overlay | Grad-CAM Saliency Overlay]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from src.explainability.gradcam import GradCAM, DiseaseAttentionMap
from scripts.train_chexpert_b1_v2 import (
    CheXpertLocalDatasetV2,
    make_patient_splits,
    CHEXPERT_14_LABELS,
)

ARTIFACTS_DIR = Path(r"C:\Users\savi2\.gemini\antigravity\brain\b14ade40-8de8-46de-8ce1-916d01c7aa2a")

TARGET_DISEASES = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
    "Atelectasis",
]


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized [3, H, W] tensor to [H, W, 3] float image in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.cpu().numpy() * std + mean
    img = np.clip(img, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def plot_heatmaps_for_sample(
    orig_img: np.ndarray,
    dag_map: np.ndarray,
    gradcam_map: np.ndarray,
    disease_name: str,
    prob: float,
    ground_truth: float,
    patient_id: str,
    save_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # 1. Original Chest X-Ray
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Chest X-Ray\nPatient: {patient_id}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. DAG Disease Attention Map
    axes[1].imshow(orig_img)
    im1 = axes[1].imshow(dag_map, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Learned Disease Attention (DAG)\nPredicted P({disease_name}) = {prob:.3f}", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Grad-CAM Map
    axes[2].imshow(orig_img)
    im2 = axes[2].imshow(gradcam_map, cmap="jet", alpha=0.45)
    gt_str = "Positive (1.0)" if ground_truth == 1.0 else ("Uncertain (-1.0)" if ground_truth == -1.0 else "Negative (0.0)")
    axes[2].set_title(f"Grad-CAM (Feature Layer 8)\nGround Truth: {gt_str}", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Antigravity B1v2 Model Attention Verification — {disease_name.upper()}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/heatmaps",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    model = CheXpertB1V2Model(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    # Setup extractors
    dag = DiseaseAttentionMap(model)
    gradcam = GradCAM(model, target_layer_name="backbone.features.8")

    # Load validation data
    index_csv = PROJECT_ROOT / "data/chexpert_subset/chexpert_materialized_index.csv"
    df = pd.read_csv(index_csv)
    _, val_df, _ = make_patient_splits(df, 0.1, 0.1, 42)

    val_ds = CheXpertLocalDatasetV2(
        val_df,
        split="val",
        cfg={
            "uncertain_policy": "u_zeros",
            "cache_dir": str(PROJECT_ROOT / "data/chexpert_cache"),
            "model_input_resolution": model_cfg.get("input_resolution", 224),
        },
    )

    print(f"Searching for representative validation cases for {TARGET_DISEASES}...")
    saved_images = []

    for disease in TARGET_DISEASES:
        cls_idx = CHEXPERT_14_LABELS.index(disease)
        # Find positive validation cases
        pos_indices = val_df.index[val_df[disease] == 1.0].tolist()
        if not pos_indices:
            pos_indices = val_df.index[val_df[disease] == -1.0].tolist()

        if not pos_indices:
            print(f"Warning: No positive case found for {disease}")
            continue

        # Score top cases to find highest confidence positive detection
        best_idx = None
        best_prob = -1.0
        best_x = None
        best_gt = None
        best_pid = None

        # Sample up to 15 cases to find the clearest clinical presentation
        candidates = pos_indices[:15]
        for c_idx in candidates:
            # Map index relative to val_df
            loc_idx = val_df.index.get_loc(c_idx)
            img_t, targets, masks = val_ds[loc_idx]
            x = img_t.unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(x)
                prob = torch.sigmoid(logits[0, cls_idx]).item()

            if prob > best_prob:
                best_prob = prob
                best_idx = loc_idx
                best_x = x
                best_gt = val_df.iloc[loc_idx][disease]
                raw_filename = str(val_df.iloc[loc_idx].get("local_filename", "patient_unknown"))
                best_pid = raw_filename.split("/")[0] if "/" in raw_filename else f"sample_{loc_idx}"

        if best_x is not None:
            # Generate DAG attention map
            dag_map = dag.generate(best_x, cls_idx)
            # Generate GradCAM
            gc_map = gradcam.generate(best_x, cls_idx)

            img_np = denormalize(best_x[0])
            safe_name = disease.lower().replace(" ", "_")
            out_filename = f"heatmap_{safe_name}.png"
            local_path = out_dir / out_filename
            artifact_path = ARTIFACTS_DIR / out_filename

            plot_heatmaps_for_sample(
                orig_img=img_np,
                dag_map=dag_map,
                gradcam_map=gc_map,
                disease_name=disease,
                prob=best_prob,
                ground_truth=best_gt,
                patient_id=best_pid,
                save_path=local_path,
            )

            # Copy to artifact dir for direct embedding
            shutil.copyfile(local_path, artifact_path)
            saved_images.append({
                "disease": disease,
                "prob": best_prob,
                "ground_truth": best_gt,
                "patient_id": best_pid,
                "artifact_path": str(artifact_path),
                "filename": out_filename,
            })
            print(f"✓ Saved {disease} heatmap (Prob: {best_prob:.3f}) -> {local_path}")

    dag.cleanup()
    gradcam.cleanup()
    print("\nAll heatmaps generated successfully!")


if __name__ == "__main__":
    main()

