"""
src/explainability/gradcam.py
==============================
Grad-CAM and Disease Attention map utilities for CheXpertB1V2Model.

Also includes:
- Deletion / insertion curve AUC comparison
- Localization evaluator scaffold
- Overlay visualization saving
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# =============================================================================
# GradCAM
# =============================================================================

class GradCAM:
    """
    Grad-CAM for any target layer in the model.

    Args:
        model:             The CheXpertB1V2Model (or any nn.Module).
        target_layer_name: Dot-separated attribute path, e.g. "backbone.features.8"
                           or a submodule object (via target_layer_module).
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "backbone.features.8"):
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List = []

        # Resolve target layer
        layer = model
        for attr in target_layer_name.split("."):
            layer = getattr(layer, attr)
        self.target_layer = layer

        # Register hooks
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Compute Grad-CAM for a single class and single image.

        Args:
            x:          [1, 3, H, W] input tensor (requires_grad NOT needed here)
            class_idx:  which class to generate map for

        Returns:
            cam: [H_orig, W_orig] numpy array, values in [0, 1]
        """
        self.model.eval()
        x = x.clone().requires_grad_(False)

        # Forward
        self.model.zero_grad()
        logits, _ = self.model(x)

        # Backward on target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks did not fire. Check target_layer_name.")

        # Pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input size
        H, W = x.shape[2], x.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_all_classes(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM for all classes.

        Returns:
            cams: [num_classes, H, W] numpy array
        """
        self.model.eval()
        num_classes = self.model.num_classes
        cams = []
        for c in range(num_classes):
            cam = self.generate(x, c)
            cams.append(cam)
        return np.stack(cams, axis=0)

    def cleanup(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =============================================================================
# Disease Attention Maps (hook-based extraction)
# =============================================================================

class DiseaseAttentionMap:
    """
    Extracts raw disease attention maps from CheXpertB1V2Model.dag output.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_maps: Optional[torch.Tensor] = None
        self._hook = model.dag.register_forward_hook(self._save_maps)

    def _save_maps(self, module, input, output):
        # output: [B, num_classes, H, W]
        self._attention_maps = output.detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns the disease attention map for a single class.

        Args:
            x: [1, 3, H, W] input (or [B, 3, H, W] — uses first sample)
        Returns:
            attn_map: [H_att, W_att] numpy array in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            self.model(x)

        if self._attention_maps is None:
            raise RuntimeError("Hook did not fire.")

        maps = self._attention_maps[0, class_idx]  # [H_att, W_att]

        # Upsample to input resolution
        H, W = x.shape[2], x.shape[3]
        maps_up = F.interpolate(
            maps.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        m, M = maps_up.min(), maps_up.max()
        if M > m:
            maps_up = (maps_up - m) / (M - m)
        return maps_up

    def generate_all_classes(self, x: torch.Tensor) -> np.ndarray:
        """Returns [num_classes, H, W] numpy array."""
        self.model.eval()
        with torch.no_grad():
            self.model(x)

        if self._attention_maps is None:
            raise RuntimeError("Hook did not fire.")

        maps = self._attention_maps[0]  # [num_classes, H_att, W_att]
        H, W = x.shape[2], x.shape[3]
        num_classes = maps.shape[0]

        maps_up = F.interpolate(
            maps.unsqueeze(0).float(),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0).cpu().numpy()  # [num_classes, H, W]

        # Normalize each map independently
        for i in range(num_classes):
            m, M = maps_up[i].min(), maps_up[i].max()
            if M > m:
                maps_up[i] = (maps_up[i] - m) / (M - m)

        return maps_up

    def cleanup(self):
        self._hook.remove()


# =============================================================================
# Deletion / Insertion Curves
# =============================================================================

def _mask_top_k(image: torch.Tensor, attention_map: np.ndarray, k_frac: float,
                baseline: str = "mean") -> torch.Tensor:
    """
    Returns image with top-k% pixels (by attention) replaced by baseline.
    """
    H, W = image.shape[2], image.shape[3]
    n_pixels = H * W
    k = int(k_frac * n_pixels)
    if k == 0:
        return image.clone()

    flat_map = attention_map.ravel()
    if k >= len(flat_map):
        threshold = -np.inf
    else:
        threshold = np.partition(flat_map, len(flat_map) - k)[len(flat_map) - k]

    mask = torch.from_numpy((attention_map >= threshold).astype(np.float32))  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask = mask.expand_as(image)

    if baseline == "mean":
        bl_val = image.mean()
        result = image.clone()
        result[mask == 1] = bl_val.item()
    elif baseline == "zero":
        result = image.clone() * (1 - mask)
    elif baseline == "blur":
        import torchvision.transforms.functional as TF
        blurred = TF.gaussian_blur(image.squeeze(0), kernel_size=51, sigma=10).unsqueeze(0)
        result = image.clone() * (1 - mask) + blurred * mask
    else:
        result = image.clone()

    return result


def _insert_top_k(image: torch.Tensor, attention_map: np.ndarray, k_frac: float,
                  baseline: str = "blur") -> torch.Tensor:
    """Returns baseline image with top-k% pixels (by attention) revealed."""
    H, W = image.shape[2], image.shape[3]
    n_pixels = H * W
    k = int(k_frac * n_pixels)

    flat_map = attention_map.ravel()
    if k == 0:
        threshold = np.inf
    elif k >= len(flat_map):
        threshold = -np.inf
    else:
        threshold = np.partition(flat_map, len(flat_map) - k)[len(flat_map) - k]

    mask = torch.from_numpy((attention_map >= threshold).astype(np.float32))  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(image)

    if baseline == "blur":
        try:
            import torchvision.transforms.functional as TF
            bl = TF.gaussian_blur(image.squeeze(0), kernel_size=51, sigma=10).unsqueeze(0)
        except Exception:
            bl = torch.zeros_like(image) + image.mean()
    elif baseline == "mean":
        bl = torch.zeros_like(image) + image.mean()
    else:
        bl = torch.zeros_like(image)

    return bl * (1 - mask) + image * mask


@torch.no_grad()
def deletion_curve(
    model: nn.Module,
    x: torch.Tensor,
    attention_map: np.ndarray,
    class_idx: int,
    n_steps: int = 20,
    baseline: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Deletion curve: progressively mask top-k% pixels.
    Lower AUC = better (more important pixels were deleted first).

    Returns:
        fractions: [n_steps+1] from 0→1
        scores:    model sigmoid probability for class_idx at each step
        auc:       area under the deletion curve (trapezoidal)
    """
    model.eval()
    fractions = np.linspace(0, 1, n_steps + 1)
    scores = []

    for k_frac in fractions:
        x_masked = _mask_top_k(x, attention_map, k_frac, baseline=baseline)
        x_masked = x_masked.to(next(model.parameters()).device)
        logits, _ = model(x_masked)
        prob = torch.sigmoid(logits[0, class_idx]).item()
        scores.append(prob)

    scores_arr = np.array(scores)
    auc = float(np.trapz(scores_arr, fractions))
    return fractions, scores_arr, auc


@torch.no_grad()
def insertion_curve(
    model: nn.Module,
    x: torch.Tensor,
    attention_map: np.ndarray,
    class_idx: int,
    n_steps: int = 20,
    baseline: str = "blur",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Insertion curve: progressively reveal top-k% pixels from a blurred baseline.
    Higher AUC = better (important pixels are revealed early).

    Returns:
        fractions, scores, auc
    """
    model.eval()
    fractions = np.linspace(0, 1, n_steps + 1)
    scores = []

    for k_frac in fractions:
        x_insert = _insert_top_k(x, attention_map, k_frac, baseline=baseline)
        x_insert = x_insert.to(next(model.parameters()).device)
        logits, _ = model(x_insert)
        prob = torch.sigmoid(logits[0, class_idx]).item()
        scores.append(prob)

    scores_arr = np.array(scores)
    auc = float(np.trapz(scores_arr, fractions))
    return fractions, scores_arr, auc


def compare_deletion_insertion(
    model: nn.Module,
    x: torch.Tensor,
    gradcam: GradCAM,
    disease_attention: DiseaseAttentionMap,
    class_idx: int,
    n_steps: int = 20,
) -> Dict:
    """
    Runs deletion and insertion for both GradCAM and Disease Attention.
    Returns dict: {gradcam_del_auc, gradcam_ins_auc, disatt_del_auc, disatt_ins_auc}
    """
    gc_map = gradcam.generate(x, class_idx)
    da_map = disease_attention.generate(x, class_idx)

    _, _, gc_del_auc = deletion_curve(model, x, gc_map, class_idx, n_steps)
    _, _, gc_ins_auc = insertion_curve(model, x, gc_map, class_idx, n_steps)
    _, _, da_del_auc = deletion_curve(model, x, da_map, class_idx, n_steps)
    _, _, da_ins_auc = insertion_curve(model, x, da_map, class_idx, n_steps)

    return {
        "gradcam_del_auc":   gc_del_auc,
        "gradcam_ins_auc":   gc_ins_auc,
        "disatt_del_auc":    da_del_auc,
        "disatt_ins_auc":    da_ins_auc,
        "class_idx":         class_idx,
    }


# =============================================================================
# Localization Evaluator (scaffold)
# =============================================================================

class LocalizationEvaluator:
    """
    Interface for evaluating attention maps against bounding box annotations.
    Currently a scaffold — implement when annotations are available.

    Supported annotation formats (when annotation_file is provided):
    - CheXlocalize JSON ({"patient_id": {"class": [[x1,y1,x2,y2], ...]}, ...})
    - NIH CSV (image_index, finding_label, x, y, w, h)
    """

    def __init__(self, annotation_file: Optional[str] = None):
        self.annotations = None
        if annotation_file is not None:
            self._load_annotations(annotation_file)

    def _load_annotations(self, path: str):
        import json
        with open(path) as f:
            self.annotations = json.load(f)
        log.info("Loaded %d annotation entries from %s", len(self.annotations), path)

    def pointing_game(
        self,
        attention_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        class_name: str,
    ) -> int:
        """
        Returns 1 if the peak of attention_map falls inside bbox, else 0.
        bbox: (x1, y1, x2, y2) in pixel coords (matching attention_map resolution).
        """
        if self.annotations is None:
            raise NotImplementedError("Provide annotation_file to use LocalizationEvaluator")

        H, W = attention_map.shape
        peak_idx = np.unravel_index(attention_map.argmax(), attention_map.shape)
        py, px = peak_idx  # row, col

        x1, y1, x2, y2 = bbox
        inside = (x1 <= px <= x2) and (y1 <= py <= y2)
        return int(inside)

    def iou_metric(
        self,
        attention_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        threshold: float = 0.5,
    ) -> float:
        """
        Binarize attention_map at threshold, compute IoU with bbox mask.
        """
        if self.annotations is None:
            raise NotImplementedError("Provide annotation_file to use LocalizationEvaluator")

        H, W = attention_map.shape
        binary_att = (attention_map >= threshold).astype(np.uint8)

        # Create bbox mask
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        bbox_mask[y1:y2+1, x1:x2+1] = 1

        intersection = (binary_att & bbox_mask).sum()
        union        = (binary_att | bbox_mask).sum()
        iou = intersection / max(union, 1)
        return float(iou)


# =============================================================================
# Overlay Visualization
# =============================================================================

def _tensor_to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """Convert [1,3,H,W] tensor (normalized) back to [H,W,3] uint8."""
    # Denormalize ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _apply_heatmap_overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap (H×W, 0–1) on an RGB image (H×W×3)."""
    try:
        import cv2
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = (alpha * colored_rgb + (1 - alpha) * img).astype(np.uint8)
        return overlay
    except ImportError:
        # Fallback: simple red channel overlay without cv2
        overlay = img.copy().astype(np.float32)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + heatmap * 128, 0, 255)
        return overlay.astype(np.uint8)


def save_attention_overlays(
    model: nn.Module,
    dataset,
    output_dir: str,
    labels: List[str],
    top_k: int = 5,
    classes: Optional[List[str]] = None,
    device: str = "cuda",
    gradcam_layer: str = "backbone.features.8",
):
    """
    For each class: find top-k true positives and top-k false positives (by model confidence).
    Save side-by-side: original | disease attention | GradCAM.

    Files: {output_dir}/{class_name}_tp_{rank}.png and _fp_{rank}.png
    """
    try:
        from PIL import Image
    except ImportError:
        log.error("PIL not available. Cannot save overlays.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    # Initialize extractors
    dis_att = DiseaseAttentionMap(model)
    gradcam = GradCAM(model, target_layer_name=gradcam_layer)

    classes_to_vis = classes or labels

    for cls_name in classes_to_vis:
        if cls_name not in labels:
            continue
        cls_idx = labels.index(cls_name)
        safe_name = cls_name.replace(" ", "_").replace("/", "_")

        # Collect predictions over dataset (sample up to 500 items for speed)
        n_sample = min(500, len(dataset))
        entries = []  # (idx, prob, true_label)
        for idx in range(n_sample):
            img_t, lbl_t, mask_t = dataset[idx]
            true_label = int(lbl_t[cls_idx].item())
            with torch.no_grad():
                x = img_t.unsqueeze(0).to(dev)
                logits, _ = model(x)
                prob = torch.sigmoid(logits[0, cls_idx]).item()
            entries.append((idx, prob, true_label))

        # Sort by confidence for TP (true=1, sort desc) and FP (true=0, sort desc)
        tp_entries = sorted(
            [(i, p, t) for i, p, t in entries if t == 1],
            key=lambda e: e[1], reverse=True
        )[:top_k]
        fp_entries = sorted(
            [(i, p, t) for i, p, t in entries if t == 0],
            key=lambda e: e[1], reverse=True
        )[:top_k]

        for tag, group in [("tp", tp_entries), ("fp", fp_entries)]:
            for rank, (idx, prob, true_label) in enumerate(group, start=1):
                img_t, lbl_t, mask_t = dataset[idx]
                x = img_t.unsqueeze(0).to(dev)

                orig_img = _tensor_to_numpy_img(img_t.unsqueeze(0))

                # Disease attention
                da_map = dis_att.generate(x, cls_idx)
                da_overlay = _apply_heatmap_overlay(orig_img, da_map)

                # GradCAM
                try:
                    gc_map = gradcam.generate(x, cls_idx)
                    gc_overlay = _apply_heatmap_overlay(orig_img, gc_map)
                except Exception as e:
                    log.warning("GradCAM failed for %s idx=%d: %s", cls_name, idx, e)
                    gc_overlay = orig_img.copy()

                # Side-by-side: orig | disease attention | gradcam
                H, W = orig_img.shape[:2]
                composite = np.concatenate([orig_img, da_overlay, gc_overlay], axis=1)
                pil_img = Image.fromarray(composite)

                fname = out_path / f"{safe_name}_{tag}_{rank:02d}_prob{prob:.2f}.png"
                pil_img.save(str(fname))
                log.info("Saved: %s", fname)

    dis_att.cleanup()
    gradcam.cleanup()
    log.info("Overlay visualization complete. Output: %s", output_dir)

