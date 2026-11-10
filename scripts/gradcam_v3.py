"""
scripts/gradcam_v3.py
======================
GradCAM + attention map comparison utility for Phase 3 models.

Generates side-by-side visualizations:
  - Original X-ray
  - Phase 1 GradCAM (baseline)
  - Phase 3 disease-specific attention map (14×14, upsampled for display)
  - Phase 3 GradCAM (on the attention-weighted features)

Usage:
  python scripts/gradcam_v3.py --image path/to/xray.png --checkpoint checkpoints/phase3_regional/M2B_b1.pth
  python scripts/gradcam_v3.py --image path/to/xray.png --checkpoint checkpoints/phase3_regional/M2B_b1.pth --disease 4
"""

import sys, os, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from train_phase1_classifier import ChestXRayModel, CLASSES, NUM_CLASSES
from train_phase3_regional_v3 import (
    RegionalAttentionXRayModel, BLOCK5_CHANNELS, BLOCK8_CHANNELS
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "gradcam_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(img_path: str):
    """Load and preprocess an image for model input."""
    pil = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(pil).unsqueeze(0)   # [1, 3, 224, 224]
    return pil, tensor


def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    t = tensor.clone().squeeze(0)
    for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[i] = t[i] * s + m
    return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def overlay_heatmap(img_np, heatmap, alpha=0.45, colormap="jet"):
    """Overlay a normalized heatmap on an RGB image."""
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored = cm.get_cmap(colormap)(heatmap_norm)[:, :, :3]  # RGB
    overlaid = (1 - alpha) * img_np + alpha * colored
    return np.clip(overlaid, 0, 1)


# ── GradCAM ────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks the target layer's output and gradients w.r.t. a specific class logit.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Run forward+backward and return GradCAM heatmap [H, W] in [0,1].
        """
        self.model.eval()
        x = x.requires_grad_(True)

        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)

        # Upsample to 224×224
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        return cam_up


def load_phase1_model(backbone: str, ckpt_path: str, device):
    """Load Phase 1 EfficientNet classifier."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ChestXRayModel(backbone_name=backbone, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def load_phase3_model(ckpt_path: str, device):
    """Load Phase 3 regional attention model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    backbone_name = cfg["backbone"]

    # Load the base Phase 1 model
    base_ckpt_path = Path(ckpt_path).parent.parent.parent / "checkpoints" / \
                     cfg.get("base_checkpoint", f"best_classifier_{backbone_name}.pth")
    base_ckpt = torch.load(str(base_ckpt_path), map_location=device, weights_only=False)
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    base_model.load_state_dict(base_ckpt["model_state"])

    model = RegionalAttentionXRayModel(base_model, backbone_name, cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, cfg


def get_disease_attention_map(model: RegionalAttentionXRayModel,
                               x: torch.Tensor, disease_idx: int) -> np.ndarray:
    """Extract disease-specific attention map for visualization."""
    model.eval()
    with torch.no_grad():
        _, disease_maps = model(x, return_attention=True)
    # disease_maps: [1, 14, 14, 14]
    att = disease_maps[0, disease_idx]  # [14, 14]

    # Upsample to 224×224 for display
    att_up = F.interpolate(
        att.unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()
    return att_up


def compare_all_diseases(img_path: str, p3_ckpt_path: str, device,
                         save_prefix: str = "comparison"):
    """
    Generate a grid showing disease-specific attention for all 14 conditions.
    """
    pil, x = load_image(img_path)
    x = x.to(device)
    img_np = denormalize(x.cpu())

    model_p3, cfg = load_phase3_model(p3_ckpt_path, device)

    with torch.no_grad():
        logits, disease_maps = model_p3(x, return_attention=True)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    # First panel: original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original X-Ray", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for d, disease in enumerate(CLASSES):
        ax = axes[d + 1]
        att = disease_maps[0, d].cpu()
        att_up = F.interpolate(
            att.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        overlaid = overlay_heatmap(img_np, att_up, alpha=0.5)
        ax.imshow(overlaid)
        ax.set_title(f"{disease}\np={probs[d]:.2f}", fontsize=8)
        ax.axis("off")

    # Hide last panel (14 diseases + 1 original = 15, fits 3×5 grid)
    axes[-1].axis("off")

    plt.suptitle(f"Disease-Specific Attention Maps — {Path(img_path).name}", fontsize=12)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{save_prefix}_all_diseases.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def compare_phase1_vs_phase3(img_path: str, p1_ckpt_path: str, p3_ckpt_path: str,
                              disease_idx: int, device, save_prefix: str = "comparison"):
    """
    Generate a 4-panel comparison:
      [Original] [Phase1 GradCAM] [Phase3 Attention] [Phase3 GradCAM]
    """
    pil, x = load_image(img_path)
    x = x.to(device)
    img_np = denormalize(x.cpu())

    model_p3, cfg = load_phase3_model(p3_ckpt_path, device)
    backbone_name = cfg["backbone"]

    model_p1 = load_phase1_model(
        backbone_name, p1_ckpt_path
        or str(Path(p3_ckpt_path).parent.parent.parent / "checkpoints" /
               f"best_classifier_{backbone_name}.pth"),
        device
    )

    disease_name = CLASSES[disease_idx]
    print(f"Generating comparison for disease: [{disease_idx}] {disease_name}")

    # ── Phase 1 GradCAM ──────────────────────────────────────────────────────
    # Target: last conv layer before classifier (Block 7 features)
    if hasattr(model_p1.backbone, "features"):
        target_layer_p1 = model_p1.backbone.features[7]
    else:
        target_layer_p1 = list(model_p1.backbone.children())[-3]

    cam_p1 = GradCAM(model_p1, target_layer_p1)
    gradcam_p1 = cam_p1(x.clone(), disease_idx)
    cam_p1.remove_hooks()

    # ── Phase 3 attention map ─────────────────────────────────────────────────
    att_p3 = get_disease_attention_map(model_p3, x, disease_idx)

    # ── Phase 3 GradCAM (on the disease attention generator output) ───────────
    # Hook the spatial attention output (anatomy_att output)
    target_layer_p3 = model_p3.anatomy_att.learned_spatial[-1]  # last sigmoid layer
    cam_p3 = GradCAM(model_p3, target_layer_p3)
    # Need to call the full model to get disease-specific logit
    x_grad = x.clone().requires_grad_(True)
    output_p3 = model_p3(x_grad)
    model_p3.zero_grad()
    output_p3[0, disease_idx].backward()
    cam_p3.remove_hooks()

    # Get gradients at target layer
    if cam_p3.gradients is not None:
        weights = cam_p3.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam_p3 = (weights * cam_p3.activations).sum(dim=1).squeeze(0)
        grad_cam_p3 = F.relu(grad_cam_p3)
        gradcam_p3 = F.interpolate(
            grad_cam_p3.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()
    else:
        gradcam_p3 = att_p3   # fallback

    # ── Phase 1 prediction ────────────────────────────────────────────────────
    with torch.no_grad():
        logits_p1 = model_p1(x)
        prob_p1 = torch.sigmoid(logits_p1)[0, disease_idx].item()

    with torch.no_grad():
        logits_p3 = model_p3(x)
        prob_p3 = torch.sigmoid(logits_p3)[0, disease_idx].item()

    # ── 4-panel figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_np)
    axes[0].set_title(f"Original X-Ray\n{Path(img_path).name}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_heatmap(img_np, gradcam_p1, alpha=0.5))
    axes[1].set_title(f"Phase 1 GradCAM\n(Block 7) — p={prob_p1:.3f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap(img_np, att_p3, alpha=0.5))
    axes[2].set_title(f"Phase 3 Disease Attention\n(14×14 learned) — p={prob_p3:.3f}", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(overlay_heatmap(img_np, gradcam_p3, alpha=0.5))
    axes[3].set_title(f"Phase 3 GradCAM\n(anatomy_att layer)", fontsize=10)
    axes[3].axis("off")

    plt.suptitle(f"Explanation Comparison: [{disease_idx}] {disease_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{save_prefix}_{disease_name.replace(' ','_')}_d{disease_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,  help="Path to chest X-ray PNG")
    parser.add_argument("--checkpoint",  required=True,  help="Phase 3 checkpoint path")
    parser.add_argument("--p1-checkpoint", default=None, help="Phase 1 checkpoint (auto-detected if omitted)")
    parser.add_argument("--disease",     type=int, default=4, help="Disease index 0-13 (default: 4=Effusion)")
    parser.add_argument("--all",         action="store_true", help="Generate all 14 disease maps")
    parser.add_argument("--prefix",      default="compare", help="Output file prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Available diseases: {list(enumerate(CLASSES))}")

    if args.all:
        compare_all_diseases(args.image, args.checkpoint, device, save_prefix=args.prefix)
    else:
        compare_phase1_vs_phase3(
            args.image, args.p1_checkpoint, args.checkpoint,
            args.disease, device, save_prefix=args.prefix
        )
