"""
compare_attention_maps.py
==========================
Loads the Phase 2 MultiAttentionXRayModel (train_phase2_attention.py)
and plots ALL 14 disease-specific attention maps side-by-side for
a given image, showing each disease's individual spatial mask.

This uses the EXISTING model — no new training required.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import ChestXRayModel, NUM_CLASSES, CLASSES, get_transforms
from train_phase2_attention import MultiAttentionXRayModel


def load_phase2_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b1")
    print(f"Loading {backbone_name} from {Path(checkpoint_path).name}...")
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def get_attention_maps(model, image_path, device, temperature: float = 1.0):
    img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img)
    transform = get_transforms("val")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, att_logits_raw = model(tensor, return_attention=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0]   # [14]

        # Fix 3: Temperature sharpening on raw logits BEFORE sigmoid.
        # T > 1 makes the distribution more peaked (sharper hot spots).
        # T = 1 is the original behaviour.
        if temperature != 1.0:
            att_logits_scaled = att_logits_raw  # already sigmoid'd in model
            # Re-derive logits via logit(p) = log(p/(1-p)), then scale
            p = att_logits_raw.clamp(1e-6, 1 - 1e-6)
            logits_raw = torch.log(p / (1 - p))
            att_maps = torch.sigmoid(logits_raw * temperature).cpu().numpy()[0]
        else:
            att_maps = att_logits_raw.cpu().numpy()[0]   # [14, 7, 7]

    return img_rgb, probs, att_maps


def plot_all_disease_maps(img_rgb, probs, att_maps, image_name, save_path):
    """
    Creates a clear visualization:
    - Row 0: Original X-ray + 6 highest-probability disease heatmaps
    - Row 1: Remaining 8 disease heatmaps
    - For each disease, shows BOTH: the raw attention mask alone AND overlaid on image
    """
    n_diseases = len(CLASSES)

    # Sort diseases by probability (show most likely first)
    sorted_idx = np.argsort(probs)[::-1]

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Phase 2 MultiAttentionXRayModel — 14 Disease-Specific Attention Maps\nImage: {image_name}",
                 fontsize=13, fontweight='bold', y=0.98)

    # Grid: 3 rows × 8 cols
    # Row 0: original + 7 disease overlays
    # Row 1: 7 disease overlays
    # Row 2: pure attention masks (no image)
    gs = gridspec.GridSpec(3, 8, hspace=0.4, wspace=0.3)

    # --- Original image (row 0, col 0) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_rgb)
    ax0.set_title("Original X-ray", fontsize=8, fontweight='bold')
    ax0.axis('off')

    H, W = img_rgb.shape[:2]

    for plot_i, d_idx in enumerate(sorted_idx):
        cls  = CLASSES[d_idx]
        prob = probs[d_idx]
        mask = att_maps[d_idx]  # [7, 7]

        # Upsample mask
        zoom_y = H / mask.shape[0]
        zoom_x = W / mask.shape[1]
        mask_up = zoom(mask, (zoom_y, zoom_x), order=3)

        # Normalise for display
        mn, mx = mask_up.min(), mask_up.max()
        mask_norm = (mask_up - mn) / (mx - mn + 1e-8)

        # Row 0: cols 1-7  (first 7 diseases by probability)
        # Row 1: cols 0-6  (next 7 diseases by probability)
        if plot_i < 7:
            row, col = 0, plot_i + 1
        else:
            row, col = 1, plot_i - 7

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_rgb)
        ax.imshow(mask_norm, cmap='jet', alpha=0.45)

        # Colour title by probability
        colour = 'red' if prob > 0.5 else ('orange' if prob > 0.3 else 'black')
        ax.set_title(f"{cls}\nP={prob:.3f}", fontsize=7, color=colour, fontweight='bold' if prob > 0.5 else 'normal')
        ax.axis('off')

    # --- Row 2: RAW attention masks (no image overlay) for comparison ---
    for plot_i, d_idx in enumerate(sorted_idx):
        mask = att_maps[d_idx]  # [7, 7] — raw, not upsampled
        zoom_y = H / mask.shape[0]
        zoom_x = W / mask.shape[1]
        mask_up = zoom(mask, (zoom_y, zoom_x), order=3)
        mn, mx = mask_up.min(), mask_up.max()
        mask_norm = (mask_up - mn) / (mx - mn + 1e-8)

        col = plot_i % 8
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(mask_norm, cmap='hot', interpolation='nearest')
        ax.set_title(f"{CLASSES[d_idx]} (raw)", fontsize=6)
        ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    print(f"Saved -> {save_path}")
    plt.close()


def print_raw_attention_values(att_maps, probs):
    """Print the raw [7x7] attention grid for each disease — proves they are NOT identical."""
    print("\n" + "="*70)
    print("RAW 7×7 ATTENTION TENSORS (proves uniqueness per disease)")
    print("="*70)
    np.set_printoptions(precision=3, suppress=True, linewidth=80)
    for i, cls in enumerate(CLASSES):
        print(f"\n[{i:2d}] {cls:22s}  (P={probs[i]:.3f})")
        print(att_maps[i])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_attention_efficientnet_b1.pth")
    parser.add_argument("--print-raw",  action="store_true", help="Print raw 7x7 tensors")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sharpen attention maps (>1 = sharper, e.g. 3.0). Fix 3, no retrain needed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_phase2_model(args.checkpoint, device)

    img_path  = Path(args.image)
    suffix    = f"_T{args.temperature}" if args.temperature != 1.0 else ""
    save_path = img_path.stem + f"_all14_disease_maps{suffix}.png"
    img_rgb, probs, att_maps = get_attention_maps(model, img_path, device, temperature=args.temperature)

    if args.print_raw:
        print_raw_attention_values(att_maps, probs)

    plot_all_disease_maps(img_rgb, probs, att_maps, img_path.name, save_path)
