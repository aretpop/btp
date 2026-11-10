import sys, torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import ChestXRayModel, NUM_CLASSES, CLASSES, get_transforms
from train_phase2_attention import MultiAttentionXRayModel

def plot_attention_maps(image_path, model, device, save_path="attention_visualization.png"):
    # 1. Load and prepare image
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)
    
    transform = get_transforms("val")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. Forward pass with return_attention=True
    model.eval()
    with torch.no_grad():
        logits, att_maps = model(img_tensor, return_attention=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        att_maps = att_maps.cpu().numpy()[0] # Shape: [14, 7, 7]
        
    # 3. Plotting
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot original image in the first box
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image\n{image_path.name}")
    axes[0].axis('off')
    
    # Plot 14 disease-specific masks
    for i, cls in enumerate(CLASSES):
        ax = axes[i + 1]
        mask = att_maps[i]
        
        # Resize mask to original image size (224x224) using cubic interpolation (zoom)
        zoom_factors = (img_rgb.shape[0] / mask.shape[0], img_rgb.shape[1] / mask.shape[1])
        mask_resized = zoom(mask, zoom_factors, order=3)
        
        # Normalize mask for visualization
        mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)
        
        ax.imshow(img_rgb)
        ax.imshow(mask_norm, cmap='jet', alpha=0.4) # overlay heatmap
        
        prob_str = f"Prob: {probs[i]:.3f}"
        ax.set_title(f"{cls}\n{prob_str}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Attention model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    
    print(f"Loading {backbone_name} from {args.checkpoint}...")
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    
    # Plot
    output_name = Path(args.image).stem + "_attention.png"
    plot_attention_maps(Path(args.image), model, device, save_path=output_name)
