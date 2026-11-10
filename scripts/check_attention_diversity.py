import sys, torch
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import ChestXRayModel, NUM_CLASSES, CLASSES, get_transforms
from train_phase2_attention import MultiAttentionXRayModel
from PIL import Image

def analyze_attention_diversity(model, device, image_paths):
    model.eval()
    transform = get_transforms("val")
    
    all_att_maps = []
    
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            _, att_maps = model(img_tensor, return_attention=True)
            # att_maps shape: [1, 14, 7, 7]
            all_att_maps.append(att_maps.cpu().numpy()[0])
            
    all_att_maps = np.array(all_att_maps) # [N, 14, 7, 7]
    N = len(image_paths)
    
    print(f"=== Attention Matrix Analysis ({N} images) ===")
    
    # 1. Diversity across diseases (Are the 14 maps identical?)
    # Calculate std dev across the 14 disease channels per image, then average
    disease_std = np.mean(np.std(all_att_maps, axis=1))
    print(f"Average standard deviation between the 14 disease masks per image: {disease_std:.4f}")
    if disease_std < 0.05:
        print("WARNING: The 14 disease masks are almost identical. The attention generator suffered from mode collapse!")
    else:
        print("GOOD: The 14 disease masks show variance and are structurally distinct.")
        
    # 2. Spatial bias (Does it always look at the lower right lung?)
    # Average across all images and all diseases to get the "Global Average Heatmap"
    global_heatmap = np.mean(all_att_maps, axis=(0, 1)) # [7, 7]
    print("\nGlobal Average Heatmap (7x7 spatial grid over all diseases and images):")
    np.set_printoptions(precision=3, suppress=True)
    print(global_heatmap)
    
    # 3. Disease-specific spatial bias
    print("\nDisease-specific 'Center of Mass' (Row, Col) on 7x7 grid (0-6):")
    for i, cls in enumerate(CLASSES):
        disease_heatmap = np.mean(all_att_maps[:, i, :, :], axis=0) # [7, 7]
        # Calculate center of mass
        total_mass = np.sum(disease_heatmap)
        y_indices, x_indices = np.indices((7, 7))
        com_y = np.sum(y_indices * disease_heatmap) / total_mass
        com_x = np.sum(x_indices * disease_heatmap) / total_mass
        print(f" - {cls:20s}: y={com_y:.2f}, x={com_x:.2f} (Max Val: {np.max(disease_heatmap):.2f})")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load("checkpoints/best_attention_efficientnet_b3.pth", map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b3")
    
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    
    # Get 20 random images
    img_dir = Path("data/nih_cxr14/images/images-224")
    all_imgs = list(img_dir.glob("*.png"))
    random.seed(42)
    sample_imgs = random.sample(all_imgs, min(20, len(all_imgs)))
    
    analyze_attention_diversity(model, device, sample_imgs)
