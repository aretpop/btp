import sys, torch
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import ChestXRayModel, NUM_CLASSES, CLASSES, get_transforms
from train_phase2_attention import MultiAttentionXRayModel
from plot_attention import plot_attention_maps
from PIL import Image

def find_unique_attention(model, device, image_paths):
    model.eval()
    transform = get_transforms("val")
    
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            logits, att_maps = model(img_tensor, return_attention=True)
            
            att_maps_np = att_maps.cpu().numpy()[0] # [14, 7, 7]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Look for an image where the model is highly confident (>50%) about a disease
            # AND its attention is focused heavily on the UPPER lung (y < 2.5) or LEFT lung (x < 2.5).
            for i, cls in enumerate(CLASSES):
                if probs[i] > 0.4:
                    heatmap = att_maps_np[i]
                    # Calculate center of mass
                    total_mass = np.sum(heatmap)
                    y_indices, x_indices = np.indices((7, 7))
                    com_y = np.sum(y_indices * heatmap) / total_mass
                    com_x = np.sum(x_indices * heatmap) / total_mass
                    
                    # Normal center is usually y=3.5, x=3.1 (lower right/center)
                    # We want to find something that looks UP (y < 2.5) or completely LEFT (x < 2.5 or x > 4.5)
                    if com_y < 2.8 or com_x > 4.0:
                        print(f"\nFound unique pathology!")
                        print(f"Image: {path.name}")
                        print(f"Disease: {cls} (Prob: {probs[i]:.2f})")
                        print(f"Focus Center of Mass: y={com_y:.2f}, x={com_x:.2f}")
                        
                        save_path = "unique_attention.png"
                        plot_attention_maps(path, model, device, save_path=save_path)
                        return path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load("checkpoints/best_attention_efficientnet_b3.pth", map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone", "efficientnet_b3")
    
    base_model = ChestXRayModel(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    in_features = base_model.backbone.classifier[1].in_features
    model = MultiAttentionXRayModel(base_model, feature_dim=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    
    img_dir = Path("data/nih_cxr14/images/images-224")
    all_imgs = list(img_dir.glob("*.png"))
    # Shuffle so we get a random search
    random.seed(99)
    random.shuffle(all_imgs)
    
    find_unique_attention(model, device, all_imgs)
