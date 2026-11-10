"""
=============================================================================
predict_xray.py
=============================================================================
Run inference on a single chest X-ray image using the trained Phase 1 model.
Applies the optimal clinical thresholds found on the sealed test set.

Usage:
  python scripts/predict_xray.py path/to/image.png
=============================================================================
"""
import sys
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Need to import CLASSES from our training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import CLASSES, NUM_CLASSES, CHECKPOINT_DIR

def get_inference_transforms():
    """Exact same transforms used during validation/testing."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class AutoModel(nn.Module):
    """Dynamically creates B0 or B3 based on the checkpoint."""
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        if backbone_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights=None)
            dropout = 0.3
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            dropout = 0.2
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

def load_model_and_thresholds(device):
    # Try B3 first, fall back to B0
    ckpt_path = CHECKPOINT_DIR / "best_classifier_v2.pth"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "best_classifier_v1.pth"
        
    print(f"[INFO] Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    model = AutoModel(backbone_name, NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Load thresholds (fallback to 0.5 if missing)
    thresh_path = CHECKPOINT_DIR / "optimal_thresholds.json"
    if thresh_path.exists():
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    else:
        print("[WARNING] optimal_thresholds.json not found. Using 0.5 for all.")
        thresholds = {cls: 0.5 for cls in CLASSES}
        
    return model, thresholds

def predict(image_path: str):
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, thresholds = load_model_and_thresholds(device)
    transform = get_inference_transforms()

    # Load and preprocess image
    # Note: CXR images are usually grayscale, but EfficientNet expects RGB (3 channels)
    # PIL .convert('RGB') handles duplicating the grayscale channel to R, G, B
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # Run inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    print("\n" + "="*50)
    print(f" INFERENCE RESULTS: {Path(image_path).name}")
    print("="*50)
    
    findings = []
    
    for i, cls in enumerate(CLASSES):
        p = probs[i]
        thr = thresholds.get(cls, 0.5)
        is_positive = p >= thr
        
        # Format the output nicely (no emojis, to avoid cp1252 errors)
        marker = "!!! DETECTED" if is_positive else "   Clear   "
        
        if is_positive:
            findings.append(f"{cls} (Prob: {p:.3f}, Threshold: {thr:.3f})")
            
        print(f"[{marker}] {cls:<20}: {p:.3f}  (Cutoff: {thr:.3f})")
        
    print("="*50)
    if not findings:
        print(" FINAL DIAGNOSIS: No Finding (Healthy)")
    else:
        print(" FINAL DIAGNOSIS: Positive for:")
        for f in findings:
            print(f"   - {f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CXR Phase 1 Inference")
    parser.add_argument("image", type=str, help="Path to the .png or .jpg X-ray image")
    args = parser.parse_args()
    
    predict(args.image)
