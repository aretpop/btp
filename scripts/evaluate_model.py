import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchmetrics.classification import MultilabelAUROC
from pathlib import Path
import numpy as np

# Import classes and configs from our training script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_phase1_classifier import (
    ChestXRayModel, CXRDataset, get_transforms, validate,
    CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR, CLASSES, NUM_CLASSES
)

def evaluate_best_model():
    print("=== Phase 1 Model Accuracy Test ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = CHECKPOINT_DIR / "best_classifier_v1.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    print("1. Loading Validation Data...")
    # Recreate the exact validation split used during training (Seed 42, 80/20)
    full_val_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))
    n_total = len(full_val_ds)
    n_val = int(n_total * 0.20)
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(42)
    _, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
    val_ds = Subset(full_val_ds, val_indices.indices)
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=(device.type == "cuda")
    )
    print(f"   Loaded {len(val_ds)} validation samples.")
    
    print("2. Loading Model & Checkpoint...")
    model = ChestXRayModel(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print(f"   Loaded checkpoint from Epoch {checkpoint.get('epoch', 'N/A')}")
    
    print("3. Running Evaluation (Calculating AUROC)...")
    # We pass a dummy criterion since we just want the AUROC metrics
    criterion = nn.BCEWithLogitsLoss() 
    auroc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average=None).to(device)
    
    avg_loss, macro_auroc, per_cls_auc = validate(
        model, val_loader, criterion, auroc_metric, device
    )
    
    print("\n" + "="*50)
    print(" FINAL ACCURACY / AUROC TEST RESULTS")
    print("="*50)
    print(f" Macro-AUROC : {macro_auroc:.4f}")
    print(f" Val Loss    : {avg_loss:.4f}\n")
    print(" Per-Class AUROC:")
    
    for cls_name, auc in zip(CLASSES, per_cls_auc):
        print(f"  - {cls_name:<20s}: {auc:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_best_model()
