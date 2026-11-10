import os
import copy
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_phase1_classifier import ChestXRayModel, NUM_CLASSES, CLASSES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label imbalance.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic BCE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        
        # Asymmetric Focusing
        los_pos = los_pos * torch.pow(1 - xs_pos, self.gamma_pos)
        los_neg = los_neg * torch.pow(1 - xs_neg, self.gamma_neg)
        
        loss = - (los_pos + los_neg)
        return loss.mean()


class MultiScaleFeatureBlock(nn.Module):
    """
    Captures features at small, medium, and large receptive fields.
    Crucial for differentiating diffuse patterns (edema) from local ones (nodule).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_c = out_channels // 4
        
        # 1. Local (1x1)
        self.branch1 = nn.Conv2d(in_channels, branch_c, kernel_size=1, bias=False)
        # 2. Medium (3x3)
        self.branch2 = nn.Conv2d(in_channels, branch_c, kernel_size=3, padding=1, bias=False)
        # 3. Large (3x3 Dilated r=2)
        self.branch3 = nn.Conv2d(in_channels, branch_c, kernel_size=3, padding=2, dilation=2, bias=False)
        # 4. Global Context (3x3 Dilated r=3)
        self.branch4 = nn.Conv2d(in_channels, branch_c, kernel_size=3, padding=3, dilation=3, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.bn(out))


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Answers: 'Which feature channels are relevant to chest pathologies?'
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        attention = self.fc(y)
        assert attention.shape == (b, c, 1, 1), f"Channel attention shape mismatch: {attention.shape}"
        return x * attention


class SpatialAttention(nn.Module):
    """
    CBAM-style Spatial Attention.
    Answers: 'Which global regions of the lung field contain abnormalities?'
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.sigmoid(self.conv1(y))
        b, _, h, w = x.size()
        assert attention.shape == (b, 1, h, w), f"Spatial attention shape mismatch: {attention.shape}"
        return x * attention


class AnatomicalAttention(nn.Module):
    """
    Anatomical Mask-Guided Attention Module.
    If external masks (e.g. lung/heart segmentations) are provided during forward(),
    they are resized and multiplied into the feature map to guide the network.
    If no mask is provided, it acts as an identity/soft-spatial pass.
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1)) # Learnable weight for mask injection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        if mask is None:
            # Fallback when anatomical segmentation masks are unavailable
            return x
        
        b, c, h, w = x.shape
        # Mask shape is expected to be [B, 1, H_orig, W_orig]
        # We use bilinear interpolation because soft probabilities are better for feature gradients than nearest-neighbor.
        resized_mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        assert resized_mask.shape == (b, 1, h, w)
        
        # Branch 1: Features inside the anatomical structure
        x_masked = x * resized_mask
        
        # Branch 2: Residual Context (so the model isn't completely blind outside the mask)
        out = torch.cat([x, x_masked * self.alpha], dim=1)
        out = self.fusion_conv(out)
        return out


class MultiAttentionXRayModel(nn.Module):
    """
    Final Architecture: Design 3 (Hybrid)
    Frozen Backbone -> Multi-Scale -> Channel -> Spatial -> Anatomical -> 14 Disease-Specific Heads.
    """
    def __init__(self, original_backbone_model: nn.Module, feature_dim: int, num_classes=14):
        super().__init__()
        
        # 1. DEEPCOPY THE BACKBONE SAFELY.
        # The user requested absolute immutability of their existing trained models.
        self.backbone = copy.deepcopy(original_backbone_model)
        
        # Freeze the backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.feature_dim = feature_dim
        self.c2 = 512 # Internal bottleneck dimension
        
        # 2. Shared Refinement Path
        self.multi_scale = MultiScaleFeatureBlock(in_channels=feature_dim, out_channels=self.c2)
        self.channel_att = ChannelAttention(channels=self.c2)
        self.spatial_att = SpatialAttention(kernel_size=7)
        self.anatomical  = AnatomicalAttention(channels=self.c2)
        
        # 3. 14 Independent Disease-Specific Spatial Attention Heads
        # Rather than 14 huge CNNs, we use one efficient grouped generator.
        self.disease_attention_generator = nn.Sequential(
            nn.Conv2d(self.c2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
            # No sigmoid here, we apply it dynamically in forward
        )
        
        # 14 Independent linear classifiers applied to the pooled representation of each disease head
        self.classifiers = nn.ModuleList([
            nn.Linear(self.c2, 1) for _ in range(num_classes)
        ])

    def train(self, mode=True):
        """
        OVERRIDE: Ensure the frozen backbone NEVER enters train() mode.
        If it does, its BatchNorm running_mean and running_var will drift, 
        corrupting the frozen pretrained features!
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_backbone_features(self, x):
        """Safely extract spatial feature maps prior to the pooling/classifier of the original model."""
        # For torchvision EfficientNets, `model.features` contains the conv blocks.
        if hasattr(self.backbone, 'backbone') and hasattr(self.backbone.backbone, 'features'):
            # Handling our custom ChestXRayModel wrapper
            return self.backbone.backbone.features(x)
        elif hasattr(self.backbone, 'features'):
            # Handling bare torchvision EfficientNet
            return self.backbone.features(x)
        else:
            raise NotImplementedError("Cannot find spatial feature maps in this backbone structure.")

    def forward(self, x, mask=None, debug=False, return_attention=False):
        b, c, h, w = x.shape
        if debug: print(f"Input: [B={b}, C={c}, H={h}, W={w}]")
        
        # --- FROZEN BACKBONE ---
        with torch.no_grad():
            f = self.extract_backbone_features(x)
        
        b, c_f, h_f, w_f = f.shape
        if debug: print(f"Backbone features: [B={b}, C={c_f}, H={h_f}, W={w_f}]")
        assert c_f == self.feature_dim, f"Expected {self.feature_dim} channels, got {c_f}"
        
        # --- SHARED MULTI-ATTENTION ---
        # 1. Multi-scale fusion
        f_ms = self.multi_scale(f)
        if debug: print(f"After multi-scale: {f_ms.shape}")
        
        # 2. Channel Attention
        f_ca = self.channel_att(f_ms)
        if debug: print(f"After channel attention: {f_ca.shape}")
        
        # 3. Spatial Attention
        f_sa = self.spatial_att(f_ca)
        if debug: print(f"After spatial attention: {f_sa.shape}")
        
        # 4. Anatomical Attention
        f_shared = self.anatomical(f_sa, mask)
        if debug: print(f"After anatomical attention: {f_shared.shape}")
        
        # --- DISEASE-SPECIFIC HEADS ---
        disease_att_logits = self.disease_attention_generator(f_shared)
        disease_atts = torch.sigmoid(disease_att_logits)
        if debug: print(f"Disease attention maps: {disease_atts.shape}")
        
        logits = []
        f_shared_expanded = f_shared.unsqueeze(1)
        disease_atts_expanded = disease_atts.unsqueeze(2)
        f_weighted = f_shared_expanded * disease_atts_expanded
        
        z = f_weighted.mean(dim=(3, 4))
        
        for d in range(14):
            z_d = z[:, d, :]
            logit_d = self.classifiers[d](z_d)
            logits.append(logit_d)
            
        final_logits = torch.cat(logits, dim=1)
        
        if return_attention:
            return final_logits, disease_atts
        return final_logits


def test_immutability(original_checkpoint_path):
    """
    Verifies that loading the original checkpoint and passing it to the 
    MultiAttentionXRayModel absolutely does NOT mutate the original weights.
    """
    print("=" * 60)
    print(f"VERIFYING BACKBONE IMMUTABILITY FOR: {os.path.basename(original_checkpoint_path)}")
    print("=" * 60)
    
    device = torch.device('cpu')
    ckpt = torch.load(original_checkpoint_path, map_location=device, weights_only=True)
    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    
    # 1. Load original
    original_model = ChestXRayModel(backbone_name=backbone_name, pretrained=False)
    original_model.load_state_dict(ckpt["model_state"])
    original_model.eval()
    
    # Snapshot original weights in memory
    original_state = {k: v.clone() for k, v in original_model.state_dict().items()}
    
    # 2. Build New Architecture
    in_features = original_model.backbone.classifier[1].in_features
    print(f"Detected Backbone: {backbone_name} (in_features={in_features})")
    
    attention_model = MultiAttentionXRayModel(original_model, feature_dim=in_features)
    
    # 3. Simulate training pass to ensure no bleed-over
    attention_model.train()
    dummy_input = torch.randn(2, 3, 224, 224)
    out = attention_model(dummy_input, debug=True)
    loss = AsymmetricLoss()(out, torch.zeros_like(out))
    loss.backward()
    
    # 4. Verify Immutability
    mutated = False
    for k, v in original_model.state_dict().items():
        if not torch.equal(v, original_state[k]):
            print(f"ERROR: Weight mutated -> {k}")
            mutated = True
            
    assert not mutated, "CRITICAL ERROR: Original backbone was modified!"
    print("SUCCESS: Original backbone is completely untouched and deep-copied.")
    print("=" * 60)
    return attention_model


def train_attention_model(args):
    import mlflow
    from torch.utils.data import DataLoader, random_split, Subset
    from sklearn.metrics import roc_auc_score
    import numpy as np
    from tqdm import tqdm
    from train_phase1_classifier import CXRDataset, get_transforms, CSV_PATH, IMAGES_DIR, CHECKPOINT_DIR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Dataset splits (Identical to Phase 1)
    full_train_ds = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("train"))
    full_val_ds   = CXRDataset(CSV_PATH, IMAGES_DIR, transform=get_transforms("val"))

    n_total = len(full_train_ds)
    n_test  = int(n_total * 0.20)
    n_val   = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(range(n_total), [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(Subset(full_train_ds, train_idx.indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(full_val_ds, val_idx.indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load the original base model to act as a frozen feature extractor
    ckpt_path = Path(args.base_checkpoint)
    print(f"Loading base checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    backbone_name = ckpt.get("backbone", "efficientnet_b0")
    
    original_model = ChestXRayModel(backbone_name=backbone_name, pretrained=False)
    original_model.load_state_dict(ckpt["model_state"])
    in_features = original_model.backbone.classifier[1].in_features
    
    # 3. Initialize new Attention Architecture
    model = MultiAttentionXRayModel(original_model, feature_dim=in_features).to(device)
    
    # Verify ONLY the attention parameters require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total trainable parameters in attention heads: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    
    # 4. MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ChestXRay14_Phase2_Attention")
    
    best_auroc = 0.0
    save_name = CHECKPOINT_DIR / f"best_attention_{backbone_name}.pth"
    
    with mlflow.start_run(run_name=f"phase2_attn_{backbone_name}"):
        mlflow.log_params(vars(args))
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            # --- TRAIN ---
            model.train() # Custom train() method ensures backbone stays in eval()
            train_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # --- VALIDATE ---
            model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    all_probs.append(torch.sigmoid(logits).cpu())
                    all_labels.append(labels.cpu())
                    
            val_loss /= len(val_loader)
            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()
            
            per_class_aurocs = []
            for i in range(NUM_CLASSES):
                try:
                    score = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    per_class_aurocs.append(score)
                except ValueError:
                    per_class_aurocs.append(0.5)
            
            macro_auroc = np.mean(per_class_aurocs)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {macro_auroc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_macro_auroc": macro_auroc}, step=epoch)
            
            if macro_auroc > best_auroc:
                best_auroc = macro_auroc
                print(f"New best AUROC! Saving checkpoint to {save_name.name}...")
                torch.save({
                    "epoch": epoch,
                    "backbone": backbone_name,
                    "model_state": model.state_dict(),
                    "macro_auroc": macro_auroc,
                }, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, required=True, help="Path to frozen B0/B1/B3 checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    # Optionally run the immutability test before training
    test_immutability(args.base_checkpoint)
    
    train_attention_model(args)
