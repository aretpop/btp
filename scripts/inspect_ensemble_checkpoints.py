import torch
from pathlib import Path

ckpt_dir = Path("checkpoints/chexpert_ensemble")
for p in sorted(ckpt_dir.glob("*.pth")):
    data = torch.load(p, map_location="cpu")
    print("=" * 65)
    print(f"Checkpoint: {p.name}")
    print(f"Stage: {data.get('stage')}, Epoch: {data.get('epoch')}, Loss: {data.get('loss_type')}")
    print(f"Macro Val AUROC: {data.get('val_auroc'):.4f} | Macro Val F1: {data.get('val_f1'):.4f}")
    if "per_class_auroc" in data and data["per_class_auroc"]:
        print("Per-class AUROC:")
        for k, v in data["per_class_auroc"].items():
            print(f"  {k:28s}: {v:.4f}")

