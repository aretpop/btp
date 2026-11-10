"""
src/datasets/chexpert_report_dataset.py
========================================
Streaming dataset for VLM report generation fine-tuning.

Each sample yields:
    image_pil      : PIL.Image  — original chest X-ray (224×224 grayscale→RGB)
    attn_grid_pil  : PIL.Image  — 3×4 grid of 11 disease attention maps (224×224)
    prob_text      : str        — formatted classifier probability text
    report_text    : str        — target report (findings + impression)
    labels_dict    : dict       — ground truth binary labels

Architecture:
    - Streams from Redivis LRU cache (same as transfer training)
    - Randomly samples from train_report_index.csv (uniform, no replacement per epoch)
    - Our Stage B classifier runs in eval/no_grad to produce probs + attn maps
    - Attention maps are visualised as a color grid → passed as 2nd image to InternVL2
"""

import os, sys, random, logging
os.environ.setdefault("REDIVIS_API_TOKEN", "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk")
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, RandomSampler
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHEXPERT_LABELS = [
    "No Finding", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Support Devices",
]

# Youden-J thresholds from Stage B validation
STAGE_B_THRESHOLDS = {
    "No Finding": 0.885, "Cardiomegaly": 0.511, "Lung Opacity": 0.586,
    "Lung Lesion": 0.487, "Edema": 0.537, "Consolidation": 0.476,
    "Pneumonia": 0.500, "Pneumothorax": 0.478, "Pleural Effusion": 0.579,
    "Pleural Other": 0.351, "Support Devices": 0.715,
}

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def attn_maps_to_grid_pil(disease_maps: torch.Tensor, size: int = 224) -> Image.Image:
    """
    Convert [11, 14, 14] attention maps → single RGB PIL image (3×4 grid).
    Each map is colourised with the 'hot' colormap.
    """
    maps = disease_maps.detach().cpu().float()                   # [11, 14, 14]
    maps = F.interpolate(maps.unsqueeze(0), size=(size // 4, size // 4),
                          mode="bilinear", align_corners=False)[0]  # [11, 56, 56]

    # Normalise each map independently to [0, 1]
    for i in range(maps.shape[0]):
        mn, mx = maps[i].min(), maps[i].max()
        maps[i] = (maps[i] - mn) / (mx - mn + 1e-8)

    rows, cols = 3, 4
    h, w = maps.shape[1], maps.shape[2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    colormap = cm.get_cmap("hot")
    for idx in range(11):
        r, c = idx // cols, idx % cols
        colored = (colormap(maps[idx].numpy())[:, :, :3] * 255).astype(np.uint8)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = colored

    return Image.fromarray(grid).resize((size, size), Image.BILINEAR)


def format_classifier_text(probs: np.ndarray, labels: list) -> str:
    """Format classifier probabilities as structured text prompt."""
    lines = ["CLASSIFIER ANALYSIS:"]
    # Sort by probability descending
    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        name = labels[idx]
        p = probs[idx]
        thresh = STAGE_B_THRESHOLDS.get(name, 0.5)
        call = "POSITIVE" if p >= thresh else "negative"
        lines.append(f"  {name}: {p*100:.1f}% [{call}]")
    return "\n".join(lines)


class CheXpertReportDataset(Dataset):
    """
    Streaming dataset for VLM fine-tuning.

    Parameters
    ----------
    report_index_path : path to train_report_index.csv
    classifier_model  : loaded CheXpertTransferModel (eval, no_grad)
    cache_dir         : path to chexpert_cache (LRU NVMe cache)
    redivis_table     : redivis Table object for PNG_train streaming
    epoch_size        : how many samples per epoch (default 8000)
    seed              : random seed
    """

    def __init__(
        self,
        report_index_path: str,
        classifier_model,
        cache_dir: str,
        redivis_table=None,
        epoch_size: int = 8000,
        seed: int = 42,
    ):
        import pandas as pd
        self.df = pd.read_csv(report_index_path)
        # Frontal only (should already be filtered but double-check)
        self.df = self.df[
            self.df["local_filename"].str.contains("frontal", na=False)
        ].reset_index(drop=True)
        log.info(f"Report dataset: {len(self.df):,} samples with reports")

        self.classifier = classifier_model
        self.cache_dir  = Path(cache_dir)
        self.table      = redivis_table
        self.epoch_size = epoch_size
        self.seed       = seed
        self._rng       = random.Random(seed)

        # Sample epoch_size indices uniformly
        self._resample()

    def _resample(self):
        """Draw a new uniform random subset for the current epoch."""
        self._indices = self._rng.choices(range(len(self.df)), k=self.epoch_size)

    def __len__(self):
        return self.epoch_size

    def _load_image(self, file_id: str, local_filename: str) -> Optional[Image.Image]:
        """Try local cache first, then stream from Redivis."""
        # Try various cache filename patterns
        for fn in [file_id, file_id + ".png", file_id + ".jpg",
                   local_filename.replace("/", "_"), Path(local_filename).name]:
            p = self.cache_dir / fn
            if p.exists():
                return Image.open(p).convert("RGB")

        # Stream from Redivis
        if self.table is not None:
            try:
                f = self.table.file(local_filename)
                import io
                with f.open("rb") as fh:
                    data = fh.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                # Cache it
                cache_path = self.cache_dir / (file_id + ".jpg")
                img.save(cache_path, "JPEG", quality=92)
                return img
            except Exception as e:
                log.warning(f"Redivis stream failed for {local_filename}: {e}")

        return None

    def __getitem__(self, idx):
        row_idx = self._indices[idx]
        row     = self.df.iloc[row_idx]

        file_id       = str(row["file_id"])
        local_fn      = str(row["local_filename"])
        findings      = str(row["section_findings"]).strip()
        impression    = str(row["section_impression"]).strip()
        report_target = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

        # Load image
        pil_img = self._load_image(file_id, local_fn)
        if pil_img is None:
            # Fallback to blank if image missing
            pil_img = Image.new("RGB", (224, 224), 128)
            log.warning(f"Could not load image {local_fn}, using blank")

        # Run classifier (frozen, no_grad) → probs + attention maps
        x = IMG_TRANSFORM(pil_img).unsqueeze(0)
        device = next(self.classifier.parameters()).device
        x = x.to(device)

        with torch.inference_mode():
            logits, disease_maps = self.classifier(x, return_attention=True)
            probs = torch.sigmoid(logits)[0].cpu().numpy()      # [11]
            attn  = disease_maps[0].cpu()                        # [11, 14, 14]

        # Build attention grid image
        attn_grid_pil = attn_maps_to_grid_pil(attn, size=224)

        # Build classifier text
        prob_text = format_classifier_text(probs, CHEXPERT_LABELS)

        return {
            "image_pil":     pil_img.resize((224, 224)),
            "attn_grid_pil": attn_grid_pil,
            "prob_text":     prob_text,
            "report_text":   report_target,
            "file_id":       file_id,
        }

    def on_epoch_end(self):
        """Call at end of each epoch to resample."""
        self._resample()
