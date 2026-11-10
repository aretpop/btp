"""
CheXpert+ Streaming Dataset
- Downloads images on-the-fly from Redivis during training
- LRU disk cache (configurable size limit)
- Stratified sampler for uniform label coverage
"""
import os
import io
import time
import hashlib
import shutil
import threading
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as T


# ── Lazy redivis import (set token before use) ──────────────────────────────
_redivis_client = None
_redivis_lock = threading.Lock()

def _get_redivis_table(api_token: str, table_ref: str):
    """Thread-safe lazy init of redivis table."""
    global _redivis_client
    with _redivis_lock:
        if _redivis_client is None:
            os.environ["REDIVIS_API_TOKEN"] = api_token
            import redivis
            _redivis_client = redivis
    user = _redivis_client.user("aimi")
    dataset = user.dataset("chexpert_plus:5yyj:v1_0")
    return dataset.table(table_ref)


# ── LRU Disk Cache ───────────────────────────────────────────────────────────
class LRUDiskCache:
    """
    Keeps the N most recently used images on disk.
    Thread-safe. Automatically evicts oldest files when over budget.
    """
    def __init__(self, cache_dir: Path, max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_size_gb * 1024 ** 3)
        self._lock = threading.Lock()
        # OrderedDict: file_id -> (path, size_bytes)
        self._index: OrderedDict = OrderedDict()
        self._current_bytes = 0
        self._rebuild_index()

    def _rebuild_index(self):
        """Scan existing cache files on startup."""
        for p in sorted(self.cache_dir.glob("*.png"), key=lambda f: f.stat().st_atime):
            size = p.stat().st_size
            file_id = p.stem
            self._index[file_id] = (p, size)
            self._current_bytes += size

    def get(self, file_id: str) -> Optional[Path]:
        with self._lock:
            if file_id in self._index:
                path = self._index[file_id][0]
                if path.exists():
                    self._index.move_to_end(file_id)
                    return path
                else:
                    old_path, old_size = self._index.pop(file_id)
                    self._current_bytes = max(0, self._current_bytes - old_size)
        p = self.cache_dir / f"{file_id}.png"
        if p.exists():
            with self._lock:
                size = p.stat().st_size
                self._index[file_id] = (p, size)
                self._current_bytes += size
            return p
        return None

    def put(self, file_id: str, img_bytes: bytes) -> Path:
        path = self.cache_dir / f"{file_id}.png"
        path.write_bytes(img_bytes)
        size = len(img_bytes)
        with self._lock:
            self._index[file_id] = (path, size)
            self._current_bytes += size
            self._evict_if_needed()
        return path

    def _evict_if_needed(self):
        """Evict LRU files until under budget. Must be called with lock held."""
        while self._current_bytes > self.max_bytes and self._index:
            old_id, (old_path, old_size) = self._index.popitem(last=False)
            try:
                old_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._current_bytes -= old_size

    @property
    def size_gb(self):
        return self._current_bytes / 1024**3

    @property
    def num_files(self):
        return len(self._index)


# ── Stratified Sampler ────────────────────────────────────────────────────────
class MultiLabelStratifiedSampler(Sampler):
    """
    Samples indices such that each mini-batch has roughly equal
    representation of all positive labels (multi-label uniform sampling).
    
    Strategy: Round-robin across disease buckets. For each disease,
    maintain a shuffled list of positive indices and cycle through them.
    """
    LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]

    def __init__(self, df: pd.DataFrame, num_samples: Optional[int] = None, seed: int = 42, labels: Optional[list] = None):
        self.df = df.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.labels = list(labels) if labels is not None else list(self.LABELS)
        
        # Build buckets: label -> list of indices with positive label
        self.buckets = {}
        for label in self.labels:
            if label in df.columns:
                pos_idx = np.where(df[label].values == 1.0)[0]
                if len(pos_idx) > 0:
                    self.buckets[label] = pos_idx.tolist()
        
        # All indices for fallback
        self.all_indices = list(range(len(df)))
        self.num_samples = num_samples or len(df)
        self.current_indices = None
        
        print(f"  Stratified sampler: {len(self.buckets)} disease buckets")
        for lbl, idx in self.buckets.items():
            print(f"    {lbl:30s}: {len(idx):6,} positive samples")

    def generate_indices(self, seed: Optional[int] = None) -> list:
        """Generate and cache sampled indices for an epoch."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Shuffle each bucket
        bucket_iters = {}
        for lbl, idx in self.buckets.items():
            shuffled = self.rng.permutation(idx).tolist()
            bucket_iters[lbl] = iter(shuffled * (self.num_samples // len(idx) + 2))
        
        labels = list(bucket_iters.keys())
        n_labels = len(labels)
        samples = []
        
        for i in range(self.num_samples):
            lbl = labels[i % n_labels]
            try:
                samples.append(next(bucket_iters[lbl]))
            except StopIteration:
                # Restart this bucket
                bucket_iters[lbl] = iter(
                    self.rng.permutation(self.buckets[lbl]).tolist() * 3
                )
                samples.append(next(bucket_iters[lbl]))
        
        self.current_indices = samples
        return samples

    def __iter__(self):
        """Interleave samples from each disease bucket."""
        if self.current_indices is None:
            self.generate_indices()
        return iter(self.current_indices)

    def __len__(self):
        return self.num_samples


# ── Streaming Dataset ────────────────────────────────────────────────────────
class StreamingCheXpertDataset(Dataset):
    """
    Streams CheXpert+ images from Redivis on-the-fly.
    
    - No need to pre-download the full dataset
    - LRU disk cache avoids re-downloading the same image
    - Works with PyTorch DataLoader (num_workers > 0)
    
    Args:
        index_csv:      Path to train_index.csv or valid_index.csv
        api_token:      Redivis API token
        table_ref:      Redivis table reference (e.g. "PNG_train")
        cache_dir:      Directory for LRU disk cache
        cache_size_gb:  Max disk cache size in GB
        transform:      torchvision transforms
        frontal_only:   If True, only load frontal (AP/PA) views
        uncertainty_policy: 'ignore' (skip -1), 'zeros' (treat as 0), 
                             'ones' (treat as 1) — for uncertain labels
    """
    LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]

    def __init__(
        self,
        index_csv: str,
        api_token: str,
        table_ref: str = "PNG_train",
        cache_dir: str = "data/chexpert_cache",
        cache_size_gb: float = 5.0,
        labels: Optional[list] = None,
        transform=None,
        frontal_only: bool = True,
        uncertainty_policy: str = "zeros",  # 'ignore' | 'zeros' | 'ones'
    ):
        self.api_token = api_token
        self.table_ref = table_ref
        self.labels = list(labels) if labels is not None else list(self.LABELS)
        self.uncertainty_policy = uncertainty_policy
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load index
        df = pd.read_csv(index_csv)
        
        # Filter frontal only
        if frontal_only:
            df = df[df['local_filename'].str.contains('frontal', na=False)].reset_index(drop=True)
        
        self.df = df
        self.cache = LRUDiskCache(Path(cache_dir), max_size_gb=cache_size_gb)
        
        # Build file_id lookup
        self.file_id_map = dict(zip(df['file_name'], df['file_id']))
        
        # Lazy redivis table (initialized on first use per worker)
        self._table = None
        self._table_files = None  # file_name -> File object
        
        print(f"StreamingCheXpertDataset: {len(self.df):,} images")
        print(f"  Cache: {self.cache.num_files} files, {self.cache.size_gb:.2f} GB used")

    def _init_table(self):
        """Initialize redivis table (called once per worker process)."""
        if self._table is None:
            self._table = _get_redivis_table(self.api_token, self.table_ref)
            # Build file lookup (name → File object)
            # Note: We use file_id directly to avoid listing all files
            import redivis
            user = redivis.user("aimi")
            ds = user.dataset("chexpert_plus:5yyj:v1_0")
            self._redivis_dataset = ds

    def _download_image(self, file_id: str, file_name: str) -> bytes:
        """Download a single image from Redivis by file_name (binary PNG)."""
        self._init_table()
        # Use file_name (file_id is deprecated per redivis API)
        table = self._redivis_dataset.table(self.table_ref)
        f = table.file(file_name)
        buf = io.BytesIO()
        with f.open('rb') as fh:    # binary mode — PNG bytes, not UTF-8 text
            chunk = fh.read(65536)
            while chunk:
                buf.write(chunk)
                chunk = fh.read(65536)
        return buf.getvalue()

    def prefetch(self, indices, max_workers: int = 8):
        """Pre-download a batch of indices in parallel using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        uncached = [i for i in indices if self.cache.get(self.df.iloc[i]['file_id']) is None]
        if not uncached:
            return
        
        self._init_table()
        def _fetch(idx):
            row = self.df.iloc[idx]
            fid = row['file_id']
            fn = row['local_filename'] if 'local_filename' in row else row['file_name']
            if self.cache.get(fid) is None:
                try:
                    b = self._download_image(fid, fn)
                    self.cache.put(fid, b)
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_fetch, uncached))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['file_id']
        file_name = row['local_filename'] if 'local_filename' in row else row['file_name']

        # ── Get image ──────────────────────────────────────────────────
        cached_path = self.cache.get(file_id)
        img = None
        if cached_path is not None and cached_path.exists():
            try:
                img = Image.open(cached_path).convert('RGB')
            except Exception:
                img = None

        if img is None:
            # Download from Redivis
            for attempt in range(3):
                try:
                    img_bytes = self._download_image(file_id, file_name)
                    self.cache.put(file_id, img_bytes)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"Failed to download {file_id}: {e}")
                    time.sleep(2 ** attempt)

        if self.transform:
            img = self.transform(img)

        # ── Get labels ────────────────────────────────────────────────
        labels = []
        for lbl in self.labels:
            val = row.get(lbl, None)
            if pd.isna(val) or val is None:
                val = 0.0  # unlabeled → negative
            elif val == -1.0:
                # Uncertain label
                if self.uncertainty_policy == 'zeros':
                    val = 0.0
                elif self.uncertainty_policy == 'ones':
                    val = 1.0
                else:  # ignore → mask out in loss
                    val = -1.0
            labels.append(float(val))

        return img, torch.tensor(labels, dtype=torch.float32)

    def get_label_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance."""
        weights = []
        for lbl in self.labels:
            if lbl in self.df.columns:
                pos = (self.df[lbl] == 1.0).sum()
                neg = (self.df[lbl] != 1.0).sum()
                weight = neg / max(pos, 1)
            else:
                weight = 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    API_TOKEN = "AAALabJgZ5IUr7K01pJQnD1J8RPjGquk"
    INDEX_CSV = "data/chexpert_subset/valid_index.csv"  # start with valid (234 images)

    print("=" * 60)
    print("Testing StreamingCheXpertDataset (valid set, 234 images)")
    print("=" * 60)

    ds = StreamingCheXpertDataset(
        index_csv=INDEX_CSV,
        api_token=API_TOKEN,
        table_ref="PNG_valid:41v9",
        cache_dir="data/chexpert_cache",
        cache_size_gb=1.0,
        frontal_only=True,
    )

    print(f"\nDataset size: {len(ds)}")
    print(f"\nFetching first 3 images...")
    for i in range(3):
        img, labels = ds[i]
        pos_labels = [StreamingCheXpertDataset.LABELS[j] for j, v in enumerate(labels) if v == 1.0]
        print(f"  [{i}] img={img.shape}  positive: {pos_labels if pos_labels else ['No Finding']}")

    print("\n✅ Streaming dataset works!")
    print(f"Cache: {ds.cache.num_files} files, {ds.cache.size_gb:.3f} GB")
