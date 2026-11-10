"""
src/utils.py
-------------------------------------------------------------------------------
Shared utility functions used across the project.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path) -> dict:
    """Load a YAML configuration file and return as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logger with console (and optional file) handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Device selection ───────────────────────────────────────────────────────────

def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Experiment directory ───────────────────────────────────────────────────────

def make_experiment_dir(cfg: dict, timestamp: Optional[str] = None) -> Path:
    """
    Create a timestamped experiment directory and return its Path.

    Structure:
        experiments/baseline_2026_08_16_120000/
            predictions/
            labels/
            metrics/
            plots/
            logs/
            error_analysis/
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    project_root = Path(__file__).parent.parent
    base = project_root / cfg["output"]["experiments_dir"]
    run_name = cfg["output"]["run_name"]
    exp_dir = base / f"{run_name}_{timestamp}"

    for sub in ["predictions", "labels", "metrics", "plots", "logs", "error_analysis"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    return exp_dir
