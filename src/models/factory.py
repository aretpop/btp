"""
src/models/factory.py
-------------------------------------------------------------------------------
Model factory: instantiates the correct model class from a config dict entry.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseReportModel

logger = logging.getLogger(__name__)


def build_model(model_cfg: dict, device: str = "cpu") -> Optional[BaseReportModel]:
    """
    Instantiate a model from a config entry.

    Parameters
    ----------
    model_cfg : one entry from the `models` list in baseline.yaml
    device    : 'cuda' or 'cpu'

    Returns
    -------
    BaseReportModel instance, or None if the model is disabled.
    """
    name = model_cfg.get("name", "unknown")
    enabled = model_cfg.get("enabled", True)

    if not enabled:
        logger.info(f"Model '{name}' is DISABLED in config. Skipping.")
        return None

    checkpoint = model_cfg["checkpoint"]
    dtype = model_cfg.get("dtype", "float32")
    notes = model_cfg.get("notes", "")
    if notes:
        logger.info(f"Model '{name}' notes: {notes}")

    if name == "moondream2":
        from .moondream import Moondream2Model
        revision = model_cfg.get("revision", "2025-01-09")
        return Moondream2Model(checkpoint=checkpoint, revision=revision,
                               device=device, dtype=dtype)

    elif name == "qwen2-vl":
        from .qwen import Qwen2VLModel
        return Qwen2VLModel(checkpoint=checkpoint, device=device, dtype=dtype)

    elif name == "llava-med":
        from .llava_med import LLaVAMedModel
        return LLaVAMedModel(checkpoint=checkpoint, device=device, dtype=dtype)

    elif name == "paligemma2":
        from .paligemma import PaliGemma2Model
        return PaliGemma2Model(checkpoint=checkpoint, device=device, dtype=dtype)

    elif name == "florence2" or name == "florence2-base":
        from .florence2 import Florence2Model
        return Florence2Model(checkpoint=checkpoint, device=device, dtype=dtype)

    elif name == "internvl2" or name == "internvl2-2b":
        from .internvl2 import InternVL2Model
        return InternVL2Model(checkpoint=checkpoint, device=device, dtype=dtype)

    else:
        logger.error(
            f"Unknown model name: '{name}'. "
            f"Supported: moondream2, qwen2-vl, llava-med, paligemma2, florence2, internvl2."
        )
        raise ValueError(f"Unknown model: {name}")
