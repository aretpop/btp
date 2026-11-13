"""
src/data/labels.py
==================
Unified label conversion for CheXpert 14 classes across Train, Val, and Test.

Convention:
- raw == 1  -> y=1.0, certain_pos=1.0, u=0.0, blank=0.0, mask=1.0
- raw == 0  -> y=0.0, certain_pos=0.0, u=0.0, blank=0.0, mask=1.0
- raw == NaN-> y=0.0 (blank / true negative), blank=1.0, u=0.0, mask=1.0
- raw == -1 ->
    In TRAIN:
      y = uncertain_target[c] (default 0.5)
      u = 1.0
      mask = 1.0 (or 0.0 if uncertain_policy[c] == 'ignore')
    In EVAL:
      y = 0.0, u = 1.0, mask = 0.0 (uncertain cases are excluded from evaluation metrics)

'No Finding': Blank indicates presence of at least one finding, so blank -> 0 is
consistently correct; trained as a standard binary label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

CHEXPERT_14_LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def convert_labels(
    raw: Union[torch.Tensor, np.ndarray],
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert raw CheXpert labels (1.0, 0.0, -1.0, NaN) to standardized tensors.

    Args:
        raw: tensor or array of shape [B, C] (C=14)
        cfg: config dictionary containing `labels` section (or root config)
        mode: 'train' or 'eval'

    Returns:
        y:           [B, C] float32 target values in [0, 1]
        u:           [B, C] float32 indicator for uncertain labels (raw == -1)
        blank:       [B, C] float32 indicator for missing/unmentioned (raw was NaN)
        certain_pos: [B, C] float32 indicator for certain positive (raw == 1)
        mask:        [B, C] float32 binary mask (1 = include in loss/metric, 0 = ignore)
    """
    if cfg is None:
        cfg = {}

    labels_cfg = cfg.get("labels", cfg)
    uncertain_target_cfg = labels_cfg.get("uncertain_target", 0.5)
    uncertain_policy_cfg = labels_cfg.get("uncertain_policy", "soft")
    lam_u_cfg = labels_cfg.get("lam_u", 0.5)
    blank_weight_cfg = float(labels_cfg.get("blank_weight", 1.0))

    # Convert to torch tensor if numpy array
    if isinstance(raw, np.ndarray):
        raw_t = torch.from_numpy(raw).float()
    elif isinstance(raw, torch.Tensor):
        raw_t = raw.float()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(raw)}")

    B, C = raw_t.shape

    # Parse per-class parameters
    def _parse_param(val: Any, default: Any) -> List[Any]:
        if isinstance(val, (int, float, str)):
            return [val] * C
        if isinstance(val, (list, tuple)):
            if len(val) == C:
                return list(val)
            return list(val) + [default] * (C - len(val))
        if isinstance(val, dict):
            res = [default] * C
            for k, v in val.items():
                idx = int(k) if str(k).isdigit() else CHEXPERT_14_LABELS.index(k)
                if 0 <= idx < C:
                    res[idx] = v
            return res
        return [default] * C

    uncertain_target_list = [float(x) for x in _parse_param(uncertain_target_cfg, 0.5)]
    uncertain_policy_list = [str(x).lower() for x in _parse_param(uncertain_policy_cfg, "soft")]

    y = torch.zeros_like(raw_t)
    u = torch.zeros_like(raw_t)
    blank = torch.zeros_like(raw_t)
    certain_pos = torch.zeros_like(raw_t)
    mask = torch.ones_like(raw_t)

    is_pos = (raw_t == 1.0)
    is_neg = (raw_t == 0.0)
    is_unc = (raw_t == -1.0)
    is_nan = torch.isnan(raw_t)

    # 1. Positives (raw == 1)
    y[is_pos] = 1.0
    certain_pos[is_pos] = 1.0
    mask[is_pos] = 1.0

    # 2. Negatives (raw == 0)
    y[is_neg] = 0.0
    mask[is_neg] = 1.0

    # 3. Blank (raw == NaN) -> True negative (0.0)
    y[is_nan] = 0.0
    blank[is_nan] = 1.0
    mask[is_nan] = 1.0

    # 4. Uncertain (raw == -1)
    u[is_unc] = 1.0
    u_target_t = torch.tensor(uncertain_target_list, device=raw_t.device, dtype=raw_t.dtype).unsqueeze(0)

    if mode == "eval":
        # In EVAL: uncertain entries are EXCLUDED from metrics calculation
        mask[is_unc] = 0.0
        y[is_unc] = 0.0
    else:
        # In TRAIN: soft targets or ignore
        y[is_unc] = (torch.ones_like(raw_t) * u_target_t)[is_unc]
        for c in range(C):
            if uncertain_policy_list[c] == "ignore":
                cls_unc = is_unc[:, c]
                mask[cls_unc, c] = 0.0
                y[cls_unc, c] = 0.0

    return y, u, blank, certain_pos, mask

