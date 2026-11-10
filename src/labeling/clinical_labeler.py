"""
src/labeling/clinical_labeler.py
-------------------------------------------------------------------------------
Clinical report labeler for extracting structured findings from free-text
radiology reports.

Supported backends
------------------
1. chexpert_regex
   Rule-based NLP labeler based on the CheXpert labeling conventions.
   No model download required. Uses regex + negation detection.
   Label set: the 14 CheXpert/CheXbert conditions.

2. chexbert (NOT YET IMPLEMENTED)
   Deep-learning labeler using the CheXbert transformer model.
   Requires the CheXbert checkpoint (~400 MB).
   When implemented, will produce more accurate labels than regex.
   Currently returns NOT_AVAILABLE for all conditions.

Label values
------------
  1  = positive (finding present)
  0  = negative (finding absent)
 -1  = uncertain / possible (mention of the finding but hedged)
 -2  = not mentioned (NaN in original CheXpert paper)

Uncertainty policy
------------------
Configurable via 'uncertainty_policy' in baseline.yaml:
  "positive"  : map  -1  ->  1
  "negative"  : map  -1  ->  0
  "ignore"    : keep -1  as-is (exclude from metric calculation)

IMPORTANT: The SAME labeler and SAME policy is applied to both
generated and reference reports to ensure methodological consistency.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 14 CheXpert / CheXbert label set
# ---------------------------------------------------------------------------
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
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

# Label integer codes
POSITIVE   =  1
NEGATIVE   =  0
UNCERTAIN  = -1
NOT_MENTIONED = -2

# ---------------------------------------------------------------------------
# Simple rule-based CheXpert-style labeler
# ---------------------------------------------------------------------------
# Patterns map label -> list of keyword patterns
# Each pattern is a regex (case-insensitive)
_FINDING_PATTERNS: Dict[str, List[str]] = {
    "No Finding": [
        r"no acute",
        r"no significant",
        r"within normal limits",
        r"normal (chest|study|exam|radiograph|appearance|cardiomediastinal)",
        r"unremarkable",
        r"clear (lungs?|chest)",
        r"lungs? (are |is )?(clear|free of|without)",
        r"no (acute |significant )?(abnormality|abnormalities|findings?)",
        r"negative (chest|study)",
        r"no pathology",
    ],
    "Enlarged Cardiomediastinum": [
        r"enlarged (cardio)?mediastin",
        r"widened mediastin",
        r"mediastinal widening",
    ],
    "Cardiomegaly": [
        r"cardiomegaly",
        r"cardiac enlargement",
        r"enlarged heart",
        r"heart (is |appears? )?(enlarged|increased in size)",
        r"increased cardiac (size|silhouette)",
    ],
    "Lung Lesion": [
        r"lung (lesion|mass|nodule|opacity|lump)",
        r"pulmonary (nodule|mass|lesion)",
        r"nodule",
        r"mass",
        r"(solitary )?(pulmonary )?nodule",
    ],
    "Airspace Opacity": [
        r"airspace (opacity|opacities|disease|abnormality)",
        r"opacity",
        r"opacification",
        r"infiltrate",
        r"haziness",
    ],
    "Edema": [
        r"edema",
        r"pulmonary (vascular )?congestion",
        r"cephalization",
        r"fluid overload",
        r"congestive heart",
        r"vascular redistribution",
    ],
    "Consolidation": [
        r"consolidation",
        r"consolidat",
        r"airspace (consolidation|disease)",
        r"lobar (collapse|consolidation)",
        r"pneumonic consolidation",
    ],
    "Pneumonia": [
        r"pneumonia",
        r"pneumonic",
        r"infectious (process|etiology)",
        r"aspiration (pneumonia|pneumonitis)",
    ],
    "Atelectasis": [
        r"atelectasis",
        r"atelectat",
        r"subsegmental (collapse|atelectasis)",
        r"basilar (atelectasis|collapse)",
        r"plate.?like atelectasis",
        r"linear (opacity|atelectasis)",
        r"discoid atelectasis",
    ],
    "Pneumothorax": [
        r"pneumothorax",
        r"pneumo(thoraces|thorax)",
        r"collapsed lung",
        r"lung collapse",
    ],
    "Pleural Effusion": [
        r"pleural effusion",
        r"pleural fluid",
        r"effusion",
        r"blunting of (the )?(costophrenic|costo-phrenic) (angle|sulcus)",
        r"fluid in (the )?pleural",
    ],
    "Pleural Other": [
        r"pleural (thickening|plaque|scarring|fibrosis)",
        r"pleural (calcification|calcified)",
        r"fibrothorax",
    ],
    "Fracture": [
        r"fracture",
        r"rib fracture",
        r"clavicle fracture",
        r"vertebral (fracture|compression)",
        r"broken (rib|bone|clavicle)",
        r"acute (rib|bony) fracture",
    ],
    "Support Devices": [
        r"pacemaker",
        r"defibrillator",
        r"ICD",
        r"pacing (lead|wire|device)",
        r"central (venous )?line",
        r"CVP catheter",
        r"endotracheal (tube|intubation)",
        r"ET tube",
        r"nasogastric (tube|intubation)",
        r"NG tube",
        r"chest tube",
        r"PICC (line|catheter)",
        r"implanted (device|port|pacemaker)",
        r"tracheostomy",
    ],
}

# Negation prefixes (within ~6 words before the finding)
_NEGATION_PATTERNS = [
    r"no\b",
    r"not\b",
    r"without\b",
    r"absence of\b",
    r"absent\b",
    r"free of\b",
    r"negative for\b",
    r"deny\b",
    r"denies\b",
    r"none\b",
    r"ruled out\b",
    r"no evidence of\b",
    r"no sign of\b",
    r"no signs? of\b",
]

# Uncertainty/hedging prefixes
_UNCERTAINTY_PATTERNS = [
    r"possible\b",
    r"possibly\b",
    r"probable\b",
    r"probably\b",
    r"may\b",
    r"might\b",
    r"cannot (exclude|rule out)\b",
    r"cannot be excluded\b",
    r"questionable\b",
    r"equivocal\b",
    r"suspected\b",
    r"suspect\b",
    r"concern for\b",
    r"suggesting\b",
    r"consistent with\b",  # borderline — sometimes positive, sometimes hedged
]

_NEGATION_RE = re.compile(
    r"(" + "|".join(_NEGATION_PATTERNS) + r")",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"(" + "|".join(_UNCERTAINTY_PATTERNS) + r")",
    re.IGNORECASE,
)

# Word-window to look backwards for negation/uncertainty
_CONTEXT_WORDS = 8


def _label_finding_in_text(text: str, patterns: List[str]) -> int:
    """
    Return POSITIVE, NEGATIVE, UNCERTAIN, or NOT_MENTIONED for a finding
    in a free-text radiology report.

    Strategy:
    1. For each regex pattern for this finding, search the text.
    2. If found, check the preceding N words for negation or uncertainty.
    3. Return the most positive label found across all patterns.
    """
    text_lower = text.lower()
    best_label = NOT_MENTIONED

    for pat in patterns:
        for match in re.finditer(pat, text_lower, re.IGNORECASE):
            start = match.start()

            # Extract preceding context
            preceding = text_lower[max(0, start - 80): start]
            preceding_words = preceding.split()[-_CONTEXT_WORDS:]
            preceding_context = " ".join(preceding_words)

            if _NEGATION_RE.search(preceding_context):
                label = NEGATIVE
            elif _UNCERTAINTY_RE.search(preceding_context):
                label = UNCERTAIN
            else:
                label = POSITIVE

            # Keep the most informative label
            if label > best_label:
                best_label = label
                if best_label == POSITIVE:
                    return POSITIVE  # can't do better

    return best_label


def label_report(report: str) -> Dict[str, int]:
    """
    Apply the CheXpert-style regex labeler to a single report.

    Parameters
    ----------
    report : free-text radiology report string

    Returns
    -------
    dict mapping label_name -> int (POSITIVE=1, NEGATIVE=0, UNCERTAIN=-1,
                                    NOT_MENTIONED=-2)
    """
    labels = {}
    for finding in CHEXPERT_LABELS:
        patterns = _FINDING_PATTERNS.get(finding, [])
        labels[finding] = _label_finding_in_text(report, patterns)
    return labels


def apply_uncertainty_policy(labels: Dict[str, int], policy: str) -> Dict[str, int]:
    """
    Apply the uncertainty handling policy to a label dict.

    Parameters
    ----------
    labels : raw label dict from label_report()
    policy : "positive" | "negative" | "ignore"
    """
    if policy not in ("positive", "negative", "ignore"):
        raise ValueError(f"Unknown uncertainty_policy: {policy!r}. "
                         f"Choose from: positive, negative, ignore.")
    result = {}
    for finding, val in labels.items():
        if val == UNCERTAIN:
            if policy == "positive":
                result[finding] = POSITIVE
            elif policy == "negative":
                result[finding] = NEGATIVE
            else:  # ignore
                result[finding] = UNCERTAIN  # kept as-is; excluded in metrics
        elif val == NOT_MENTIONED:
            result[finding] = NEGATIVE   # treat not-mentioned as negative
        else:
            result[finding] = val
    return result


# ---------------------------------------------------------------------------
# High-level labeler class
# ---------------------------------------------------------------------------

class ClinicalLabeler:
    """
    Wrapper class providing a consistent interface for clinical label extraction.

    Applies the configured backend and uncertainty policy.
    Both generated and reference reports must be passed through this same
    instance to ensure methodological consistency.

    Usage
    -----
    >>> labeler = ClinicalLabeler.from_config(cfg)
    >>> ref_labels   = labeler.label("The heart is enlarged...")
    >>> gen_labels   = labeler.label("No acute cardiopulmonary process...")
    """

    def __init__(
        self,
        backend: str = "chexpert_regex",
        uncertainty_policy: str = "negative",
        chexbert_model_path: Optional[str] = None,
    ):
        if backend not in ("chexpert_regex", "chexbert"):
            raise ValueError(f"Unknown labeler backend: {backend!r}")
        if uncertainty_policy not in ("positive", "negative", "ignore"):
            raise ValueError(f"Unknown uncertainty_policy: {uncertainty_policy!r}")

        self.backend = backend
        self.uncertainty_policy = uncertainty_policy
        self.chexbert_model_path = chexbert_model_path
        self.label_set = CHEXPERT_LABELS

        if backend == "chexbert":
            logger.warning(
                "CheXbert backend requested but NOT YET IMPLEMENTED. "
                "Install CheXbert separately and wire it up in clinical_labeler.py. "
                "Falling back to chexpert_regex."
            )
            self.backend = "chexpert_regex"

        logger.info(
            f"ClinicalLabeler initialized: backend={self.backend!r}, "
            f"uncertainty_policy={self.uncertainty_policy!r}"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ClinicalLabeler":
        labeler_cfg = cfg.get("labeler", {})
        return cls(
            backend=labeler_cfg.get("backend", "chexpert_regex"),
            uncertainty_policy=labeler_cfg.get("uncertainty_policy", "negative"),
            chexbert_model_path=labeler_cfg.get("chexbert_model_path", None),
        )

    def label(self, report: str) -> Dict[str, int]:
        """
        Label a single report and apply the uncertainty policy.

        Parameters
        ----------
        report : free-text radiology report

        Returns
        -------
        dict: {finding_name: 0 or 1}  (or -1 if policy=="ignore")
        """
        if not report or not report.strip():
            logger.warning("Empty report passed to labeler; returning all negatives.")
            return {f: NEGATIVE for f in CHEXPERT_LABELS}

        if self.backend == "chexpert_regex":
            raw = label_report(report)
        else:
            # Placeholder for CheXbert
            raw = {f: NOT_MENTIONED for f in CHEXPERT_LABELS}

        return apply_uncertainty_policy(raw, self.uncertainty_policy)

    def label_batch(self, reports: List[str]) -> List[Dict[str, int]]:
        """Label a list of reports. Returns a list of label dicts."""
        return [self.label(r) for r in reports]
