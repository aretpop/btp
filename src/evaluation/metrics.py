"""
src/evaluation/metrics.py
-------------------------------------------------------------------------------
Clinical Efficacy (CE) metric computation.

Given predicted and ground-truth binary label dicts, computes:
  - Per-finding: TP, FP, FN, TN, Precision, Recall, F1, Support
  - Micro-averaged: Precision, Recall, F1
  - Macro-averaged: Precision, Recall, F1

Uncertainty handling
--------------------
If uncertainty_policy == "ignore", labels with value -1 are excluded
from metric computation for that sample.

Label conventions (after policy application)
--------------------------------------------
  1  = positive (finding present)
  0  = negative (finding absent)
 -1  = uncertain (ignored in binary metric computation)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

POSITIVE  =  1
NEGATIVE  =  0
UNCERTAIN = -1


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample confusion accumulator
# ─────────────────────────────────────────────────────────────────────────────

class ConfusionAccumulator:
    """
    Accumulates TP/FP/FN/TN counts per finding across samples.

    Usage
    -----
    acc = ConfusionAccumulator(label_set)
    for pred_labels, ref_labels in zip(predictions, references):
        acc.update(pred_labels, ref_labels)
    results = acc.compute()
    """

    def __init__(self, label_set: List[str]):
        self.label_set = label_set
        self.tp: Dict[str, int] = {f: 0 for f in label_set}
        self.fp: Dict[str, int] = {f: 0 for f in label_set}
        self.fn: Dict[str, int] = {f: 0 for f in label_set}
        self.tn: Dict[str, int] = {f: 0 for f in label_set}
        self.n_samples: int = 0

    def update(
        self,
        pred: Dict[str, int],
        ref: Dict[str, int],
    ) -> None:
        """
        Update confusion matrix for a single sample.

        Parameters
        ----------
        pred : predicted label dict (after uncertainty policy)
        ref  : reference label dict (after same uncertainty policy)
        """
        self.n_samples += 1
        for finding in self.label_set:
            p = pred.get(finding, NEGATIVE)
            r = ref.get(finding, NEGATIVE)

            # Skip uncertain labels (when policy == "ignore")
            if p == UNCERTAIN or r == UNCERTAIN:
                continue

            if p == POSITIVE and r == POSITIVE:
                self.tp[finding] += 1
            elif p == POSITIVE and r == NEGATIVE:
                self.fp[finding] += 1
            elif p == NEGATIVE and r == POSITIVE:
                self.fn[finding] += 1
            else:  # both negative
                self.tn[finding] += 1

    def compute(self) -> Dict:
        """
        Compute all metrics.

        Returns
        -------
        dict with keys:
          "per_finding" : list of per-finding metric dicts
          "micro"       : dict with micro-averaged P/R/F1
          "macro"       : dict with macro-averaged P/R/F1
        """
        per_finding = []
        total_tp = total_fp = total_fn = 0

        macro_precision = []
        macro_recall = []
        macro_f1 = []

        for finding in self.label_set:
            tp = self.tp[finding]
            fp = self.fp[finding]
            fn = self.fn[finding]
            tn = self.tn[finding]
            support = tp + fn  # total actual positives

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            per_finding.append({
                "finding":   finding,
                "tp":        tp,
                "fp":        fp,
                "fn":        fn,
                "tn":        tn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "support":   support,
            })

            total_tp += tp
            total_fp += fp
            total_fn += fn

            macro_precision.append(precision)
            macro_recall.append(recall)
            macro_f1.append(f1)

        # Micro-averaged (sum TP/FP/FN globally)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        # Macro-averaged (mean of per-finding metrics)
        n = len(self.label_set)
        macro_p  = sum(macro_precision) / n if n > 0 else 0.0
        macro_r  = sum(macro_recall) / n if n > 0 else 0.0
        macro_f1_ = sum(macro_f1) / n if n > 0 else 0.0

        return {
            "per_finding": per_finding,
            "micro": {
                "precision": round(micro_p, 4),
                "recall":    round(micro_r, 4),
                "f1":        round(micro_f1, 4),
            },
            "macro": {
                "precision": round(macro_p, 4),
                "recall":    round(macro_r, 4),
                "f1":        round(macro_f1_, 4),
            },
            "n_samples": self.n_samples,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(results: Dict, model_name: str = "model") -> None:
    """Pretty-print metric results to stdout."""
    print(f"\n{'=' * 65}")
    print(f"  Clinical Efficacy Metrics — {model_name}")
    print(f"  Samples evaluated: {results['n_samples']}")
    print(f"{'=' * 65}")

    print(f"\n  {'Finding':<28} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-' * 62}")
    for row in results["per_finding"]:
        print(
            f"  {row['finding']:<28} "
            f"{row['precision']:>6.3f} "
            f"{row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} "
            f"{row['tp']:>5} "
            f"{row['fp']:>5} "
            f"{row['fn']:>5}"
        )

    print(f"\n  {'Micro-averaged':<28} "
          f"{results['micro']['precision']:>6.3f} "
          f"{results['micro']['recall']:>6.3f} "
          f"{results['micro']['f1']:>6.3f}")
    print(f"  {'Macro-averaged':<28} "
          f"{results['macro']['precision']:>6.3f} "
          f"{results['macro']['recall']:>6.3f} "
          f"{results['macro']['f1']:>6.3f}")
    print(f"{'=' * 65}\n")
