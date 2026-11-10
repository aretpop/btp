"""
scripts/error_analysis.py
-------------------------------------------------------------------------------
Phase 15 - Error analysis.

Identifies and logs mismatched clinical findings (false positives, false negatives)
for manual review.

Output:
    experiments/<run>/error_analysis/clinical_errors.jsonl

Run:
    python scripts/error_analysis.py --exp_dir experiments/baseline_...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.clinical_labeler import POSITIVE, NEGATIVE, UNCERTAIN
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def run_error_analysis(
    labels_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> None:
    # Load raw predictions (to get the report text)
    preds = {}
    if predictions_path.exists():
        with open(predictions_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    preds[rec["sample_id"]] = rec
                except Exception:
                    pass

    n_errors_found = 0
    with (
        open(labels_path, "r", encoding="utf-8") as in_fh,
        open(output_path, "w", encoding="utf-8") as out_fh,
    ):
        for line in in_fh:
            try:
                lrec = json.loads(line)
            except Exception:
                continue

            sid = lrec["sample_id"]
            ref_labels = lrec["reference_labels"]
            gen_labels = lrec["generated_labels"]
            
            missed = []
            false_positives = []
            
            for finding, ref_val in ref_labels.items():
                gen_val = gen_labels.get(finding, NEGATIVE)
                
                if ref_val == UNCERTAIN or gen_val == UNCERTAIN:
                    continue
                    
                if ref_val == POSITIVE and gen_val == NEGATIVE:
                    missed.append(finding)
                elif ref_val == NEGATIVE and gen_val == POSITIVE:
                    false_positives.append(finding)

            if missed or false_positives:
                prec = preds.get(sid, {})
                error_rec = {
                    "sample_id": sid,
                    "model": lrec["model"],
                    "images": prec.get("images", []),
                    "reference_report": prec.get("reference_report", ""),
                    "generated_report": prec.get("generated_report", ""),
                    "missed_findings": missed,
                    "false_positives": false_positives,
                }
                out_fh.write(json.dumps(error_rec) + "\n")
                n_errors_found += 1

    logger.info(f"Model {labels_path.stem}: found {n_errors_found} samples with clinical errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis on evaluation results.")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory.")
    args = parser.parse_args()

    setup_logging(level="INFO")
    exp_dir = Path(args.exp_dir)
    labels_dir = exp_dir / "labels"
    preds_dir = exp_dir / "predictions"
    err_dir = exp_dir / "error_analysis"
    err_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        logger.error("Labels directory missing.")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*_labels.jsonl"))
    for lf in label_files:
        model_name = lf.name.replace("_labels.jsonl", "")
        pred_path = preds_dir / f"{model_name}.jsonl"
        out_path = err_dir / f"{model_name}_clinical_errors.jsonl"
        
        run_error_analysis(lf, pred_path, out_path)

    print(f"\nError analysis saved to: {err_dir}")


if __name__ == "__main__":
    main()
