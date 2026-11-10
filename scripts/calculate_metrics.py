"""
scripts/calculate_metrics.py
-------------------------------------------------------------------------------
Phase 8 & 11 - Calculate clinical efficacy metrics.

Reads the label JSONL files and computes:
  * Overall Micro/Macro Precision, Recall, F1
  * Per-finding Precision, Recall, F1

Outputs summaries to stdout and saves structured CSVs to the metrics folder.

Run:
    python scripts/calculate_metrics.py --exp_dir experiments/baseline_...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import ConfusionAccumulator, print_metrics
from src.labeling.clinical_labeler import CHEXPERT_LABELS
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_metrics_for_file(
    labels_path: Path,
    label_set: list[str],
) -> dict:
    """Read a label JSONL file and compute metrics."""
    acc = ConfusionAccumulator(label_set)

    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                acc.update(pred=rec["generated_labels"], ref=rec["reference_labels"])
            except Exception as exc:
                logger.warning(f"Error parsing line in {labels_path.name}: {exc}")

    if acc.n_samples == 0:
        logger.warning(f"No valid samples found in {labels_path.name}")
        return {}

    return acc.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Clinical Efficacy metrics.")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory.")
    args = parser.parse_args()

    setup_logging(level="INFO")
    exp_dir = Path(args.exp_dir)
    labels_dir = exp_dir / "labels"
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*_labels.jsonl"))
    if not label_files:
        logger.error(f"No label files found in {labels_dir}")
        sys.exit(1)

    all_results = {}

    for lf in label_files:
        model_name = lf.name.replace("_labels.jsonl", "")
        logger.info(f"Processing metrics for model: {model_name}")
        
        results = calculate_metrics_for_file(lf, CHEXPERT_LABELS)
        if results:
            all_results[model_name] = results
            print_metrics(results, model_name=model_name)

    # Save overall summary to CSV
    overall_csv_path = metrics_dir / "overall_metrics.csv"
    with open(overall_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Metric_Type", "Precision", "Recall", "F1"])
        for model_name, res in all_results.items():
            for mtype in ["micro", "macro"]:
                m = res[mtype]
                writer.writerow([model_name, mtype, m["precision"], m["recall"], m["f1"]])
    logger.info(f"Saved {overall_csv_path.name}")

    # Save per-finding summary to CSV
    finding_csv_path = metrics_dir / "per_finding_metrics.csv"
    with open(finding_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Finding", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN", "Support"])
        for model_name, res in all_results.items():
            for row in res["per_finding"]:
                writer.writerow([
                    model_name, row["finding"], row["precision"], row["recall"],
                    row["f1"], row["tp"], row["fp"], row["fn"], row["tn"], row["support"]
                ])
    logger.info(f"Saved {finding_csv_path.name}")


if __name__ == "__main__":
    main()
