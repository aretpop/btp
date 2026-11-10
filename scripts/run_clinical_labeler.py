"""
scripts/run_clinical_labeler.py
-------------------------------------------------------------------------------
Phase 6-7 - Extract clinical labels from generated and reference reports.

Reads prediction JSONL files and outputs label JSONL files with the same
labeler applied to both generated and reference reports.

Output:
    experiments/<run>/labels/<model_name>_labels.jsonl

Each output record:
    {
        "sample_id": "...",
        "model": "...",
        "reference_labels": {"Atelectasis": 0, "Cardiomegaly": 1, ...},
        "generated_labels": {"Atelectasis": 1, "Cardiomegaly": 0, ...}
    }

Run:
    python scripts/run_clinical_labeler.py --exp_dir experiments/baseline_...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.clinical_labeler import ClinicalLabeler
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def label_predictions_file(
    predictions_path: Path,
    labeler: ClinicalLabeler,
    output_path: Path,
) -> int:
    """Label all records in a predictions JSONL. Returns n_labeled."""
    n = 0
    with (
        open(predictions_path, "r", encoding="utf-8") as in_fh,
        open(output_path, "w", encoding="utf-8") as out_fh,
    ):
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {predictions_path}")
                continue

            ref_labels = labeler.label(rec.get("reference_report", ""))
            gen_labels = labeler.label(rec.get("generated_report", ""))

            out_rec = {
                "sample_id":        rec["sample_id"],
                "model":            rec.get("model", "unknown"),
                "reference_labels": ref_labels,
                "generated_labels": gen_labels,
            }
            out_fh.write(json.dumps(out_rec) + "\n")
            n += 1

    logger.info(f"Labeled {n} records -> {output_path}")
    return n


def verify_labels_manually(labels_path: Path, n: int = 5) -> None:
    """Print n example label pairs for manual inspection."""
    print(f"\n--- Manual label verification ({labels_path.name}) ---")
    count = 0
    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if count >= n:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            print(f"\n  Sample: {rec['sample_id']}")
            print(f"  {'Finding':<28} {'Ref':>4} {'Gen':>4}")
            print(f"  {'-' * 40}")
            for finding in rec["reference_labels"]:
                r = rec["reference_labels"][finding]
                g = rec["generated_labels"].get(finding, 0)
                match = "OK" if r == g else "!!"
                print(f"  {finding:<28} {r:>4} {g:>4}  {match}")
            count += 1
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clinical labeler on prediction files."
    )
    parser.add_argument(
        "--exp_dir", type=str, required=True,
        help="Path to the experiment directory."
    )
    parser.add_argument(
        "--verify", type=int, default=3,
        help="Print N examples for manual verification (default: 3)."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        help="Path to YAML config."
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    cfg = load_config(args.config)

    exp_dir = Path(args.exp_dir)
    predictions_dir = exp_dir / "predictions"
    labels_dir = exp_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        sys.exit(1)

    pred_files = sorted(predictions_dir.glob("*.jsonl"))
    if not pred_files:
        logger.error(f"No .jsonl prediction files found in {predictions_dir}")
        sys.exit(1)

    labeler = ClinicalLabeler.from_config(cfg)
    logger.info(
        f"Labeler: backend={labeler.backend!r}, "
        f"uncertainty_policy={labeler.uncertainty_policy!r}"
    )
    logger.info(f"Label set ({len(labeler.label_set)} labels): {labeler.label_set}")

    for pred_path in pred_files:
        model_name = pred_path.stem
        out_path = labels_dir / f"{model_name}_labels.jsonl"
        logger.info(f"Labeling: {pred_path.name} -> {out_path.name}")
        n = label_predictions_file(pred_path, labeler, out_path)
        if n > 0 and args.verify > 0:
            verify_labels_manually(out_path, n=args.verify)

    print(f"\nLabels saved to: {labels_dir}")


if __name__ == "__main__":
    main()
