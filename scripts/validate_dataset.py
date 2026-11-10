"""
scripts/validate_dataset.py
-------------------------------------------------------------------------------
Phase 2 - Dataset Validation

Validates every record in each JSONL split:
  * Checks that 'response' field exists and is non-empty.
  * Checks that every image path resolves to an existing file.
  * Groups images by examination (same CXR folder = same exam).
  * Reports counts of valid/invalid records per split.
  * Does NOT silently discard anything - all problems are logged.

Run:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --split test
    python scripts/validate_dataset.py --split all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IUXRayDataset
from src.utils import load_config, setup_logging

SEP = "-" * 60
SEP_THICK = "=" * 60

logger = logging.getLogger(__name__)


def validate_split(cfg: dict, split: str) -> bool:
    """Validate one split. Returns True if all records are valid."""
    print(f"\n{SEP}")
    print(f"  Validating split: {split.upper()}")
    print(SEP)

    ds = IUXRayDataset.from_config(
        cfg,
        split=split,
        max_samples=None,
        require_all_images=True,
    )
    vr = ds.validation_report

    print(f"\n  Results for split '{split}':")
    print(f"    Total records     : {vr.total_records}")
    print(f"    Valid             : {vr.valid}")
    print(f"    Invalid           : {vr.invalid}")
    if vr.missing_response:
        print(f"    Missing response  : {len(vr.missing_response)} (lines: {vr.missing_response[:5]})")
    if vr.empty_response:
        print(f"    Empty response    : {len(vr.empty_response)} (lines: {vr.empty_response[:5]})")
    if vr.missing_images:
        print(f"    Missing images    : {len(vr.missing_images)}")
        for line_no, path in vr.missing_images[:5]:
            print(f"      line {line_no}: {path}")

    all_ok = vr.invalid == 0
    if all_ok:
        print(f"\n  [OK] Split '{split}' is fully valid.")
    else:
        print(f"\n  [!!] Split '{split}' has {vr.invalid} invalid record(s).")

    # ── Multi-image examination check ─────────────────────────────────────────
    print(f"\n  Multi-image examination breakdown:")
    multi_image = sum(1 for s in ds.samples if len(s.image_paths) > 1)
    single_image = len(ds.samples) - multi_image
    print(f"    Single-image exams : {single_image}")
    print(f"    Multi-image exams  : {multi_image}")
    if multi_image > 0:
        example = next(s for s in ds.samples if len(s.image_paths) > 1)
        print(f"    Example (multi)    : {example.sample_id}")
        for p in example.image_paths:
            print(f"      -> {p}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate IU X-Ray dataset splits."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to validate (default: all).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    setup_logging(level="WARNING")   # suppress INFO noise during validation
    cfg = load_config(args.config)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print("\n" + SEP_THICK)
    print("  CXR Baseline - Dataset Validation")
    print(SEP_THICK)

    all_passed = True
    for split in splits:
        passed = validate_split(cfg, split)
        all_passed = all_passed and passed

    print("\n" + SEP_THICK)
    if all_passed:
        print("  [OK] All splits passed validation.")
    else:
        print("  [!!] Some splits have issues - review warnings above.")
    print(SEP_THICK + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
