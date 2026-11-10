"""
scripts/inspect_sample.py
-------------------------------------------------------------------------------
Phase 3 - Load and Display One Sample

Loads a single IU X-Ray record from the test split and prints:
  * Sample ID
  * Image paths (verified to exist)
  * Image dimensions
  * Reference radiology report
  * First 200 characters of the query

Run:
    python scripts/inspect_sample.py
    python scripts/inspect_sample.py --index 3
    python scripts/inspect_sample.py --sample_id CXR3030_IM-1405
    python scripts/inspect_sample.py --show          # saves sample image plot
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IUXRayDataset
from src.utils import load_config, setup_logging

SEP = "-" * 70
SEP_THICK = "=" * 70


def display_sample(sample, save_plot: bool = False, exp_dir: Path = None) -> None:
    """Pretty-print a CXRSample and optionally save its images as PNG."""

    print("\n" + SEP_THICK)
    print("  IU X-Ray Sample Inspection")
    print(SEP_THICK)

    print(f"\n  Sample ID   : {sample.sample_id}")
    print(f"  Num images  : {len(sample.image_paths)}")

    for i, img_path in enumerate(sample.image_paths):
        exists = "[OK]" if img_path.exists() else "[MISSING]"
        print(f"\n  Image {i}     : {exists}")
        print(f"    Path      : {img_path}")

        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            print(f"    Size      : {img.size[0]} x {img.size[1]} px")
            print(f"    Mode      : {img.mode}")
            file_size_kb = img_path.stat().st_size / 1024
            print(f"    File size : {file_size_kb:.1f} KB")
        except ImportError:
            print("    (Pillow not installed - cannot read image metadata)")
        except Exception as exc:
            print(f"    (Error reading image: {exc})")

    print(f"\n  Query (first 250 chars):")
    print(f"    {sample.query[:250]!r}")

    print(f"\n  Reference Report:")
    print(SEP)
    wrapped = textwrap.fill(sample.reference_report, width=68,
                            initial_indent="  ", subsequent_indent="  ")
    print(wrapped)
    print(SEP)
    print(f"\n  Word count  : {len(sample.reference_report.split())}")
    print(f"  Char count  : {len(sample.reference_report)}")
    print(SEP_THICK + "\n")

    if save_plot:
        _save_sample_plot(sample, exp_dir)


def _save_sample_plot(sample, exp_dir: Path = None) -> None:
    """Save sample images as a combined PNG figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        n = len(sample.image_paths)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Sample: {sample.sample_id}", fontsize=13, fontweight="bold")

        for ax, img_path in zip(axes, sample.image_paths):
            img = PILImage.open(img_path).convert("L")  # grayscale
            ax.imshow(img, cmap="gray", aspect="equal")
            ax.set_title(img_path.name, fontsize=10)
            ax.axis("off")

        # Add report text below
        report_short = sample.reference_report[:300]
        if len(sample.reference_report) > 300:
            report_short += "..."
        wrapped_report = textwrap.fill(report_short, width=90)
        fig.text(0.5, 0.01, wrapped_report, ha="center", fontsize=8,
                 style="italic", color="#333333",
                 bbox=dict(facecolor="lightyellow", alpha=0.7, edgecolor="gray"))

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        if exp_dir is None:
            exp_dir = PROJECT_ROOT / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        out_path = exp_dir / f"sample_inspection_{sample.sample_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {out_path}")

    except ImportError as exc:
        print(f"  Cannot create plot: {exc}")
    except Exception as exc:
        print(f"  Plot error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a single IU X-Ray test sample."
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Zero-based index of the sample to display (default: 0)."
    )
    parser.add_argument(
        "--sample_id", type=str, default=None,
        help="Specific sample ID to look up (overrides --index)."
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to load from (default: test)."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Save a PNG image of the X-ray images to experiments/."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        help="Path to YAML config."
    )
    args = parser.parse_args()

    setup_logging(level="WARNING")
    cfg = load_config(args.config)

    print(f"\nLoading split '{args.split}'...")
    ds = IUXRayDataset.from_config(cfg, split=args.split, max_samples=None)

    if not ds.samples:
        print("ERROR: No valid samples found in the split.")
        sys.exit(1)

    # Find sample
    if args.sample_id:
        matches = [s for s in ds.samples if s.sample_id == args.sample_id]
        if not matches:
            print(f"ERROR: Sample ID '{args.sample_id}' not found in '{args.split}'.")
            print(f"  Available IDs (first 5): {[s.sample_id for s in ds.samples[:5]]}")
            sys.exit(1)
        sample = matches[0]
    else:
        idx = args.index
        if idx >= len(ds.samples):
            print(f"ERROR: Index {idx} out of range (split has {len(ds.samples)} samples).")
            sys.exit(1)
        sample = ds.samples[idx]

    display_sample(sample, save_plot=args.show)
    print(f"  Total samples in '{args.split}' split: {len(ds.samples)}")


if __name__ == "__main__":
    main()
