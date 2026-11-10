"""
scripts/run_ablation_sweep.py
==============================
Runs all ablations defined in configs/ablation_sweep.yaml, one by one.
Applies each ablation's overrides on top of the base config and runs training.

Usage:
    python scripts/run_ablation_sweep.py --config configs/ablation_sweep.yaml \\
        --base_config configs/chexpert_b1_v2.yaml \\
        --smoke_test

    # Run only specific ablations
    python scripts/run_ablation_sweep.py --only baseline_old fuse_block5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run ablation sweep")
    parser.add_argument("--config", default="configs/ablation_sweep.yaml",
                        help="Path to ablation sweep config")
    parser.add_argument("--base_config", default="configs/chexpert_b1_v2.yaml",
                        help="Base training config")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Pass --smoke_test to each training run")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only specific ablation names")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable to use")
    args = parser.parse_args()

    # Load ablation configs
    with open(PROJECT_ROOT / args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    ablations = sweep_cfg.get("ablations", [])
    if not ablations:
        print("No ablations found in config.")
        return

    # Filter by --only
    if args.only:
        ablations = [a for a in ablations if a["name"] in args.only]
        print(f"Running {len(ablations)} ablation(s): {[a['name'] for a in ablations]}")

    results_summary = {}

    for abl in ablations:
        name    = abl["name"]
        overrides = abl.get("overrides", {})
        print(f"\n{'='*70}")
        print(f"ABLATION: {name}")
        print(f"Overrides: {overrides}")
        print(f"{'='*70}")

        # Build override args: key=value for each override
        # Also force experiment name to match ablation name
        override_list = [f"experiment.name={name}"] + [
            f"{k}={v}" for k, v in overrides.items()
        ]

        cmd = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "train_chexpert_b1_v2.py"),
            "--config", str(PROJECT_ROOT / args.base_config),
            "--override", *override_list,
        ]
        if args.smoke_test:
            cmd.append("--smoke_test")

        print(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
        results_summary[name] = {
            "overrides": overrides,
            "exit_code": result.returncode,
            "success": success,
        }

        if not success:
            print(f"\n[WARNING] Ablation '{name}' exited with code {result.returncode}")

    # Print summary
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    for name, info in results_summary.items():
        status = "✓ OK" if info["success"] else f"✗ FAILED (code={info['exit_code']})"
        print(f"  {name:<35} {status}")

    # Save summary JSON
    summary_path = PROJECT_ROOT / "outputs" / "ablation_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()

