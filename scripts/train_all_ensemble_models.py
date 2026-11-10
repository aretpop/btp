"""
scripts/train_all_ensemble_models.py
====================================
Orchestrates the sequential training of the 3 EfficientNet-B0 candidate models:
  1. bce      : Standard Binary Cross-Entropy
  2. bce_wpos : BCE with Sqrt-smoothed positive weights
  3. asl      : Tuned Asymmetric Loss (gamma_neg=2.0, gamma_pos=0.0, clip=0.05)

Using:
  - Method 1: Tiered 40/40/20 Multi-Label Sampler
  - Stage 1: 3 epochs head warmup (LR = 1e-3)
  - Stage 2: 5 epochs full backbone fine-tuning (LR = 2e-5, Cosine Annealing)
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable

MODELS = ["bce", "bce_wpos", "asl"]

def run_model(loss_name, stage1_epochs=3, stage2_epochs=5, batches=600, workers=4):
    print("\n" + "=" * 90)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING ENSEMBLE MODEL: EfficientNet-B0 [{loss_name.upper()}]")
    print("=" * 90)
    
    cmd = [
        PYTHON_EXE,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_chexpert_ensemble_b0.py"),
        "--loss", loss_name,
        "--stage1-epochs", str(stage1_epochs),
        "--stage2-epochs", str(stage2_epochs),
        "--batches-per-epoch", str(batches),
        "--workers", str(workers)
    ]
    
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    
    if ret.returncode != 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Model {loss_name} failed with return code {ret.returncode} after {elapsed/60:.1f}m")
        return False
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] COMPLETED: Model {loss_name} in {elapsed/60:.1f}m")
        return True

def main():
    print(f"Starting sequential ensemble training pipeline for: {MODELS}")
    results = {}
    for loss in MODELS:
        success = run_model(loss, stage1_epochs=3, stage2_epochs=5, batches=600, workers=4)
        results[loss] = success
        if not success:
            print(f"Aborting sequence due to failure in {loss}.")
            break
            
    print("\n" + "=" * 90)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 90)
    for loss, st in results.items():
        print(f"  - EfficientNet-B0 [{loss}]: {'SUCCESS' if st else 'FAILED'}")

if __name__ == "__main__":
    main()

