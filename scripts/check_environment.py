"""
scripts/check_environment.py
-------------------------------------------------------------------------------
Phase 1 - Environment Diagnostics

Prints a full hardware and software report needed to decide which
models can realistically be loaded on this machine.

Run:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Force UTF-8 output on Windows to avoid cp1252 issues
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7

SEP = "-" * 60
SEP_THICK = "=" * 60


def _try_import(module: str) -> tuple:
    """Return (available, version_string)."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "NOT INSTALLED"


def _nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "nvidia-smi returned non-zero exit code"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nvidia-smi not found (no NVIDIA driver or not in PATH)"


def _bytes_to_gb(b: int) -> float:
    return round(b / 1024 ** 3, 2)


def main() -> None:
    print(SEP_THICK)
    print("  CXR Baseline - Environment Diagnostics")
    print(SEP_THICK)

    # ── OS & Python ───────────────────────────────────────────────────────────
    print("\n[1] System")
    print(SEP)
    print(f"  OS              : {platform.system()} {platform.version()}")
    print(f"  Machine         : {platform.machine()}")
    print(f"  Processor       : {platform.processor()}")
    print(f"  Python          : {sys.version}")
    print(f"  Python exec     : {sys.executable}")

    # ── RAM ───────────────────────────────────────────────────────────────────
    print("\n[2] Memory")
    print(SEP)
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  Total RAM       : {_bytes_to_gb(vm.total)} GB")
        print(f"  Available RAM   : {_bytes_to_gb(vm.available)} GB")
        print(f"  Used RAM        : {_bytes_to_gb(vm.used)} GB  ({vm.percent:.1f}%)")
    except ImportError:
        print("  psutil not installed - install with: pip install psutil")

    # ── Disk ──────────────────────────────────────────────────────────────────
    print("\n[3] Disk")
    print(SEP)
    for drive in ["c:\\", "d:\\"]:
        if Path(drive).exists():
            total, used, free = shutil.disk_usage(drive)
            print(f"  Drive {drive:<8}: total={_bytes_to_gb(total)} GB  "
                  f"used={_bytes_to_gb(used)} GB  free={_bytes_to_gb(free)} GB")

    # ── GPU ───────────────────────────────────────────────────────────────────
    print("\n[4] GPU")
    print(SEP)
    cuda_ok, torch_ver = _try_import("torch")
    if cuda_ok:
        import torch
        print(f"  PyTorch         : {torch_ver}")
        if torch.cuda.is_available():
            print(f"  CUDA available  : YES")
            print(f"  CUDA version    : {torch.version.cuda}")
            print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = _bytes_to_gb(props.total_memory)
                print(f"  GPU {i}           : {props.name}  |  VRAM: {vram_gb} GB")
        else:
            print("  CUDA available  : NO  (CPU-only PyTorch or no NVIDIA GPU)")
            print(f"  nvidia-smi      : {_nvidia_smi()}")
    else:
        print("  PyTorch         : NOT INSTALLED")
        print(f"  nvidia-smi      : {_nvidia_smi()}")

    # ── Key Python libraries ──────────────────────────────────────────────────
    print("\n[5] Key Libraries")
    print(SEP)
    libraries = [
        ("torch",           "PyTorch"),
        ("torchvision",     "TorchVision"),
        ("transformers",    "Transformers (HuggingFace)"),
        ("accelerate",      "Accelerate"),
        ("PIL",             "Pillow"),
        ("numpy",           "NumPy"),
        ("pandas",          "Pandas"),
        ("sklearn",         "scikit-learn"),
        ("matplotlib",      "Matplotlib"),
        ("seaborn",         "Seaborn"),
        ("yaml",            "PyYAML"),
        ("tqdm",            "tqdm"),
        ("psutil",          "psutil"),
        ("nltk",            "NLTK"),
        ("rouge_score",     "rouge-score"),
    ]
    for module, label in libraries:
        ok, ver = _try_import(module)
        status = f"[OK] {ver}" if ok else "[--] NOT INSTALLED"
        print(f"  {label:<35} {status}")

    # ── Model feasibility estimate ────────────────────────────────────────────
    print("\n[6] Model Feasibility Estimate")
    print(SEP)
    try:
        import psutil, torch
        ram_gb = psutil.virtual_memory().total / 1024 ** 3
        cuda = torch.cuda.is_available()
        vram_gb = 0.0
        if cuda:
            for i in range(torch.cuda.device_count()):
                vram_gb += torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
    except Exception:
        ram_gb, cuda, vram_gb = 0.0, False, 0.0

    models_est = [
        ("moondream2   (~2B params, ~4 GB RAM)",  "FEASIBLE  (CPU float32)"),
        ("Qwen2-VL-2B  (~2B params, ~4 GB RAM)",  "FEASIBLE  (CPU float32)"),
        ("LLaVA-Med-7B (~7B params, ~14 GB RAM)", "RISKY     (requires >=14 GB RAM on CPU; disabled by default)"),
    ]
    for model_label, verdict in models_est:
        print(f"  {model_label:<45} {verdict}")

    if not cuda:
        print(
            "\n  WARNING: No CUDA GPU detected. All inference will run on CPU.\n"
            "           Expect SLOW generation (minutes per sample).\n"
            "           Consider a CUDA GPU or Google Colab for faster runs."
        )

    # ── Dataset path check ────────────────────────────────────────────────────
    print("\n[7] Dataset Paths")
    print(SEP)
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / "IU-Xray"
    paths_to_check = [
        (dataset_root,                   "IU-Xray root"),
        (dataset_root / "train.jsonl",   "train.jsonl"),
        (dataset_root / "val.jsonl",     "val.jsonl"),
        (dataset_root / "test.jsonl",    "test.jsonl"),
        (dataset_root / "image",         "image/"),
    ]
    for p, label in paths_to_check:
        status = "[OK]    EXISTS" if p.exists() else "[MISS]  MISSING"
        print(f"  {label:<20}  {status}  =>  {p}")

    print("\n" + SEP_THICK)
    print("  Diagnostics complete.")
    print(SEP_THICK + "\n")


if __name__ == "__main__":
    main()
