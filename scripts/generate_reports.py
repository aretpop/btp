"""
scripts/generate_reports.py
-------------------------------------------------------------------------------
Phase 4-5 / 9-10 - Run model inference and save generated reports.

For each enabled model and each test sample:
  1. Load images from disk (PIL).
  2. Pass images + prompt to the model.
  3. Save: sample_id, model_name, image_paths, prompt, generated_report,
           reference_report, generation_time_seconds.

Output location:
    experiments/<run_name>_<timestamp>/predictions/<model_name>.jsonl

Run smoke test (8 samples):
    python scripts/generate_reports.py --smoke

Run full test set:
    python scripts/generate_reports.py

Run specific model only:
    python scripts/generate_reports.py --model moondream2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IUXRayDataset
from src.models.factory import build_model
from src.utils import load_config, setup_logging, get_device, set_seed, make_experiment_dir

logger = logging.getLogger(__name__)


def load_pil_images(sample):
    """Open all images for a sample as RGB PIL Images."""
    from PIL import Image
    images = []
    for p in sample.image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as exc:
            logger.error(f"Failed to open image {p}: {exc}")
    return images


def run_model_inference(
    model,
    dataset,
    prompt: str,
    gen_cfg: dict,
    output_path: Path,
    resume: bool = True,
) -> int:
    """
    Run inference for one model over a dataset split.

    Returns the number of samples processed.
    """
    # Load already-completed sample IDs to support resume
    done_ids = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["sample_id"])
                except Exception:
                    pass
        if done_ids:
            logger.info(f"Resuming: {len(done_ids)} samples already done.")

    n_processed = 0
    n_skipped = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(dataset.samples, desc=model.name, unit="sample")
    except ImportError:
        iterator = dataset.samples

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for sample in iterator:
            if sample.sample_id in done_ids:
                n_skipped += 1
                continue

            images = load_pil_images(sample)
            if not images:
                logger.error(f"No valid images for {sample.sample_id}; skipping.")
                continue

            t0 = time.perf_counter()
            try:
                generated = model.generate(
                    images=images,
                    prompt=prompt,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                    temperature=gen_cfg.get("temperature", 0.0),
                    do_sample=gen_cfg.get("do_sample", False),
                )
            except Exception as exc:
                logger.error(f"Generation failed for {sample.sample_id}: {exc}")
                generated = f"<GENERATION_ERROR: {exc}>"
            elapsed = time.perf_counter() - t0

            record = {
                "sample_id":              sample.sample_id,
                "model":                  model.name,
                "images":                 [str(p) for p in sample.image_paths],
                "prompt":                 prompt,
                "generated_report":       generated,
                "reference_report":       sample.reference_report,
                "generation_time_seconds": round(elapsed, 3),
            }
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            n_processed += 1

            logger.debug(
                f"[{model.name}] {sample.sample_id} | {elapsed:.1f}s | "
                f"gen={generated[:80]!r}"
            )

    logger.info(
        f"[{model.name}] Done: {n_processed} new samples "
        f"({n_skipped} skipped / already done)."
    )
    return n_processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chest X-ray reports using baseline models."
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run smoke test on first N samples (N from config smoke_test_n)."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only this model (by name). Default: all enabled models."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        help="Path to YAML config."
    )
    parser.add_argument(
        "--exp_dir", type=str, default=None,
        help="Explicit experiment directory (overrides auto-timestamping)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"))
    set_seed(cfg.get("seed", 42))
    device = get_device()

    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'SMOKE TEST' if args.smoke else 'FULL EVALUATION'}")

    # ── Experiment directory ──────────────────────────────────────────────────
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        for sub in ["predictions", "labels", "metrics", "plots", "logs", "error_analysis"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = make_experiment_dir(cfg)
    logger.info(f"Experiment directory: {exp_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    max_samples = cfg["dataset"].get("smoke_test_n", 8) if args.smoke else None
    split = cfg["dataset"].get("eval_split", "test")

    dataset = IUXRayDataset.from_config(cfg, split=split, max_samples=max_samples)
    logger.info(f"Loaded {len(dataset)} samples from '{split}'.")

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt = cfg["prompt"]["text"].strip()
    logger.info(f"Prompt: {prompt[:100]!r}")

    # ── Models ────────────────────────────────────────────────────────────────
    gen_cfg = cfg.get("generation", {})
    model_cfgs = cfg.get("models", [])

    if args.model:
        model_cfgs = [m for m in model_cfgs if m["name"] == args.model]
        if not model_cfgs:
            logger.error(f"No model named '{args.model}' found in config.")
            sys.exit(1)

    for model_cfg in model_cfgs:
        model_name = model_cfg["name"]
        enabled = model_cfg.get("enabled", True)
        if not enabled:
            logger.info(f"Skipping disabled model: {model_name}")
            logger.info(f"  Reason: {model_cfg.get('notes', 'disabled in config')}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"{'=' * 60}")

        output_path = exp_dir / "predictions" / f"{model_name}.jsonl"

        try:
            model = build_model(model_cfg, device=device)
            if model is None:
                continue

            logger.info(f"Loading {model_name} ...")
            model.load()

            run_model_inference(
                model=model,
                dataset=dataset,
                prompt=prompt,
                gen_cfg=gen_cfg,
                output_path=output_path,
                resume=True,
            )

        except Exception as exc:
            logger.error(f"Model '{model_name}' FAILED: {exc}", exc_info=True)
            logger.error(
                f"  Model '{model_name}' marked as NOT RUN. "
                f"Results will be absent from predictions/{model_name}.jsonl"
            )
            continue

        finally:
            try:
                model.unload()
            except Exception:
                pass

    logger.info(f"\nAll done. Results saved to: {exp_dir / 'predictions'}")
    print(f"\nExperiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
