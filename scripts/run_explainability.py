"""
scripts/run_explainability.py
==============================
CLI for generating attention overlays and comparing GradCAM vs Disease Attention.

Usage:
    python scripts/run_explainability.py \\
        --checkpoint checkpoints/chexpert_b1_v2/stageB_chexpert_b1_v2_baseline.pth \\
        --config configs/chexpert_b1_v2.yaml \\
        --split val \\
        --n_samples 50 \\
        --output_dir outputs/chexpert_b1_v2/explainability \\
        --compare_gradcam
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chexpert_b1_v2 import CheXpertB1V2Model
from src.explainability.gradcam import (
    GradCAM,
    DiseaseAttentionMap,
    compare_deletion_insertion,
    save_attention_overlays,
)
from scripts.train_chexpert_b1_v2 import (
    CheXpertLocalDatasetV2,
    make_patient_splits,
    CHEXPERT_14_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("run_explainability")


def main():
    parser = argparse.ArgumentParser(description="Explainability visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/chexpert_b1_v2.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to scan per class")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top TP and FP samples to save per class")
    parser.add_argument("--output_dir", default="outputs/chexpert_b1_v2/explainability")
    parser.add_argument("--compare_gradcam", action="store_true",
                        help="Compute deletion/insertion AUC comparison")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Only process these classes (default: all 14)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load model
    ckpt = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device)
    effective_model_cfg = {**model_cfg, **ckpt.get("cfg", {}).get("model", {})}
    model = CheXpertB1V2Model(effective_model_cfg)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device).eval()

    # Load dataset split
    df_all = pd.read_csv(PROJECT_ROOT / data_cfg.get("index_csv", "data/chexpert_subset/chexpert_materialized_index.csv"))
    train_df, val_df, test_df = make_patient_splits(
        df_all,
        val_frac=data_cfg.get("val_frac", 0.10),
        test_frac=data_cfg.get("test_frac", 0.10),
        seed=data_cfg.get("split_seed", 42),
    )
    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]

    ds_kwargs = {
        "uncertain_policy": data_cfg.get("uncertain_policy", "u_zeros"),
        "cache_dir": str(PROJECT_ROOT / data_cfg.get("cache_dir", "data/chexpert_cache")),
        "model_input_resolution": effective_model_cfg.get("input_resolution", 224),
    }
    dataset = CheXpertLocalDatasetV2(split_df, split="val", cfg=ds_kwargs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overlay visualization
    log.info("Generating attention overlays...")
    save_attention_overlays(
        model=model,
        dataset=dataset,
        output_dir=str(output_dir / "overlays"),
        labels=CHEXPERT_14_LABELS,
        top_k=args.top_k,
        classes=args.classes,
        device=str(device),
    )

    # GradCAM vs Disease Attention comparison
    if args.compare_gradcam:
        log.info("Computing deletion/insertion AUC comparison (GradCAM vs Disease Attention)...")

        gradcam = GradCAM(model, target_layer_name="backbone.features.8")
        dis_att = DiseaseAttentionMap(model)

        comparison_results = []
        n_per_class = max(1, args.n_samples // len(CHEXPERT_14_LABELS))
        classes_to_compare = args.classes or CHEXPERT_14_LABELS

        for cls_name in classes_to_compare:
            if cls_name not in CHEXPERT_14_LABELS:
                continue
            cls_idx = CHEXPERT_14_LABELS.index(cls_name)

            # Find samples with this class positive
            cls_samples = [
                i for i in range(min(500, len(dataset)))
                if dataset.df.iloc[i].get(cls_name, 0.0) == 1.0
            ][:n_per_class]

            if not cls_samples:
                log.warning("No positive samples for class: %s", cls_name)
                continue

            class_metrics = {
                "class": cls_name,
                "gradcam_del_auc": [], "gradcam_ins_auc": [],
                "disatt_del_auc": [], "disatt_ins_auc": [],
            }

            for idx in cls_samples:
                img_t, _, _ = dataset[idx]
                x = img_t.unsqueeze(0).to(device)
                try:
                    result = compare_deletion_insertion(
                        model, x, gradcam, dis_att, cls_idx, n_steps=10
                    )
                    for key in ["gradcam_del_auc", "gradcam_ins_auc", "disatt_del_auc", "disatt_ins_auc"]:
                        class_metrics[key].append(result[key])
                except Exception as e:
                    log.warning("Error processing sample %d for %s: %s", idx, cls_name, e)

            # Average
            avg_metrics = {
                "class": cls_name,
                "n_samples": len(cls_samples),
                "gradcam_del_auc_mean":  float(np.mean(class_metrics["gradcam_del_auc"])) if class_metrics["gradcam_del_auc"] else float("nan"),
                "gradcam_ins_auc_mean":  float(np.mean(class_metrics["gradcam_ins_auc"])) if class_metrics["gradcam_ins_auc"] else float("nan"),
                "disatt_del_auc_mean":   float(np.mean(class_metrics["disatt_del_auc"])) if class_metrics["disatt_del_auc"] else float("nan"),
                "disatt_ins_auc_mean":   float(np.mean(class_metrics["disatt_ins_auc"])) if class_metrics["disatt_ins_auc"] else float("nan"),
            }
            comparison_results.append(avg_metrics)

            log.info(
                "%-35s GC_Del=%.3f GC_Ins=%.3f DA_Del=%.3f DA_Ins=%.3f",
                cls_name,
                avg_metrics["gradcam_del_auc_mean"],
                avg_metrics["gradcam_ins_auc_mean"],
                avg_metrics["disatt_del_auc_mean"],
                avg_metrics["disatt_ins_auc_mean"],
            )

        # Print comparison table
        print(f"\n{'='*80}")
        print(f"{'Class':<35} {'GC Del↓':>8} {'GC Ins↑':>8} {'DA Del↓':>8} {'DA Ins↑':>8}")
        print("-" * 80)
        for r in comparison_results:
            print(
                f"{r['class']:<35} {r['gradcam_del_auc_mean']:>8.3f} "
                f"{r['gradcam_ins_auc_mean']:>8.3f} {r['disatt_del_auc_mean']:>8.3f} "
                f"{r['disatt_ins_auc_mean']:>8.3f}"
            )

        # Save JSON
        comp_path = output_dir / "gradcam_vs_disatt_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        log.info("Saved comparison: %s", comp_path)

        gradcam.cleanup()
        dis_att.cleanup()

    log.info("Explainability done. Output: %s", output_dir)


if __name__ == "__main__":
    main()

