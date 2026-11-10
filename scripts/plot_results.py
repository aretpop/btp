"""
scripts/plot_results.py
-------------------------------------------------------------------------------
Phase 14 - Generate publication-quality plots.

Generates:
1. Overall Clinical Efficacy (grouped bar chart)
2. Per-pathology F1 (grouped bar chart)
3. Per-pathology Heatmap
4. Precision vs Recall scatter plot

Saves PNG plots to the plots/ folder.

Run:
    python scripts/plot_results.py --exp_dir experiments/baseline_...
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def plot_overall_ce(overall_df: pd.DataFrame, out_dir: Path):
    """Plot 1: Grouped bar chart of Overall Micro metrics."""
    df = overall_df[overall_df["Metric_Type"] == "micro"].copy()
    if df.empty:
        return
        
    df_melt = df.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"], 
                      var_name="Metric", value_name="Score")
                      
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Overall Clinical Efficacy (Micro-Averaged)")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_ce.png", dpi=300)
    plt.close()


def plot_per_pathology_f1(finding_df: pd.DataFrame, out_dir: Path):
    """Plot 2: Grouped bar chart of F1 per finding."""
    plt.figure(figsize=(14, 6))
    sns.barplot(data=finding_df, x="Finding", y="F1", hue="Model", palette="viridis")
    plt.title("Per-Pathology F1 Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(out_dir / "per_finding_f1.png", dpi=300)
    plt.close()


def plot_heatmap(finding_df: pd.DataFrame, out_dir: Path):
    """Plot 3: Heatmap of F1 scores (Findings x Models)."""
    pivot = finding_df.pivot(index="Finding", columns="Model", values="F1")
    
    plt.figure(figsize=(6, 8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1.0, fmt=".2f", cbar_kws={'label': 'F1 Score'})
    plt.title("Clinical F1 Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "clinical_f1_heatmap.png", dpi=300)
    plt.close()


def plot_precision_recall(overall_df: pd.DataFrame, finding_df: pd.DataFrame, out_dir: Path):
    """Plot 4: Precision vs Recall scatter plot (Micro averages and findings)."""
    plt.figure(figsize=(8, 6))
    
    # Plot overall micro
    micro_df = overall_df[overall_df["Metric_Type"] == "micro"]
    sns.scatterplot(data=micro_df, x="Recall", y="Precision", hue="Model", 
                    s=150, marker="X", style="Model", palette="viridis", legend="brief")
    
    # Optionally plot the individual findings as smaller dots
    sns.scatterplot(data=finding_df, x="Recall", y="Precision", hue="Model", 
                    s=40, alpha=0.5, palette="viridis", legend=False)
                    
    plt.title("Precision vs Recall (X=Overall, dots=Per-Finding)")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for clinical efficacy.")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory.")
    args = parser.parse_args()

    setup_logging(level="INFO")
    exp_dir = Path(args.exp_dir)
    metrics_dir = exp_dir / "metrics"
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall_csv = metrics_dir / "overall_metrics.csv"
    finding_csv = metrics_dir / "per_finding_metrics.csv"

    if not overall_csv.exists() or not finding_csv.exists():
        logger.error(f"Metrics CSVs not found in {metrics_dir}. Run calculate_metrics.py first.")
        sys.exit(1)

    overall_df = pd.read_csv(overall_csv)
    finding_df = pd.read_csv(finding_csv)
    
    # Filter out Support=0 findings for cleaner plots if they drag down everything,
    # or keep them. We'll keep them but users can see 0.0 scores.
    
    logger.info("Generating plots...")
    plot_overall_ce(overall_df, plots_dir)
    plot_per_pathology_f1(finding_df, plots_dir)
    plot_heatmap(finding_df, plots_dir)
    plot_precision_recall(overall_df, finding_df, plots_dir)
    
    logger.info(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
