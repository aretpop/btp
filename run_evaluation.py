import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from evaluator.nlp_metrics import NLPMetrics
from evaluator.clinical_metrics import ClinicalMetrics, RadGraphMetrics
from evaluator.error_analyzer import ErrorAnalyzer

def plot_radar(metrics_df, output_dir):
    # Normalize metrics to 0-1 for radar plot
    categories = ['BLEU-4', 'ROUGE-L', 'BERTScore-F1', 'Micro-F1', 'RadGraph-F1']
    
    # We only plot models as rows
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    for idx, row in metrics_df.iterrows():
        values = [row.get(cat, 0.0) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Comparison (Radar Chart)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"))
    plt.close()

def plot_bar(f1_df, output_dir):
    f1_df.plot(kind='bar', figsize=(14, 6))
    plt.title("Per-Pathology F1 Scores")
    plt.ylabel("F1 Score")
    plt.xlabel("Pathology")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pathology_f1_bar.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation pipeline")
    parser.add_argument("--input_file", type=str, required=True, help="CSV or JSON with columns: image_id, reference_report, generated_report_X")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save artifacts")
    parser.add_argument("--device", type=str, default="cuda", help="Device for BERTScore")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    else:
        df = pd.read_json(args.input_file)
        
    df.fillna("", inplace=True)
    
    reference = df['reference_report'].tolist()
    model_cols = [c for c in df.columns if c.startswith('generated_report_')]
    
    print(f"Found {len(model_cols)} models to evaluate: {model_cols}")
    
    nlp_evaluator = NLPMetrics(device=args.device)
    clinical_evaluator = ClinicalMetrics()
    radgraph_evaluator = RadGraphMetrics()
    error_analyzer = ErrorAnalyzer(clinical_evaluator.labels)
    
    all_metrics = []
    pathology_f1s = {}
    
    for col in model_cols:
        model_name = col.replace('generated_report_', '')
        print(f"\n--- Evaluating {model_name} ---")
        cands = df[col].tolist()
        
        # NLP Metrics
        nlp_res = nlp_evaluator.compute(reference, cands)
        
        # Clinical Metrics
        clin_res, per_path, y_true, y_pred = clinical_evaluator.compute(reference, cands)
        
        # Error Analysis
        err_df = error_analyzer.analyze(y_true, y_pred)
        err_df.to_csv(os.path.join(args.output_dir, f"{model_name}_errors.csv"), index=False)
        
        # RadGraph
        rg_res = radgraph_evaluator.compute(reference, cands)
        
        # Combine
        combined = {"Model": model_name}
        combined.update(nlp_res)
        combined.update(clin_res)
        combined.update(rg_res)
        
        all_metrics.append(combined)
        
        # Save F1 per pathology for bar chart
        pathology_f1s[model_name] = {k: v['F1'] for k, v in per_path.items()}
        
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    
    f1_df = pd.DataFrame(pathology_f1s)
    f1_df.to_csv(os.path.join(args.output_dir, "pathology_breakdown.csv"))
    
    print("Generating Plots...")
    plot_radar(metrics_df, args.output_dir)
    plot_bar(f1_df, args.output_dir)
    
    print(f"\nEvaluation Complete! Results saved in '{args.output_dir}'")
    print(metrics_df)

if __name__ == "__main__":
    main()
