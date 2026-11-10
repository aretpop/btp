"""
scripts/calculate_nlp_metrics.py
-------------------------------------------------------------------------------
Calculates standard NLP metrics (BLEU, ROUGE) for the generated reports.
While clinical efficacy is the primary metric, these are standard baseline metrics.

Outputs to metrics/nlp_metrics.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import setup_logging

logger = logging.getLogger(__name__)

def compute_metrics_for_file(predictions_path: Path):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    total_bleu1, total_bleu2, total_bleu3, total_bleu4 = 0.0, 0.0, 0.0, 0.0
    total_rouge1, total_rouge2, total_rougel = 0.0, 0.0, 0.0
    count = 0

    with open(predictions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                ref = rec.get("reference_report", "")
                gen = rec.get("generated_report", "")

                if not ref:
                    continue

                ref_tokens = nltk.word_tokenize(ref.lower())
                gen_tokens = nltk.word_tokenize(gen.lower())

                # BLEU
                b1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                b2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
                b3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
                b4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

                # ROUGE
                scores = scorer.score(ref, gen)

                total_bleu1 += b1
                total_bleu2 += b2
                total_bleu3 += b3
                total_bleu4 += b4
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                total_rougel += scores['rougeL'].fmeasure
                
                count += 1
            except Exception as e:
                pass

    if count == 0:
        return {}

    return {
        "BLEU-1": total_bleu1 / count,
        "BLEU-2": total_bleu2 / count,
        "BLEU-3": total_bleu3 / count,
        "BLEU-4": total_bleu4 / count,
        "ROUGE-1": total_rouge1 / count,
        "ROUGE-2": total_rouge2 / count,
        "ROUGE-L": total_rougel / count,
        "Samples": count
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()

    setup_logging(level="INFO")
    exp_dir = Path(args.exp_dir)
    preds_dir = exp_dir / "predictions"
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_csv = metrics_dir / "nlp_metrics.csv"
    
    results = []
    for pred_file in preds_dir.glob("*.jsonl"):
        model_name = pred_file.stem
        logger.info(f"Computing NLP metrics for {model_name}...")
        metrics = compute_metrics_for_file(pred_file)
        if metrics:
            metrics["Model"] = model_name
            results.append(metrics)
            print(f"--- {model_name} ---")
            for k, v in metrics.items():
                if k != "Model" and k != "Samples":
                    print(f"  {k}: {v:.4f}")

    if results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Samples", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        logger.info(f"Saved NLP metrics to {out_csv}")

if __name__ == "__main__":
    main()
