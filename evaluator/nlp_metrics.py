import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import numpy as np
from tqdm import tqdm

class NLPMetrics:
    def __init__(self, device="cuda"):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        print("Loading ClinicalBERT for BERTScore...")
        self.bert_scorer = BERTScorer(model_type="emilyalsentzer/Bio_ClinicalBERT", num_layers=8, device=device)
        print("ClinicalBERT loaded.")

    def compute(self, refs, cands):
        bleu1, bleu2, bleu4 = [], [], []
        meteor = []
        rouge_l = []
        
        smooth = SmoothingFunction().method1
        
        print("Computing BLEU, ROUGE, METEOR...")
        for r, c in tqdm(zip(refs, cands), total=len(refs)):
            r_tokens = nltk.word_tokenize(r)
            c_tokens = nltk.word_tokenize(c)
            if len(c_tokens) == 0:
                c_tokens = [""]
            if len(r_tokens) == 0:
                r_tokens = [""]
            
            bleu1.append(sentence_bleu([r_tokens], c_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
            bleu2.append(sentence_bleu([r_tokens], c_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
            bleu4.append(sentence_bleu([r_tokens], c_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
            
            meteor.append(meteor_score([r_tokens], c_tokens))
            rouge_l.append(self.rouge.score(r, c)['rougeL'].fmeasure)
            
        print("Computing BERTScore...")
        # Truncate to 300 words to avoid BERT 512-token limit (especially for hallucinatory models)
        cands_trunc = [' '.join(c.split()[:300]) for c in cands]
        refs_trunc = [' '.join(r.split()[:300]) for r in refs]
        P, R, F1 = self.bert_scorer.score(cands_trunc, refs_trunc)
        
        return {
            "BLEU-1": np.mean(bleu1),
            "BLEU-2": np.mean(bleu2),
            "BLEU-4": np.mean(bleu4),
            "ROUGE-L": np.mean(rouge_l),
            "METEOR": np.mean(meteor),
            "BERTScore-F1": F1.mean().item()
        }
