import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Reuse our chexpert_regex labeler for Stanford AIMI CE
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.labeling.clinical_labeler import ClinicalLabeler

class ClinicalMetrics:
    def __init__(self):
        self.labeler = ClinicalLabeler(backend="chexpert_regex")
        self.labels = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity', 'Lung Lesion', 
            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]
        
    def extract_labels(self, texts):
        arrays = []
        for t in texts:
            d = self.labeler.label(t)
            arrays.append([d.get(l, 0) for l in self.labels])
        return np.array(arrays)
        
    def compute(self, refs, cands):
        y_true = self.extract_labels(refs)
        y_pred = self.extract_labels(cands)
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        per_pathology = {}
        for i, l in enumerate(self.labels):
            per_pathology[l] = {"Precision": p_cls[i], "Recall": r_cls[i], "F1": f1_cls[i]}
            
        return {
            "Micro-F1": f1_micro,
            "Micro-Precision": p_micro,
            "Micro-Recall": r_micro,
            "Macro-F1": f1_macro,
            "Macro-Precision": p_macro,
            "Macro-Recall": r_macro
        }, per_pathology, y_true, y_pred


class RadGraphMetrics:
    def __init__(self):
        try:
            from radgraph import F1RadGraph
            self.rg = F1RadGraph(reward_level="partial")
            self.available = True
            print("RadGraph module loaded.")
        except ImportError:
            self.available = False
            print("Warning: 'radgraph' not found. RadGraph F1 will be 0.")
            
    def compute(self, refs, cands):
        if not self.available:
            return {"RadGraph-F1": 0.0}
        try:
            print(f"Computing RadGraph F1 for {len(cands)} samples (this usually takes 2-3 minutes per model)...")
            score, _, _, _ = self.rg(hyps=cands, refs=refs)
            return {"RadGraph-F1": score}
        except Exception as e:
            print(f"RadGraph Evaluation Error: {e}")
            return {"RadGraph-F1": 0.0}
