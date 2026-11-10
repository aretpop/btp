import numpy as np
import pandas as pd

class ErrorAnalyzer:
    def __init__(self, labels):
        self.labels = labels
        
    def analyze(self, y_true, y_pred):
        fp_counts = np.sum((y_pred == 1) & (y_true == 0), axis=0)
        fn_counts = np.sum((y_pred == 0) & (y_true == 1), axis=0)
        
        return pd.DataFrame({
            "Pathology": self.labels,
            "False_Positives": fp_counts,
            "False_Negatives": fn_counts
        })
