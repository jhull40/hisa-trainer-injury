from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def evaluate_classification(y_true: pd.Series, y_pred: pd.Series, y_pred_soft_labels: pd.Series) -> Dict:

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_soft_labels)
    aucpr = auc(recall, precision)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]


    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_pred_soft_labels),
        'auprc': aucpr,
        'threshold': optimal_threshold,
    }

    return metrics
