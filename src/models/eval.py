from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)



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


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
        }
    
        return metrics


def get_feature_importance(model, X_train: pd.DataFrame) -> pd.Series:
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances = model.coef_
    else:
        raise ValueError('Model does not have feature importances')
    
    feature_importances = pd.Series(feature_importances, index=X_train.columns)
    feature_importances = feature_importances / feature_importances.sum()
    feature_importances = feature_importances.sort_values(ascending=False)
    
    return feature_importances