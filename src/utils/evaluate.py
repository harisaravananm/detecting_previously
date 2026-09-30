"""
evaluate.py — Metrics, threshold sweep, and reporting for anomaly detection.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import json
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray = None) -> dict:
    """Compute full evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'fpr': round(fpr, 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }
    if y_score is not None:
        metrics['auc_roc'] = round(roc_auc_score(y_true, y_score), 4)
    return metrics


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray,
                    n_steps: int = 100) -> list:
    """Sweep thresholds and return metrics at each point."""
    thresholds = np.linspace(y_score.min(), y_score.max(), n_steps)
    results = []
    for t in thresholds:
        y_pred = (y_score > t).astype(int)
        m = compute_metrics(y_true, y_pred)
        m['threshold'] = round(float(t), 6)
        results.append(m)
    return results


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Find threshold that maximizes F1."""
    sweep = threshold_sweep(y_true, y_score)
    best = max(sweep, key=lambda x: x['f1'])
    logger.info(f"Best F1={best['f1']} at threshold={best['threshold']}")
    return best['threshold'], best


def print_report(metrics: dict, model_name: str = 'Model'):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")
    print(f"{'='*50}\n")


def save_metrics(metrics: dict, path: str):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")
