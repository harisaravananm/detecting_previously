"""Tests for evaluation metrics."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.evaluate import compute_metrics, threshold_sweep, best_f1_threshold


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m['f1'] == 1.0
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['fpr'] == 0.0


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m['f1'] == 0.0
    assert m['recall'] == 0.0
    assert m['fpr'] == 1.0


def test_compute_metrics_with_auc():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, y_score)
    assert 'auc_roc' in m
    assert m['auc_roc'] == 1.0


def test_threshold_sweep_returns_list():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
    result = threshold_sweep(y_true, y_score, n_steps=10)
    assert isinstance(result, list)
    assert len(result) == 10
    assert all('threshold' in r for r in result)
    assert all('f1' in r for r in result)


def test_best_f1_threshold():
    np.random.seed(42)
    y_true = np.array([0]*50 + [1]*50)
    y_score = np.concatenate([np.random.rand(50)*0.4, np.random.rand(50)*0.4 + 0.6])
    threshold, best = best_f1_threshold(y_true, y_score)
    assert 0 < threshold < 1
    assert best['f1'] > 0.5


def test_fpr_zero_when_no_fp():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m['fpr'] == 0.0
    assert m['fp'] == 0
