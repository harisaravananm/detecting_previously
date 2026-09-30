"""Tests for baseline anomaly detection models."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.baselines import IsolationForestDetector, OneClassSVMDetector


@pytest.fixture
def normal_data():
    np.random.seed(42)
    return np.random.randn(200, 10).astype(np.float32)


@pytest.fixture
def anomaly_data():
    np.random.seed(1)
    return (np.random.randn(50, 10) * 5 + 10).astype(np.float32)


def test_isolation_forest_predict_shape(normal_data):
    det = IsolationForestDetector(contamination=0.05)
    det.fit(normal_data)
    preds = det.predict(normal_data)
    assert preds.shape == (200,)
    assert set(preds).issubset({0, 1})


def test_isolation_forest_score_shape(normal_data):
    det = IsolationForestDetector()
    det.fit(normal_data)
    scores = det.score(normal_data)
    assert scores.shape == (200,)


def test_isolation_forest_detects_anomalies(normal_data, anomaly_data):
    det = IsolationForestDetector(contamination=0.1)
    det.fit(normal_data)
    normal_scores = det.score(normal_data).mean()
    anomaly_scores = det.score(anomaly_data).mean()
    assert anomaly_scores > normal_scores


def test_ocsvm_predict_shape(normal_data):
    det = OneClassSVMDetector(nu=0.05)
    det.fit(normal_data)
    preds = det.predict(normal_data)
    assert preds.shape == (200,)
    assert set(preds).issubset({0, 1})


def test_ocsvm_detects_anomalies(normal_data, anomaly_data):
    det = OneClassSVMDetector(nu=0.1)
    det.fit(normal_data)
    normal_scores = det.score(normal_data).mean()
    anomaly_scores = det.score(anomaly_data).mean()
    assert anomaly_scores > normal_scores
