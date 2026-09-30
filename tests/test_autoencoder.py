"""Tests for Autoencoder anomaly detector."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.autoencoder import Autoencoder, AnomalyDetector
import torch


@pytest.fixture
def small_normal_data():
    np.random.seed(0)
    return np.random.randn(300, 20).astype(np.float32)


@pytest.fixture
def trained_detector(small_normal_data):
    det = AnomalyDetector(input_dim=20, epochs=5, batch_size=64)
    det.fit(small_normal_data, percentile=95.0)
    return det


def test_autoencoder_forward_shape():
    model = Autoencoder(input_dim=20)
    x = torch.randn(8, 20)
    out = model(x)
    assert out.shape == (8, 20)


def test_autoencoder_reconstruction_error_shape():
    model = Autoencoder(input_dim=20)
    x = torch.randn(10, 20)
    err = model.reconstruction_error(x)
    assert err.shape == (10,)
    assert all(e >= 0 for e in err)


def test_detector_fit_sets_threshold(trained_detector):
    assert trained_detector.threshold is not None
    assert trained_detector.threshold > 0


def test_detector_predict_returns_binary(trained_detector, small_normal_data):
    preds = trained_detector.predict(small_normal_data)
    assert set(preds).issubset({0, 1})
    assert preds.shape == (300,)


def test_detector_score_non_negative(trained_detector, small_normal_data):
    scores = trained_detector.score(small_normal_data)
    assert all(s >= 0 for s in scores)


def test_normal_data_mostly_normal(trained_detector, small_normal_data):
    preds = trained_detector.predict(small_normal_data)
    # ~95% of normal training data should be predicted normal
    normal_rate = (preds == 0).mean()
    assert normal_rate >= 0.90


def test_anomaly_higher_score(trained_detector):
    np.random.seed(1)
    normal = np.random.randn(50, 20).astype(np.float32)
    anomaly = (np.random.randn(50, 20) * 10 + 20).astype(np.float32)
    normal_scores = trained_detector.score(normal).mean()
    anomaly_scores = trained_detector.score(anomaly).mean()
    assert anomaly_scores > normal_scores


def test_detector_predict_before_fit_raises():
    det = AnomalyDetector(input_dim=10)
    with pytest.raises(RuntimeError):
        det.predict(np.random.randn(5, 10).astype(np.float32))


def test_detector_save_load(trained_detector, small_normal_data, tmp_path):
    path = str(tmp_path / "model.pt")
    trained_detector.save(path)

    loaded = AnomalyDetector(input_dim=20)
    loaded.load(path)

    orig_scores = trained_detector.score(small_normal_data)
    load_scores = loaded.score(small_normal_data)
    np.testing.assert_allclose(orig_scores, load_scores, rtol=1e-5)
