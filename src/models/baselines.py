"""
baselines.py — Baseline unsupervised anomaly detectors for benchmarking.
Includes Isolation Forest and One-Class SVM.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray):
        logger.info("Fitting Isolation Forest...")
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal (sklearn uses -1/1)."""
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: higher = more anomalous."""
        return -self.model.score_samples(X)


class OneClassSVMDetector:
    def __init__(self, nu: float = 0.05, kernel: str = 'rbf', gamma: str = 'scale'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X: np.ndarray):
        logger.info("Fitting One-Class SVM (may be slow on large datasets)...")
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)
