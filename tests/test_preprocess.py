"""Tests for data preprocessing module."""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import split_normal_attack, select_features, scale_features


@pytest.fixture
def sample_cicids_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Flow Duration': np.random.rand(n) * 1000,
        'Total Fwd Packets': np.random.randint(1, 100, n).astype(float),
        'Total Backward Packets': np.random.randint(1, 50, n).astype(float),
        'Flow Bytes/s': np.random.rand(n) * 1e6,
        'Fwd Packet Length Mean': np.random.rand(n) * 500,
        'Label': ['BENIGN'] * 150 + ['DoS Hulk'] * 50
    })
    return df


def test_split_normal_attack(sample_cicids_df):
    normal, attack = split_normal_attack(sample_cicids_df, 'cicids')
    assert len(normal) == 150
    assert len(attack) == 50
    assert all(normal['Label'] == 'BENIGN')
    assert all(attack['Label'] != 'BENIGN')


def test_split_empty_attack():
    df = pd.DataFrame({
        'Flow Duration': [1.0, 2.0],
        'Label': ['BENIGN', 'BENIGN']
    })
    normal, attack = split_normal_attack(df, 'cicids')
    assert len(attack) == 0


def test_select_features_drops_label(sample_cicids_df):
    normal, _ = split_normal_attack(sample_cicids_df, 'cicids')
    feat = select_features(normal, 'cicids')
    assert 'Label' not in feat.columns
    assert feat.shape[1] > 0


def test_select_features_drops_zero_variance(sample_cicids_df):
    normal, _ = split_normal_attack(sample_cicids_df, 'cicids')
    normal = normal.copy()
    normal['constant_col'] = 0.0
    feat = select_features(normal, 'cicids')
    assert 'constant_col' not in feat.columns


def test_scale_features_shape(sample_cicids_df):
    normal, _ = split_normal_attack(sample_cicids_df, 'cicids')
    feat = select_features(normal, 'cicids')
    half = len(feat) // 2
    train, test, scaler = scale_features(feat.iloc[:half], feat.iloc[half:])
    assert train.shape[1] == test.shape[1]
    assert scaler is not None


def test_scale_features_zero_mean():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    X_df = pd.DataFrame(X, columns=['a', 'b'])
    train_scaled, _, _ = scale_features(X_df, X_df)
    assert abs(train_scaled.mean()) < 1e-10
