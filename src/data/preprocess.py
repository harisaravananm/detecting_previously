"""
preprocess.py — Data ingestion and preprocessing for network traffic anomaly detection.
Supports CICIDS2017/2018, UNSW-NB15, NSL-KDD datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known label columns per dataset
LABEL_COLUMNS = {
    'cicids': 'Label',
    'unsw': 'label',
    'nslkdd': 'label'
}

NORMAL_LABELS = {
    'cicids': ['BENIGN'],
    'unsw': [0],
    'nslkdd': ['normal']
}


def load_dataset(path: str, dataset_type: str = 'cicids') -> pd.DataFrame:
    """Load and do initial cleanup on a raw dataset CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")

    # Replace infinities with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_normal_attack(df: pd.DataFrame, dataset_type: str = 'cicids'):
    """Split dataframe into normal and attack subsets."""
    label_col = LABEL_COLUMNS[dataset_type]
    normal_labels = NORMAL_LABELS[dataset_type]

    normal_mask = df[label_col].isin(normal_labels)
    normal_df = df[normal_mask].copy()
    attack_df = df[~normal_mask].copy()

    logger.info(f"Normal: {len(normal_df)} | Attack: {len(attack_df)}")
    return normal_df, attack_df


def select_features(df: pd.DataFrame, dataset_type: str = 'cicids',
                    variance_threshold: float = 0.01) -> pd.DataFrame:
    """Drop label column and low-variance features."""
    label_col = LABEL_COLUMNS[dataset_type]
    feature_df = df.drop(columns=[label_col], errors='ignore')

    # Drop non-numeric
    feature_df = feature_df.select_dtypes(include=[np.number])

    # Variance filter
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(feature_df)
    selected_cols = feature_df.columns[selector.get_support()]
    feature_df = feature_df[selected_cols]

    logger.info(f"Features after selection: {feature_df.shape[1]}")
    return feature_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   scaler_type: str = 'standard'):
    """Fit scaler on train, transform both train and test."""
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, scaler


def build_pipeline(train_path: str, test_path: str, dataset_type: str = 'cicids'):
    """End-to-end preprocessing pipeline."""
    train_raw = load_dataset(train_path, dataset_type)
    test_raw = load_dataset(test_path, dataset_type)

    train_normal, _ = split_normal_attack(train_raw, dataset_type)
    test_normal, test_attack = split_normal_attack(test_raw, dataset_type)

    train_feat = select_features(train_normal, dataset_type)
    test_all = pd.concat([test_normal, test_attack])
    test_feat = select_features(test_all, dataset_type)

    # Align columns
    common_cols = train_feat.columns.intersection(test_feat.columns)
    train_feat = train_feat[common_cols]
    test_feat = test_feat[common_cols]

    X_train, X_test, scaler = scale_features(train_feat, test_feat)

    label_col = LABEL_COLUMNS[dataset_type]
    normal_labels = NORMAL_LABELS[dataset_type]
    y_test = (~test_all[label_col].isin(normal_labels)).astype(int).values

    return X_train, X_test, y_test, scaler, list(common_cols)
