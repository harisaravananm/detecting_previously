"""
explain.py — SHAP-based explainability for flagged anomalies.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def explain_anomalies(model, X_anomalies: np.ndarray, feature_names: list,
                      n_samples: int = 100, background_size: int = 50):
    """
    Generate SHAP values for flagged anomalous samples.
    Uses KernelExplainer (model-agnostic) for compatibility with autoencoder.
    """
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return None

    if len(X_anomalies) == 0:
        logger.warning("No anomalies to explain.")
        return None

    # Use a random background sample
    bg = X_anomalies[:background_size] if len(X_anomalies) >= background_size \
        else X_anomalies

    def reconstruction_score(X):
        """Wrapper: returns per-sample reconstruction error."""
        import torch
        tensor = torch.FloatTensor(X).to(model.device)
        return model.model.reconstruction_error(tensor).reshape(-1, 1)

    explainer = shap.KernelExplainer(reconstruction_score, bg)
    sample = X_anomalies[:min(n_samples, len(X_anomalies))]
    shap_values = explainer.shap_values(sample, nsamples=100)

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values[0]).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_shap), key=lambda x: -x[1])

    logger.info("Top features driving anomaly scores:")
    for feat, val in ranked[:10]:
        logger.info(f"  {feat:<40} {val:.6f}")

    return ranked


def top_features(shap_ranked: list, top_k: int = 10) -> list:
    return shap_ranked[:top_k]
