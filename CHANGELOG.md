# Changelog

All notable changes to ML-T2-002 will be documented here.
Format: [Semantic Versioning](https://semver.org/)

---

## [1.0.0] — 2024-09-04

### Added
- `src/data/preprocess.py` — Full preprocessing pipeline: load, clean, split normal/attack, variance filter, scaling
- `src/models/autoencoder.py` — Autoencoder + `AnomalyDetector` with fit/predict/score/save/load
- `src/models/baselines.py` — Isolation Forest and One-Class SVM baselines
- `src/utils/evaluate.py` — Metrics: F1, AUC-ROC, FPR, threshold sweep, best-F1 finder
- `src/utils/explain.py` — SHAP KernelExplainer for anomaly attribution
- `src/api/app.py` — FastAPI inference server with `/predict` and `/predict/batch` endpoints
- `train.py` — End-to-end CLI training script with baseline comparison flag
- `configs/config.yaml` — Centralized hyperparameter config
- `tests/` — Full pytest suite covering preprocessing, model, baselines, and metrics

### Security
- No API keys or secrets in codebase
- Model path configurable via environment variable `MODEL_PATH`

---

## [Unreleased]
- Adaptive threshold via Peaks Over Threshold (POT)
- Federated learning support
- Concept drift detection
- Kafka streaming inference pipeline
