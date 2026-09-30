# ML-T2-002 — Detecting Previously Unseen Network Attacks

> Unsupervised anomaly detection for zero-day network intrusions using Autoencoder reconstruction error.

**Domain:** Cybersecurity · **Type:** Anomaly Detection · **Track:** ML  

---

## Problem

Signature-based IDS fail on zero-day and novel attacks. This system learns normal traffic behavior and flags deviations — no labeled attack samples required.

## Architecture

```
Raw Traffic CSV
      │
      ▼
Preprocessing (normalize, variance filter, train/test split)
      │
      ▼
Autoencoder trained on NORMAL traffic only
      │
      ▼
Reconstruction Error per flow
      │
      ▼
Threshold (95th percentile of training errors)
      │
      ├── error > threshold → ANOMALY
      └── error ≤ threshold → NORMAL
      │
      ▼
SHAP explainability on flagged flows
      │
      ▼
FastAPI Inference Endpoint
```

## Project Structure

```
ml-t2-002/
├── src/
│   ├── data/
│   │   └── preprocess.py       # Load, clean, split, scale
│   ├── models/
│   │   ├── autoencoder.py      # Autoencoder + AnomalyDetector
│   │   └── baselines.py        # Isolation Forest, One-Class SVM
│   ├── utils/
│   │   ├── evaluate.py         # Metrics, threshold sweep
│   │   └── explain.py          # SHAP explainability
│   └── api/
│       └── app.py              # FastAPI inference server
├── tests/
│   ├── test_preprocess.py
│   ├── test_autoencoder.py
│   ├── test_baselines.py
│   └── test_evaluate.py
├── configs/
│   └── config.yaml
├── train.py                    # CLI training script
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Datasets

| Dataset | Source | Format |
|---------|--------|--------|
| CICIDS2017/2018 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | CSV |
| UNSW-NB15 | [ADFA](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | CSV |
| NSL-KDD | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | CSV |

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Train
python train.py \
  --train data/train.csv \
  --test data/test.csv \
  --dataset cicids \
  --epochs 50 \
  --baselines

# Serve
MODEL_PATH=artifacts/model.pt uvicorn src.api.app:app --reload

# Test
pytest tests/ -v --cov=src
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model load status |
| `/features` | GET | Feature list |
| `/predict` | POST | Single flow prediction |
| `/predict/batch` | POST | Batch prediction |

**Single flow example:**
```json
POST /predict
{"features": [0.12, 0.45, ...]}

→ {"is_anomaly": true, "anomaly_score": 0.087, "threshold": 0.021}
```

## Evaluation

| Metric | Target |
|--------|--------|
| AUC-ROC | ≥ 0.95 |
| F1-Score | ≥ 0.85 |
| False Positive Rate | ≤ 0.05 |

## References

1. Lanvin et al. (2023) — Errors in CICIDS2017. https://arxiv.org/abs/2305.01735  
2. Abdulhammed et al. (2021) — Deep Learning for IDS. IEEE Access.  
3. Yang et al. (2022) — MTH-IDS. IEEE IoT Journal.  
4. Andresini et al. (2021) — Autoencoder IDS. IEEE Access.  
5. Moustafa et al. (2023) — UNSW-NB15 revisited. https://arxiv.org/abs/2301.09183  

---

**Author:** Hari Saravanan M · IFET College of Engineering · B.Tech AI&ML 2023–2027
