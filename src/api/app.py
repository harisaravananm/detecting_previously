"""
app.py -- FastAPI inference endpoint for real-time anomaly detection.
Serves the web dashboard, metrics, and prediction endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import torch
import json
import os
import logging

from src.models.autoencoder import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
detector: AnomalyDetector = None
feature_names: list = []
all_metrics: dict = {}
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
META_PATH = os.getenv("META_PATH", "artifacts/meta.json")
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and metrics on startup."""
    global detector, feature_names, all_metrics
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        feature_names = meta["features"]
        detector = AnomalyDetector(input_dim=len(feature_names))
        detector.load(MODEL_PATH)
        logger.info("Model loaded. Input dim: %d", len(feature_names))
    else:
        logger.warning("Model not found at %s. Train first.", MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            all_metrics = json.load(f)
        logger.info("Metrics loaded from %s", METRICS_PATH)

    yield  # server runs here
    # Cleanup (none needed)


app = FastAPI(
    title="ML-T2-002 Network Anomaly Detector",
    description="Detects previously unseen network attacks using Autoencoder-based anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files (dashboard)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    features: list[float]

    model_config = {
        "json_schema_extra": {"example": {"features": [0.1] * 10}}
    }


class BatchFlowFeatures(BaseModel):
    flows: list[list[float]]


class PredictionResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the web dashboard."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Dashboard not found. Place index.html in src/api/static/"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/features")
def get_features():
    return {"features": feature_names, "count": len(feature_names)}


@app.get("/metrics")
def get_metrics():
    """Return all saved training/evaluation metrics."""
    if not all_metrics:
        raise HTTPException(status_code=404, detail="No metrics found. Run training first.")
    return all_metrics


@app.post("/predict", response_model=PredictionResult)
def predict(flow: FlowFeatures):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    if len(flow.features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail="Expected %d features, got %d" % (len(feature_names), len(flow.features))
        )
    X = np.array([flow.features], dtype=np.float32)
    score = float(detector.score(X)[0])
    pred = bool(detector.predict(X)[0])
    return PredictionResult(
        is_anomaly=pred,
        anomaly_score=round(score, 6),
        threshold=round(detector.threshold, 6),
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchFlowFeatures):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    X = np.array(batch.flows, dtype=np.float32)
    scores = detector.score(X).tolist()
    preds = detector.predict(X).tolist()
    return {
        "results": [
            {"is_anomaly": bool(p), "anomaly_score": round(s, 6)}
            for p, s in zip(preds, scores)
        ],
        "threshold": round(detector.threshold, 6),
    }
