"""
autoencoder.py — Autoencoder for unsupervised anomaly detection on network traffic.
Reconstruction error is used as the anomaly score.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dims: list = None):
        super().__init__()
        if encoding_dims is None:
            encoding_dims = [64, 32, 16]

        # Encoder
        encoder_layers = []
        prev = input_dim
        for dim in encoding_dims:
            encoder_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU()]
            prev = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        for dim in reversed(encoding_dims[:-1]):
            decoder_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU()]
            prev = dim
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1).cpu().numpy()


class AnomalyDetector:
    def __init__(self, input_dim: int, encoding_dims: list = None,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 256,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, encoding_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = None
        logger.info(f"AnomalyDetector using device: {self.device}")

    def fit(self, X_train: np.ndarray, percentile: float = 95.0):
        """Train on normal traffic only. Sets threshold from training reconstruction errors."""
        tensor = torch.FloatTensor(X_train).to(self.device)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}] Loss: {epoch_loss/len(loader):.6f}")

        # Set threshold using training reconstruction errors
        train_errors = self._get_errors(X_train)
        self.threshold = float(np.percentile(train_errors, percentile))
        logger.info(f"Threshold set at {percentile}th percentile: {self.threshold:.6f}")
        return self

    def _get_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        return self.model.reconstruction_error(tensor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal."""
        if self.threshold is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        errors = self._get_errors(X)
        return (errors > self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns raw reconstruction error scores."""
        return self._get_errors(X)

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.threshold = ckpt['threshold']
        logger.info(f"Model loaded from {path}, threshold={self.threshold:.6f}")
        return self
