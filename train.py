"""
train.py — Train the Autoencoder anomaly detector and save artifacts.

Usage:
    python train.py --train data/train.csv --test data/test.csv --dataset cicids
"""

import argparse
import os
import json
import numpy as np

from src.data.preprocess import build_pipeline
from src.models.autoencoder import AnomalyDetector
from src.models.baselines import IsolationForestDetector, OneClassSVMDetector
from src.utils.evaluate import compute_metrics, print_report, save_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, help='Path to training CSV')
    p.add_argument('--test', required=True, help='Path to test CSV')
    p.add_argument('--dataset', default='cicids', choices=['cicids', 'unsw', 'nslkdd'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--percentile', type=float, default=95.0,
                   help='Percentile of training errors to use as threshold')
    p.add_argument('--output', default='artifacts', help='Output directory')
    p.add_argument('--baselines', action='store_true', help='Also run baseline models')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n[1/4] Preprocessing data...")
    X_train, X_test, y_test, scaler, features = build_pipeline(
        args.train, args.test, args.dataset
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape} | Anomaly rate: {y_test.mean():.2%}")

    print(f"\n[2/4] Training Autoencoder...")
    detector = AnomalyDetector(
        input_dim=X_train.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    detector.fit(X_train, percentile=args.percentile)

    print(f"\n[3/4] Evaluating...")
    scores = detector.score(X_test)
    preds = detector.predict(X_test)
    metrics = compute_metrics(y_test, preds, scores)
    print_report(metrics, model_name='Autoencoder')

    if args.baselines:
        print("  Running baselines...")
        iso = IsolationForestDetector()
        iso.fit(X_train)
        iso_m = compute_metrics(y_test, iso.predict(X_test), iso.score(X_test))
        print_report(iso_m, model_name='Isolation Forest')

        ocsvm = OneClassSVMDetector()
        ocsvm.fit(X_train[:5000])  # subsample for speed
        ocsvm_m = compute_metrics(y_test, ocsvm.predict(X_test))
        print_report(ocsvm_m, model_name='One-Class SVM')

        save_metrics({'autoencoder': metrics, 'isolation_forest': iso_m,
                      'one_class_svm': ocsvm_m}, f"{args.output}/metrics.json")
    else:
        save_metrics({'autoencoder': metrics}, f"{args.output}/metrics.json")

    print(f"\n[4/4] Saving artifacts...")
    detector.save(f"{args.output}/model.pt")
    with open(f"{args.output}/meta.json", 'w') as f:
        json.dump({'features': features, 'dataset': args.dataset,
                   'input_dim': X_train.shape[1]}, f, indent=2)
    print(f"  Artifacts saved to {args.output}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
