# -*- coding: utf-8 -*-
"""
generate_data.py -- Synthetic CICIDS-format network traffic data generator.

Produces realistic normal traffic for training and a held-out test set
with NOVEL attack types not present in training (simulating unseen threats).

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --train_normal 15000 --test_normal 8000 --test_attack 3500
"""

import argparse
import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Feature definitions (CICIDS-style)
FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

LABEL_COL = "Label"
NORMAL_LABEL = "BENIGN"


def gen_normal(n, seed=42):
    """Generate plausible BENIGN network flow records."""
    rng = np.random.default_rng(seed)
    rows = {}

    rows["Destination Port"] = rng.choice([80, 443, 53, 22, 8080, 8443, 3389, 25, 143, 993], n)
    rows["Flow Duration"] = rng.exponential(scale=500_000, size=n).clip(1, 1e8)
    rows["Total Fwd Packets"] = rng.integers(2, 30, n)
    rows["Total Backward Packets"] = rng.integers(1, 25, n)
    fwd_pkt_len = rng.gamma(2, 300, n).clip(0, 65535)
    bwd_pkt_len = rng.gamma(2, 400, n).clip(0, 65535)
    rows["Total Length of Fwd Packets"] = fwd_pkt_len
    rows["Total Length of Bwd Packets"] = bwd_pkt_len
    rows["Fwd Packet Length Max"] = fwd_pkt_len * rng.uniform(0.8, 1.0, n)
    rows["Fwd Packet Length Min"] = fwd_pkt_len * rng.uniform(0.0, 0.2, n)
    rows["Fwd Packet Length Mean"] = fwd_pkt_len * rng.uniform(0.4, 0.7, n)
    rows["Fwd Packet Length Std"] = rng.gamma(1, 50, n)
    rows["Bwd Packet Length Max"] = bwd_pkt_len * rng.uniform(0.8, 1.0, n)
    rows["Bwd Packet Length Min"] = bwd_pkt_len * rng.uniform(0.0, 0.2, n)
    rows["Bwd Packet Length Mean"] = bwd_pkt_len * rng.uniform(0.4, 0.7, n)
    rows["Bwd Packet Length Std"] = rng.gamma(1, 60, n)
    rows["Flow Bytes/s"] = rng.gamma(3, 20_000, n).clip(0, 1e8)
    rows["Flow Packets/s"] = rng.gamma(2, 50, n).clip(0, 1e6)
    iat = rng.exponential(100_000, n)
    rows["Flow IAT Mean"] = iat
    rows["Flow IAT Std"] = iat * rng.uniform(0.1, 0.5, n)
    rows["Flow IAT Max"] = iat * rng.uniform(1.5, 3.0, n)
    rows["Flow IAT Min"] = iat * rng.uniform(0.0, 0.3, n)
    rows["Fwd IAT Total"] = rng.exponential(400_000, n)
    rows["Fwd IAT Mean"] = rng.exponential(80_000, n)
    rows["Fwd IAT Std"] = rng.exponential(40_000, n)
    rows["Fwd IAT Max"] = rng.exponential(200_000, n)
    rows["Fwd IAT Min"] = rng.exponential(10_000, n)
    rows["Bwd IAT Total"] = rng.exponential(300_000, n)
    rows["Bwd IAT Mean"] = rng.exponential(70_000, n)
    rows["Bwd IAT Std"] = rng.exponential(35_000, n)
    rows["Bwd IAT Max"] = rng.exponential(180_000, n)
    rows["Bwd IAT Min"] = rng.exponential(8_000, n)
    rows["Fwd PSH Flags"] = rng.integers(0, 2, n)
    rows["Bwd PSH Flags"] = rng.integers(0, 2, n)
    rows["Fwd URG Flags"] = np.zeros(n, int)
    rows["Bwd URG Flags"] = np.zeros(n, int)
    rows["Fwd Header Length"] = rng.integers(20, 60, n) * rows["Total Fwd Packets"]
    rows["Bwd Header Length"] = rng.integers(20, 60, n) * rows["Total Backward Packets"]
    rows["Fwd Packets/s"] = rows["Flow Packets/s"] * rng.uniform(0.4, 0.7, n)
    rows["Bwd Packets/s"] = rows["Flow Packets/s"] * rng.uniform(0.3, 0.6, n)
    pkt_len = rng.gamma(2, 350, n)
    rows["Min Packet Length"] = pkt_len * rng.uniform(0.0, 0.2, n)
    rows["Max Packet Length"] = pkt_len * rng.uniform(0.8, 1.0, n)
    rows["Packet Length Mean"] = pkt_len * rng.uniform(0.4, 0.7, n)
    rows["Packet Length Std"] = rng.gamma(1, 55, n)
    rows["Packet Length Variance"] = rows["Packet Length Std"] ** 2
    rows["FIN Flag Count"] = rng.integers(0, 2, n)
    rows["SYN Flag Count"] = rng.integers(0, 2, n)
    rows["RST Flag Count"] = rng.integers(0, 1, n)
    rows["PSH Flag Count"] = rng.integers(0, 3, n)
    rows["ACK Flag Count"] = rng.integers(1, 5, n)
    rows["URG Flag Count"] = np.zeros(n, int)
    rows["CWE Flag Count"] = np.zeros(n, int)
    rows["ECE Flag Count"] = np.zeros(n, int)
    rows["Down/Up Ratio"] = rng.uniform(0.5, 3.0, n)
    rows["Average Packet Size"] = rows["Packet Length Mean"]
    rows["Avg Fwd Segment Size"] = rows["Fwd Packet Length Mean"]
    rows["Avg Bwd Segment Size"] = rows["Bwd Packet Length Mean"]
    for col in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        rows[col] = np.zeros(n)
    rows["Subflow Fwd Packets"] = rows["Total Fwd Packets"]
    rows["Subflow Fwd Bytes"] = rows["Total Length of Fwd Packets"]
    rows["Subflow Bwd Packets"] = rows["Total Backward Packets"]
    rows["Subflow Bwd Bytes"] = rows["Total Length of Bwd Packets"]
    rows["Init_Win_bytes_forward"] = rng.choice([8192, 16384, 32768, 65535, -1], n)
    rows["Init_Win_bytes_backward"] = rng.choice([8192, 16384, 32768, 65535, -1], n)
    rows["act_data_pkt_fwd"] = rng.integers(1, 10, n)
    rows["min_seg_size_forward"] = rng.integers(20, 40, n)
    active = rng.exponential(50_000, n)
    rows["Active Mean"] = active
    rows["Active Std"] = active * rng.uniform(0.1, 0.4, n)
    rows["Active Max"] = active * rng.uniform(1.2, 2.5, n)
    rows["Active Min"] = active * rng.uniform(0.0, 0.5, n)
    idle = rng.exponential(200_000, n)
    rows["Idle Mean"] = idle
    rows["Idle Std"] = idle * rng.uniform(0.1, 0.4, n)
    rows["Idle Max"] = idle * rng.uniform(1.2, 2.5, n)
    rows["Idle Min"] = idle * rng.uniform(0.0, 0.5, n)
    rows[LABEL_COL] = NORMAL_LABEL

    return pd.DataFrame(rows)


def gen_port_scan(n, rng):
    """Port scan: many short flows, varying dst ports, tiny packets."""
    df = gen_normal(n, seed=int(rng.integers(0, 9999)))
    df["Destination Port"] = rng.integers(1, 65535, n)
    df["Flow Duration"] = rng.integers(1, 500, n).astype(float)
    df["Total Fwd Packets"] = rng.integers(1, 3, n)
    df["Total Backward Packets"] = rng.integers(0, 2, n)
    df["Total Length of Fwd Packets"] = rng.integers(40, 80, n).astype(float)
    df["Total Length of Bwd Packets"] = rng.integers(0, 40, n).astype(float)
    df["Flow Bytes/s"] = rng.uniform(1e5, 5e6, n)
    df["Flow Packets/s"] = rng.uniform(1e4, 1e6, n)
    df["SYN Flag Count"] = rng.integers(1, 3, n)
    df["RST Flag Count"] = rng.integers(0, 2, n)
    df["ACK Flag Count"] = np.zeros(n, int)
    df[LABEL_COL] = "PortScan"
    return df


def gen_dos_slowloris(n, rng):
    """DoS Slowloris: very long duration, many packets, slow IAT."""
    df = gen_normal(n, seed=int(rng.integers(0, 9999)))
    df["Flow Duration"] = rng.integers(50_000_000, 100_000_000, n).astype(float)
    df["Total Fwd Packets"] = rng.integers(100, 500, n)
    df["Total Backward Packets"] = rng.integers(1, 5, n)
    df["Flow IAT Mean"] = rng.exponential(5_000_000, n)
    df["Flow Bytes/s"] = rng.uniform(10, 500, n)
    df["Flow Packets/s"] = rng.uniform(0.1, 2, n)
    df["FIN Flag Count"] = np.zeros(n, int)
    df["ACK Flag Count"] = rng.integers(50, 200, n)
    df[LABEL_COL] = "DoS Slowloris"
    return df


def gen_c2_beacon(n, rng):
    """C2 Beacon: very regular IAT, small packets, periodic."""
    df = gen_normal(n, seed=int(rng.integers(0, 9999)))
    beacon_period = rng.choice([30_000_000, 60_000_000, 300_000_000], n)
    df["Flow IAT Mean"] = beacon_period.astype(float)
    df["Flow IAT Std"] = (beacon_period * 0.02).astype(float)
    df["Flow IAT Max"] = (beacon_period * 1.05).astype(float)
    df["Flow IAT Min"] = (beacon_period * 0.95).astype(float)
    df["Total Fwd Packets"] = rng.integers(2, 6, n)
    df["Total Backward Packets"] = rng.integers(2, 6, n)
    df["Packet Length Mean"] = rng.integers(64, 256, n).astype(float)
    df["Packet Length Std"] = rng.uniform(0, 5, n)
    df[LABEL_COL] = "C2Beacon"
    return df


def gen_data_exfil(n, rng):
    """Data Exfiltration: large outbound, small inbound, unusual ports."""
    df = gen_normal(n, seed=int(rng.integers(0, 9999)))
    df["Destination Port"] = rng.choice([4444, 8888, 31337, 12345, 54321], n)
    df["Total Length of Fwd Packets"] = rng.uniform(1e6, 5e6, n)
    df["Total Length of Bwd Packets"] = rng.uniform(10, 500, n)
    df["Down/Up Ratio"] = rng.uniform(0.0, 0.05, n)
    df["Flow Bytes/s"] = rng.uniform(5e5, 2e6, n)
    df[LABEL_COL] = "DataExfiltration"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CICIDS-format data")
    parser.add_argument("--train_normal", type=int, default=15000)
    parser.add_argument("--test_normal", type=int, default=8000)
    parser.add_argument("--test_attack", type=int, default=3500)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(99)

    print("[1/3] Generating %d normal training flows..." % args.train_normal)
    train_df = gen_normal(args.train_normal)
    train_path = os.path.join(args.out_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print("  -> %s  shape=%s" % (train_path, str(train_df.shape)))

    print("\n[2/3] Generating test set (%d normal + %d attack)..." % (args.test_normal, args.test_attack))
    test_normal_df = gen_normal(args.test_normal, seed=77)

    n_per_attack = args.test_attack // 4
    attack_parts = [
        gen_port_scan(n_per_attack, rng),
        gen_dos_slowloris(n_per_attack, rng),
        gen_c2_beacon(n_per_attack, rng),
        gen_data_exfil(args.test_attack - 3 * n_per_attack, rng),
    ]
    attack_df = pd.concat(attack_parts, ignore_index=True)
    test_df = pd.concat([test_normal_df, attack_df], ignore_index=True).sample(frac=1, random_state=7)
    test_path = os.path.join(args.out_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print("  -> %s  shape=%s" % (test_path, str(test_df.shape)))
    print("  Attack types: %s" % str(dict(attack_df[LABEL_COL].value_counts())))

    anomaly_rate = (test_df[LABEL_COL] != NORMAL_LABEL).mean()
    print("\n[3/3] Done. Anomaly rate in test: %.2f%%" % (anomaly_rate * 100))


if __name__ == "__main__":
    main()
