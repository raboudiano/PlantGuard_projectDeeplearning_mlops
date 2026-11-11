from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_feature_drift(reference_features: np.ndarray, recent_features: np.ndarray, threshold: float = 0.15) -> dict:
    ref = np.asarray(reference_features, dtype=np.float32).ravel()
    rec = np.asarray(recent_features, dtype=np.float32).ravel()
    ref_mean = float(np.mean(ref))
    rec_mean = float(np.mean(rec))
    score = abs(ref_mean - rec_mean) / max(abs(ref_mean), 1e-6)
    return {
        "drift_status": "DRIFT DETECTED" if score > threshold else "OK",
        "score": float(score),
        "threshold": threshold,
        "reference_mean": ref_mean,
        "recent_mean": rec_mean,
    }


def sample_drift_summary(reference_path: str | Path = "data/processed/class_distribution.json", recent_path: str | Path = "data/processed/class_distribution.json", threshold: float = 0.15) -> dict:
    with open(reference_path, "r", encoding="utf-8") as fp:
        ref = json.load(fp)
    with open(recent_path, "r", encoding="utf-8") as fp:
        rec = json.load(fp)
    ref_values = np.array(list(ref.values()), dtype=np.float32)
    rec_values = np.array(list(rec.values()), dtype=np.float32)
    return compute_feature_drift(ref_values, rec_values, threshold=threshold)
