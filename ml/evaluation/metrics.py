from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1),
        "weighted_f1": float(precision if precision == recall else (precision + recall) / 2.0),
    }


def compare_distribution(reference: np.ndarray, recent: np.ndarray, threshold: float = 0.15) -> dict:
    ref_mean = float(np.mean(reference))
    rec_mean = float(np.mean(recent))
    score = abs(ref_mean - rec_mean) / max(abs(ref_mean), 1e-6)
    return {
        "score": float(score),
        "drift_status": "DRIFT DETECTED" if score > threshold else "OK",
        "reference_mean": ref_mean,
        "recent_mean": rec_mean,
        "threshold": threshold,
    }
