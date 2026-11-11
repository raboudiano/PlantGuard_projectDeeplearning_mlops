import numpy as np

from ml.monitoring.drift import compute_feature_drift


def test_compute_feature_drift():
    ref = np.array([0.13, 0.15, 0.12, 0.14], dtype=np.float32)
    recent = np.array([0.12, 0.13, 0.11, 0.12], dtype=np.float32)
    result = compute_feature_drift(ref, recent, threshold=0.15)
    assert result["drift_status"] in {"OK", "DRIFT DETECTED"}
    assert "score" in result
