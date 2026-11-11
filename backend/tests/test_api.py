import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import service


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint_with_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(json.dumps({
        "version": "test-v1",
        "architecture": "simple_cnn",
        "class_names": ["test_class"],
        "dataset_root": str(tmp_path),
        "registry_stage": "candidate",
    }), encoding="utf-8")
    monkeypatch.setattr(service, "meta_path", metadata_path)
    monkeypatch.setattr(service, "predict", lambda image_bytes: {
        "predicted_class": "test_class",
        "confidence": 1.0,
        "top_predictions": [{"class_name": "test_class", "confidence": 1.0}],
        "model_version": "test-v1",
        "inference_time_ms": 0.0,
    })

    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    image_bytes = BytesIO()
    Image.fromarray(image_array).save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})
        assert response.status_code == 200
        payload = response.json()
        assert "predicted_class" in payload
        assert "top_predictions" in payload
