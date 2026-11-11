import io
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms as T

from backend.app.config import MODEL_META_PATH, MODEL_PATH, ROOT_DIR
from ml.training.train import build_model

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(MODEL_PATH)
        self.meta_path = Path(MODEL_META_PATH)
        self.model = None
        self.transform = None
        self.loaded_mtime = None
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        self.meta = self._load_metadata()
        self.model_version = self.meta.get("version", "local-dev-v1")
        self.class_names = self.meta.get("class_names", [])
        if self.class_names and self.model_path.exists():
            mtime = self.model_path.stat().st_mtime
            if self.model is None or mtime != self.loaded_mtime:
                self._load_model(mtime)

    def _load_model(self, mtime: float) -> None:
        checkpoint = torch.load(self.model_path, map_location="cpu")
        if "arch" not in checkpoint:
            raise RuntimeError("Old checkpoint format. Retrain with ml.training.train.")
        if checkpoint["class_names"] != self.class_names:
            raise RuntimeError("Model metadata and checkpoint class mappings do not match.")

        model_name = checkpoint.get("model_name") or checkpoint.get("arch") or "mobilenet_v3_small"
        model = build_model(len(self.class_names), model_name=model_name, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        size = checkpoint["image_size"]
        self.transform = T.Compose([
            T.Resize(int(size * 256 / 224)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(checkpoint["mean"], checkpoint["std"]),
        ])
        self.model = model
        self.loaded_mtime = mtime

    def _load_metadata(self) -> Dict:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        return {
            "version": "local-dev-v1",
            "architecture": "mobilenet_v3_small",
            "class_names": [],
            "dataset_root": str(ROOT_DIR / "archive/plantvillage dataset" / "color"),
            "registry_stage": "development",
        }

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict(self, image_bytes: bytes) -> Dict:
        self.refresh_metadata()
        if not self.class_names:
            raise RuntimeError("No model metadata is available. Train the model first.")
        if self.model is None:
            raise RuntimeError("The model is not loaded. Train the model first.")

        start = time.perf_counter()
        input_tensor = self.preprocess_image(image_bytes)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]
        top_scores, top_indices = torch.topk(probabilities, k=min(3, len(self.class_names)))
        confidence = float(top_scores[0])
        return {
            "predicted_class": self.class_names[int(top_indices[0])],
            "confidence": confidence,
            "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
            "top_predictions": [
                {"class_name": self.class_names[int(i)], "confidence": float(s)}
                for s, i in zip(top_scores, top_indices)
            ],
            "model_version": self.model_version,
            "inference_time_ms": (time.perf_counter() - start) * 1000.0,
        }

    def get_model_info(self) -> Dict:
        self.refresh_metadata()
        return {
            "model_name": "PlantGuardClassifier",
            "version": self.model_version,
            "architecture": self.meta.get("architecture", "mobilenet_v3_small"),
            "dataset_root": self.meta.get("dataset_root", str(ROOT_DIR / "archive/plantvillage dataset" / "color")),
            "training_status": "ready" if self.class_names else "training_required",
            "registry_stage": self.meta.get("registry_stage", "development"),
        }

    def get_monitoring_snapshot(self) -> Dict:
        self.refresh_metadata()
        return {
            "prediction_count": 0,
            "average_inference_latency_ms": 0.0,
            "drift_status": "OK",
            "deployed_model": self.model_version,
        }