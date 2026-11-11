from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ml.preprocessing.preprocess import build_split_metadata
from ml.training.train import PlantDiseaseCNN, SimplePlantDataset


def evaluate_model(model_path: str | Path = "models/plantguard_model.pt") -> dict:
    build_split_metadata(Path("archive/plantvillage dataset/color"), Path("data/processed"), random_state=42)
    splits_path = Path("data/processed") / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as fp:
        splits = json.load(fp)

    class_names = splits["classes"]
    label_to_index = {label: index for index, label in enumerate(class_names)}
    validation_samples = splits["validation"]
    checkpoint = torch.load(model_path, map_location="cpu")
    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = DataLoader(SimplePlantDataset(validation_samples), batch_size=32)
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1).tolist()
            actual.extend(label_to_index[label] for label in labels)
            predicted.extend(predictions)

    rate = accuracy_score(actual, predicted)
    metrics = {
        "accuracy": float(rate),
        "precision": float(precision_score(actual, predicted, average="weighted", zero_division=0)),
        "recall": float(recall_score(actual, predicted, average="weighted", zero_division=0)),
        "f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "validation_samples": len(actual),
    }
    out_path = Path("mlruns") / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics


if __name__ == "__main__":
    print(evaluate_model())
