from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import mlflow
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT
from ml.preprocessing.preprocess import build_split_metadata

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SUPPORTED_MODELS = (
    "mobilenet_v3_small",
    "resnet18",
    "efficientnet_b0",
)

train_tf = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomRotation(25),
    T.ColorJitter(0.4, 0.4, 0.3, 0.05),
    T.RandomApply([T.GaussianBlur(5)], p=0.3),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class PlantDataset(Dataset):
    def __init__(self, samples, class_to_idx, tf):
        self.samples, self.c2i, self.tf = samples, class_to_idx, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["image_path"]).convert("RGB")
        return self.tf(img), self.c2i[s["label"]]


def build_model(num_classes: int, model_name: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def _configure_mlflow() -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("plantguard-model-benchmark")
    return tracking_uri


def train_model(dataset_root=None, epochs=10, batch_size=32, learning_rate=3e-4,
                output_dir=None, max_samples_per_class=None, model_name: str = "mobilenet_v3_small",
                checkpoint_name=None, log_to_mlflow: bool = True) -> dict:
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    out_dir = Path(output_dir) if output_dir else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")

    checkpoint_name = checkpoint_name or ("plantguard_model.pt" if model_name == "mobilenet_v3_small" else f"plantguard_model_{model_name}.pt")
    model_path = out_dir / checkpoint_name
    metadata_path = out_dir / ("model_metadata.json" if checkpoint_name == "plantguard_model.pt" else f"model_metadata_{model_name}.json")

    build_split_metadata(root, Path("data/processed"), random_state=42)
    with open(Path("data/processed") / "splits.json", encoding="utf-8") as fp:
        splits = json.load(fp)

    class_names = splits["classes"]
    c2i = {c: i for i, c in enumerate(class_names)}

    train_samples = splits["train"]
    if max_samples_per_class:
        by_cls = {}
        for s in train_samples:
            by_cls.setdefault(s["label"], []).append(s)
        train_samples = [s for v in by_cls.values() for s in v[:max_samples_per_class]]

    train_dl = DataLoader(PlantDataset(train_samples, c2i, train_tf),
                          batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(PlantDataset(splits["validation"], c2i, eval_tf),
                        batch_size=batch_size, num_workers=2)

    model = build_model(len(class_names), model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    run = None
    if log_to_mlflow:
        _configure_mlflow()
        run = mlflow.start_run(run_name=f"{model_name}-epochs{epochs}")

    try:
        for epoch in range(epochs):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()
            acc = evaluate(model, val_dl, device)
            print(f"epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")
            if log_to_mlflow:
                mlflow.log_metric("val_accuracy", float(acc), step=epoch + 1)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": model_name,
                    "arch": model_name,
                    "image_size": IMG,
                    "mean": MEAN,
                    "std": STD,
                }, model_path)

        metadata = {
            "version": "2",
            "architecture": model_name,
            "model_name": model_name,
            "class_names": class_names,
            "dataset_root": str(root),
            "registry_stage": "candidate",
            "num_classes": len(class_names),
            "trained_on": "PlantVillage color split",
            "val_accuracy": best_acc,
            "model_path": str(model_path),
        }
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        if log_to_mlflow:
            mlflow.log_params({
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_samples_per_class": max_samples_per_class,
                "dataset_root": str(root),
            })
            mlflow.log_metric("best_val_accuracy", float(best_acc))
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(metadata_path), artifact_path="models")
        return metadata
    finally:
        if run is not None:
            mlflow.end_run()


def run_model_experiments(dataset_root=None, epochs=5, batch_size=32, learning_rate=3e-4,
                          output_dir=None, max_samples_per_class=None,
                          model_names: Iterable[str] | None = None) -> dict:
    model_names = list(model_names or SUPPORTED_MODELS)
    out_dir = Path(output_dir) if output_dir else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in model_names:
        metadata = train_model(
            dataset_root=dataset_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=out_dir,
            max_samples_per_class=max_samples_per_class,
            model_name=model_name,
            log_to_mlflow=True,
        )
        results.append({
            "model_name": model_name,
            "architecture": metadata["architecture"],
            "val_accuracy": metadata["val_accuracy"],
            "model_path": metadata["model_path"],
        })

    best_model = max(results, key=lambda item: item["val_accuracy"])
    best_checkpoint = Path(best_model["model_path"])
    default_checkpoint = out_dir / "plantguard_model.pt"
    default_metadata = out_dir / "model_metadata.json"
    if best_checkpoint.exists():
        shutil.copy2(best_checkpoint, default_checkpoint)
    best_metadata_path = out_dir / f"model_metadata_{best_model['model_name']}.json"
    if Path(best_metadata_path).exists():
        shutil.copy2(best_metadata_path, default_metadata)

    report = {
        "best_model": best_model["model_name"],
        "best_val_accuracy": float(best_model["val_accuracy"]),
        "experiments": results,
    }
    report_path = out_dir / "experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    return report


if __name__ == "__main__":
    run_model_experiments(epochs=6)
