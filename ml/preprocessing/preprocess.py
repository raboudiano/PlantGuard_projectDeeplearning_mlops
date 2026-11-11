from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT, discover_classes


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_image_array(path: str | Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb = ImageOps.fit(rgb, target_size, Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array


def build_split_metadata(dataset_root: str | Path | None = None, output_dir: str | Path | None = None, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> dict:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    out_dir = Path(output_dir) if output_dir is not None else Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(root)
    samples: list[dict] = []
    for class_name in class_names:
        class_dir = root / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                if not validate_image(image_path):
                    continue
                samples.append({"image_path": str(image_path), "label": class_name})

    labels = [sample["label"] for sample in samples]
    class_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(index)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    can_stratify = all(len(indices) >= 3 for indices in class_to_indices.values())
    if can_stratify:
        train_idx, temp_idx = train_test_split(
            range(len(samples)),
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
        remaining_labels = [labels[i] for i in temp_idx]
        remaining_by_class = {}
        for index, label in zip(temp_idx, remaining_labels):
            remaining_by_class.setdefault(label, []).append(index)
        if all(len(indices) >= 2 for indices in remaining_by_class.values()):
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                stratify=remaining_labels,
                random_state=random_state,
            )
        else:
            for class_name in class_names:
                indices = remaining_by_class.get(class_name, [])
                if not indices:
                    continue
                if len(indices) == 1:
                    val_idx.append(indices[0])
                else:
                    val_idx.append(indices[0])
                    test_idx.append(indices[1])
                    for extra_index in indices[2:]:
                        if len(test_idx) <= len(val_idx):
                            test_idx.append(extra_index)
                        else:
                            val_idx.append(extra_index)
    else:
        for class_name in class_names:
            indices = class_to_indices.get(class_name, [])
            if not indices:
                continue
            if len(indices) == 1:
                train_idx.append(indices[0])
            elif len(indices) == 2:
                train_idx.append(indices[0])
                val_idx.append(indices[1])
            else:
                train_needed = max(1, int(round(len(indices) * (1.0 - test_size - val_size))))
                train_needed = min(len(indices) - 1, max(1, train_needed))
                remaining = len(indices) - train_needed
                val_needed = max(1, int(round(remaining * 0.5)))
                if val_needed >= remaining:
                    val_needed = max(1, remaining - 1)
                test_needed = remaining - val_needed
                train_idx.extend(indices[:train_needed])
                val_idx.extend(indices[train_needed:train_needed + val_needed])
                test_idx.extend(indices[train_needed + val_needed:])

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    splits = {
        "classes": class_names,
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples,
    }

    with open(out_dir / "splits.json", "w", encoding="utf-8") as fp:
        json.dump(splits, fp, indent=2)

    class_distribution = {label: sum(1 for s in samples if s["label"] == label) for label in class_names}
    with open(out_dir / "class_distribution.json", "w", encoding="utf-8") as fp:
        json.dump(class_distribution, fp, indent=2)

    return splits


def augment_image(image_array: np.ndarray) -> np.ndarray:
    augmented = image_array.copy()
    if np.random.rand() > 0.5:
        augmented = np.flip(augmented, axis=1)
    angle = np.random.uniform(-10, 10)
    # A lightweight augmentation pass without altering the image semantics too aggressively.
    augmented = np.rot90(augmented, k=int(np.round(angle / 90.0)) % 4)
    return augmented
