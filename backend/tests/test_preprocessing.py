from pathlib import Path

import numpy as np
from PIL import Image

from ml.preprocessing.preprocess import build_split_metadata, validate_image


def test_validate_image(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
    assert validate_image(image_path) is True


def test_split_metadata(tmp_path: Path):
    dataset_root = tmp_path / "plantvillage"
    class_a = dataset_root / "ClassA"
    class_b = dataset_root / "ClassB"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    for label, count in [("ClassA", 3), ("ClassB", 3)]:
        class_dir = class_a if label == "ClassA" else class_b
        for idx in range(count):
            image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(image).save(class_dir / f"sample_{idx}.png")

    splits = build_split_metadata(dataset_root, tmp_path / "processed", random_state=42)
    assert set(splits.keys()) == {"classes", "train", "validation", "test"}
    assert len(splits["classes"]) == 2
    assert sum(len(v) for v in [splits["train"], splits["validation"], splits["test"]]) >= 4
