from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "archive" / "plantvillage dataset" / "color"


def discover_classes(dataset_root: str | Path | None = None) -> List[str]:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    classes = sorted([entry.name for entry in root.iterdir() if entry.is_dir()])
    return classes


def list_image_files(dataset_root: str | Path | None = None) -> List[Path]:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    image_files: List[Path] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_files.append(file_path)
    return image_files


def build_manifest(dataset_root: str | Path | None = None, output_path: str | Path | None = None) -> Dict[str, List[str]]:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    classes = discover_classes(root)
    manifest: Dict[str, List[str]] = {}
    for class_name in classes:
        class_dir = root / class_name
        image_paths = [
            str(path)
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        manifest[class_name] = image_paths

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

    return manifest
