from __future__ import annotations

import json
from pathlib import Path

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT, build_manifest, discover_classes


def prepare_data(dataset_root: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    out_dir = Path(output_dir) if output_dir is not None else Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(root)
    manifest = build_manifest(root, out_dir / "dataset_manifest.json")

    summary = {
        "dataset_root": str(root),
        "class_count": len(classes),
        "classes": classes,
        "manifest_path": str(out_dir / "dataset_manifest.json"),
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary


if __name__ == "__main__":
    prepare_data()
