from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(ROOT_DIR / "archive/plantvillage dataset" / "color")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "models" / "plantguard_model.pt")))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", str(ROOT_DIR / "models" / "model_metadata.json")))
MONITORING_PATH = ROOT_DIR / "monitoring" / "predictions.jsonl"
