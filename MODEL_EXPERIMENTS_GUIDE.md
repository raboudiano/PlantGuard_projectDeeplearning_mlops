# PlantGuard model experiments guide

This file explains how the project evaluates multiple CNN models, logs results in MLflow, and selects the best model for deployment.

## 1. Project goal

The project trains a plant disease classification model over the PlantVillage dataset and compares several deep learning architectures to decide which one performs best on validation data.

## 2. Main training file

The core experiment code is in:
- `ml/training/train.py`

It contains:
- dataset loading and preprocessing
- model builders for several architectures
- training loop
- validation accuracy calculation
- MLflow logging
- experiment runner for comparison across multiple models

## 3. Supported models

The experiment runner currently supports:
- `mobilenet_v3_small`
- `resnet18`
- `efficientnet_b0`

These are defined in `SUPPORTED_MODELS` inside `ml/training/train.py`.

## 4. How experiments work

The function `run_model_experiments(...)` does the following:

1. Takes a list of model names to compare.
2. Trains each model separately.
3. Logs validation metrics to MLflow.
4. Saves the best checkpoint for each model.
5. Compares validation accuracy across runs.
6. Copies the best model into `models/plantguard_model.pt`.
7. Saves a summary JSON report at `models/experiment_report.json`.

## 5. MLflow integration

The project includes an MLflow service in `docker-compose.yml`:
- service name: `mlflow`
- tracking server: `http://mlflow:5000`
- backend store: SQLite at `./mlruns/mlflow.db`

In the training code, MLflow is configured with:
- `mlflow.set_tracking_uri(...)`
- `mlflow.set_experiment("plantguard-model-benchmark")`

During each training run, metrics like `val_accuracy` and `best_val_accuracy` are logged.

## 6. How to run the benchmark

From the project root:

```bash
cd "C:\Users\boub0\Desktop\mlops project"
set PYTHONPATH=.
python -c "from ml.training.train import run_model_experiments; print(run_model_experiments(epochs=6, model_names=['mobilenet_v3_small','resnet18','efficientnet_b0']))"
```

You can also run a smaller experiment for testing:

```bash
python -c "from ml.training.train import run_model_experiments; print(run_model_experiments(epochs=1, max_samples_per_class=2, model_names=['mobilenet_v3_small','resnet18']))"
```

## 7. Output files

After a successful benchmark, these files are produced:

- `models/experiment_report.json`
  - contains the best model and all experiment results
- `models/plantguard_model.pt`
  - selected best checkpoint used by the app
- `models/model_metadata.json`
  - metadata for the deployed model
- `mlruns/`
  - MLflow experiment metadata and metrics

## 8. How model selection is decided

The runner picks the winner using:

```python
best_model = max(results, key=lambda item: item["val_accuracy"])
```

This means the best model is the one with the highest validation accuracy.

## 9. The deployed model

The backend service loads the chosen checkpoint and metadata from:
- `models/plantguard_model.pt`
- `models/model_metadata.json`

The loader is implemented in:
- `backend/app/services/model_service.py`

This service supports the architecture saved in the checkpoint, instead of assuming only one architecture.

## 10. Validation status

The project test suite passes with the current setup:

```bash
set PYTHONPATH=.
pytest backend/tests -q
```

Result:
- 6 tests passed

## 11. Recommended workflow

For a real model selection cycle:

1. Prepare the dataset.
2. Run experiment benchmark for multiple models.
3. Inspect `models/experiment_report.json`.
4. Compare validation accuracy and choose the best model.
5. Keep the best checkpoint as the production model.
6. Track all experiment runs in MLflow for reproducibility.

## 12. Useful references

- `ml/training/train.py`
- `backend/app/services/model_service.py`
- `docker-compose.yml`
- `ml/evaluation/evaluate.py`
- `ml/evaluation/metrics.py`

This setup gives you a clean benchmark workflow for model comparison and deployment decision-making.
