# PlantGuard implementation plan

## Project scope
This project implements a production-oriented plant disease detection and MLOps platform built around the real PlantVillage dataset stored in `archive/plantvillage dataset/color`.

## Dataset commitment
- The source dataset is not committed to Git.
- The runtime code uses the dataset directory structure to discover classes dynamically.
- `data/raw` and `data/processed` will contain local working artifacts only.

## Phased deliverables
### Phase 1 — dataset and preprocessing
- Validate image files and detect corruption.
- Discover classes automatically from the dataset folder tree.
- Add reproducible preprocessing and augmentation logic.
- Create train/validation/test splits without leakage.

### Phase 2 — deep learning experiments
- Train a baseline CNN from scratch.
- Add transfer learning using `ResNet50`.
- Add fine-tuning workflow and compare metrics.
- Save model metadata and evaluation reports.

### Phase 3 — MLOps foundation
- Track experiments with MLflow.
- Save artifacts and metrics per run.
- Package the model and store a registry concept.

### Phase 4 — deployment and API
- Build a FastAPI service for inference.
- Add model-info and health endpoints.
- Expose a prediction API with validation and latency metadata.

### Phase 5 — frontend and monitoring
- Build a React dashboard to upload or use webcam inputs.
- Log prediction metadata and monitor latency/error rates.
- Add drift status calculation and monitoring summaries.

### Phase 6 — Docker, CI/CD and reproducibility
- Add Docker Compose services.
- Add GitHub Actions workflows for CI and CD.
- Add DVC configuration and project reproducibility notes.

## Technical decisions
- Training data source: `archive/plantvillage dataset/color`.
- Validation strategy: stratified split with class-aware handling.
- Model family: PyTorch CNN + ResNet-based transfer learning.
- Deployment artifact: ONNX-compatible model and a local production pointer.
- Monitoring: feature-space drift against reference statistics plus prediction logs.

## Implementation checkpoints
1. Dataset discovery working against the actual class folders.
2. Preprocessing pipeline produces train/val/test splits.
3. Baseline model trains successfully in a lightweight mode.
4. Transfer learning and fine tuning run with comparable metrics.
5. FastAPI prediction service loads the model and returns latency metadata.
6. Frontend displays uploaded-image inference in the dashboard.
7. Docker Compose orchestrates the project locally.
8. CI workflow runs lint + tests + Docker build.
9. Monitoring and drift checks produce actionable outputs.
10. Final project README documents the full end-to-end workflow.
