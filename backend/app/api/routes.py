from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.model_service import ModelService

router = APIRouter()
service = ModelService()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "plantguard-backend"}


@router.get("/model/info")
def model_info() -> Dict[str, str]:
    return service.get_model_info()


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    if not file.filename or not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image file.")
    contents = await file.read()
    try:
        result = service.predict(contents)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top_predictions": result["top_predictions"],
        "model_version": result["model_version"],
        "inference_time_ms": result["inference_time_ms"],
    }


@router.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> Dict[str, List[Dict]]:
    results = []
    for file in files:
        contents = await file.read()
        results.append(service.predict(contents))
    return {"results": results}


@router.get("/metrics")
def metrics() -> Dict[str, float | int | str]:
    snapshot = service.get_monitoring_snapshot()
    return {
        "prediction_count": snapshot["prediction_count"],
        "average_inference_latency_ms": snapshot["average_inference_latency_ms"],
        "drift_status": snapshot["drift_status"],
        "deployed_model": snapshot["deployed_model"],
    }
