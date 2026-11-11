from typing import List
from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_uncertain: bool = False
    top_predictions: List[TopPrediction]
    model_version: str
    inference_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    architecture: str
    dataset_root: str
    training_status: str
    registry_stage: str


class MetricsSummary(BaseModel):
    prediction_count: int
    average_inference_latency_ms: float
    drift_status: str
    deployed_model: str
