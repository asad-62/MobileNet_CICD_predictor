from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique ID for this prediction request")
    model_name: str
    model_version: str

    face_type: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    confidence_percent: float = Field(..., ge=0.0, le=100.0)
    confidence_label: str

    suggested_glasses: list[str]
    all_probabilities: dict[str, float]

    latency_ms: float
    warning: str | None = None


class ErrorResponse(BaseModel):
    prediction_id: str
    error: str
    latency_ms: float | None = None
