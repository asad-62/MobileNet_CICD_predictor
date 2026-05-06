from urllib import response
from uuid import uuid4

from fastapi.responses import JSONResponse, Response
import gradio as gr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.config import (
    ALLOWED_IMAGE_TYPES,
    CLASS_NAMES,
    MAX_UPLOAD_BYTES,
    MODEL_NAME,
    MODEL_VERSION,
)
from app.gradio_app import demo
from app.inference import predict
from app.monitoring import get_prediction_metrics, log_prediction, record_prometheus_metrics
from app.schemas import ErrorResponse, PredictionResponse

api = FastAPI(
    title="Face Type Classifier API",
    description="Production-style API for face type classification and sunglasses recommendation.",
    version="1.0.0",
)


@api.get("/")
def root():
    return {
        "service": "face-type-classifier",
        "status": "running",
    }


@api.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@api.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
    }


@api.get("/metrics")
def metrics():
    return get_prediction_metrics()
## prometheus metrics endpoint
@api.get("/prometheus")
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)



@api.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def predict_face(file: UploadFile = File(...)):
    prediction_id = str(uuid4())

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        error = ErrorResponse(
            prediction_id=prediction_id,
            error="Uploaded file must be an image.",
            latency_ms=None,
        )
        log_prediction(error.model_dump())
        record_prometheus_metrics(error.model_dump())

        return JSONResponse(status_code=400, content=error.model_dump())

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        error = ErrorResponse(
            prediction_id=prediction_id,
            error="Uploaded image is too large. Maximum allowed size is 5 MB.",
            latency_ms=None,
        )
        log_prediction(error.model_dump())
        record_prometheus_metrics(error.model_dump())
        return JSONResponse(status_code=400, content=error.model_dump())

    if len(image_bytes) == 0:
        error = ErrorResponse(
            prediction_id=prediction_id,
            error="Uploaded image is empty.",
            latency_ms=None,
        )
        log_prediction(error.model_dump())
        record_prometheus_metrics(error.model_dump())
        return JSONResponse(status_code=400, content=error.model_dump())

    result = predict(image_bytes)

    if "error" in result:
        error = ErrorResponse(
            prediction_id=prediction_id,
            error=result["error"],
            latency_ms=result.get("latency_ms"),
        )
        log_prediction(error.model_dump())
        record_prometheus_metrics(error.model_dump())
        return JSONResponse(status_code=400, content=error.model_dump())

    result["prediction_id"] = prediction_id
    response = PredictionResponse(**result)
    
    log_prediction(response.model_dump())
    record_prometheus_metrics(response.model_dump())
    return response

api = gr.mount_gradio_app(api, demo, path="/ui")