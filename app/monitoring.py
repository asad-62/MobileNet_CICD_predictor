import json
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from app.config import LOG_DIR

####
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_PATH = LOG_DIR / "predictions.log"
####


def log_prediction(event:dict[str,Any])->None:
    """
    Append one prediction/error event as JSONL.
    JSONL = one JSON object per line.
    """
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with PREDICTION_LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


def get_prediction_metrics() -> dict[str, Any]:
    if not PREDICTION_LOG_PATH.exists():
        return {
            "total_requests": 0,
            "successful_predictions": 0,
            "errors": 0,
            "class_distribution": {},
            "average_confidence": None,
            "average_latency_ms": None,
            "low_confidence_count": 0,
        }
    
    total_requests = 0
    successful_predictions = 0
    errors = 0
    class_distribution: dict[str, int] = {}
    confidence_sum = 0.0
    latency_sum = 0.0
    latency_count = 0
    low_confidence_count = 0

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            event = json.loads(line)
            total_requests += 1

            if "error" in event:
                errors += 1
                continue

            successful_predictions += 1

            face_type = event.get("face_type")
            if face_type:
                class_distribution[face_type] = class_distribution.get(face_type, 0) + 1

            confidence = event.get("confidence")
            if confidence is not None:
                confidence_sum += float(confidence)

            latency_ms = event.get("latency_ms")
            if latency_ms is not None:
                latency_sum += float(latency_ms)
                latency_count += 1

            if event.get("confidence_label") == "low":
                low_confidence_count += 1

    average_confidence = None
    if successful_predictions > 0:
        average_confidence = round(confidence_sum / successful_predictions, 6)



    average_latency_ms = None
    if latency_count > 0:
        average_latency_ms = round(latency_sum / latency_count, 2)


    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "errors": errors,
        "class_distribution": class_distribution,
        "average_confidence": average_confidence,
        "average_latency_ms": average_latency_ms,
        "low_confidence_count": low_confidence_count,
    }