import io
import time
from typing import Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


from app.config import (
    CLASS_NAMES,
    HIGH_CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)
from app.recommender import recommend_sunglasses

MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)


## preprocessing function
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image into MobileNetV2-ready batch tensor.
    """
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def get_confidence_label(confidence: float) -> str:
    """
    Confidence is expected between 0 and 1.
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    return "medium"


def predict(image_bytes: bytes) -> dict[str, Any]:
    """
    Run model inference on raw image bytes.
    """
    start_time = time.perf_counter()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()

        width, height = img.size

        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            raise ValueError(
                f"Image is too small. Minimum size is {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}."
            )

       # arr = preprocess_image(img)
        arr = preprocess_image(img)

        outputs = MODEL.predict(arr, verbose=0)[0]
        probabilities = {
            class_name: round(float(prob), 6)
            for class_name, prob in zip(CLASS_NAMES, outputs)
        }

        idx = int(np.argmax(outputs))
        face_type = CLASS_NAMES[idx]
        confidence = float(outputs[idx])

        warning = None
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warning = "Low confidence prediction. Use a clearer frontal face image."

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "face_type": face_type,
            "confidence": round(confidence, 6),
            "confidence_percent": round(confidence * 100, 2),
            "confidence_label": get_confidence_label(confidence),
            "suggested_glasses": recommend_sunglasses(face_type),
            "all_probabilities": probabilities,
            "latency_ms": latency_ms,
            "warning": warning,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "error": str(exc),
            "latency_ms": latency_ms,
        }
    

