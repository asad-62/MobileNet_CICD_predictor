import tensorflow as tf
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define class names for face types
CLASS_NAMES = ['heart', 'long', 'oval', 'round', 'square']
SUNGLASSES_RECOMMENDATIONS = {
    "heart": ["Aviator", "Cat-eye"],
    "long": ["Oversized", "Square/Rectangular"],
    "oval": ["Square/Rectangular", "Cat-eye"],
    "round": ["Square/Rectangular", "Cat-eye"],
    "square": ["Round/Oval", "Aviator"]
}

def preprocess(img: Image.Image) -> np.ndarray:
    """
    Preprocess the PIL image:
    - Resize to 224×224
    - Apply MobileNetV2 preprocess_input
    - Add batch dimension
    """
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Load dummy model once
MODEL_PATH = "face_type_classifier.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.
    Returns: dict with face_type, confidence, sunglasses recommendations.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = preprocess(img)
        outputs = MODEL.predict(arr)
        idx = int(np.argmax(outputs, axis=1)[0])
        cls = CLASS_NAMES[idx]
        confidence = float(np.max(outputs) * 100)
        return {
            "face_type": cls,
            "confidence": round(confidence, 2),
            "suggested_glasses": SUNGLASSES_RECOMMENDATIONS.get(cls, [])
        }
    except Exception as e:
        return {"error": str(e)}
