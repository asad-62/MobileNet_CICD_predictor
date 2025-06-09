import tensorflow as tf
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Define class names for face types
CLASS_NAMES = ['heart', 'long', 'oval', 'round', 'square']
SUNGLASSES_RECOMMENDATIONS = {
    "heart": ["Aviator", "Cat-eye"],
    "long": ["Oversized", "Square/Rectangular"],
    "oval": ["Square/Rectangular", "Cat-eye"],
    "round": ["Square/Rectangular", "Cat-eye"],
    "square": ["Round/Oval", "Aviator"]
}

def load_model(model_path: str):
    """
    Load and return a Keras model from the given file path.
    """
    return tf.keras.models.load_model(model_path)

def preprocess(img: Image.Image) -> np.ndarray:
    """
    Preprocess the PIL image:
    - Resize to 224×224
    - Scale to [0, 1]
    - Normalize by ImageNet mean/std
    - Add batch dimension
    """
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return np.expand_dims(img_array, axis=0)

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and run the preprocess step.
    """
    img = Image.open(image_path).convert("RGB")
    return preprocess(img)

def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes. Returns a dict with:
    - face_type (str)
    - confidence (float percent)
    - suggested_glasses (list)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = preprocess(img)
        model = load_model("face_type_classifier_5classes.h5")
        outputs = model.predict(arr)
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

def predict_face_type(model, image_path: str) -> str:
    """
    Given a loaded model and path to an image, return the face type
    (title-cased) without extra info.
    """
    arr = preprocess_image(image_path)
    outputs = model.predict(arr)
    idx = int(np.argmax(outputs, axis=1)[0])
    return CLASS_NAMES[idx].title()
