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

# Load the model
model = tf.keras.models.load_model("face_type_classifier_5classes.h5")

def preprocess(img):
    """Preprocess the input image."""
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Apply mean and std
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image_bytes):
    """Run inference and return face type and suggested sunglasses."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = preprocess(img)

        outputs = model.predict(image_array)
        confidence = np.max(outputs)
        predicted = np.argmax(outputs, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted]
        confidence_score = confidence * 100

        # Sunglasses recommendation
        suggestions = SUNGLASSES_RECOMMENDATIONS.get(predicted_class, [])

        result = {
            "face_type": predicted_class,
            "confidence": round(confidence_score, 2),
            "suggested_glasses": suggestions
        }

        return result
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}
