import tensorflow as tf
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Define class names for face types
CLASS_NAMES = ['heart', 'long', 'oval', 'round', 'square']

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
    """Run inference and decode prediction."""
    try:
        # Open and convert image to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Preprocess image
        image_array = preprocess(img)
        # Run inference
        outputs = model.predict(image_array)
        confidence = np.max(outputs)
        predicted = np.argmax(outputs, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted]
        confidence_score = confidence * 100
        label = f"{predicted_class} ({confidence_score:.2f}%)"
        return label
    except Exception as e:
        print("Prediction error:", e)
        return "error"