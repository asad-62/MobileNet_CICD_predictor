import tensorflow as tf
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Define class names for face types
CLASS_NAMES = ['heart', 'long', 'oval', 'round', 'square']

def load_model(model_path="face_type_classifier_5classes.h5"):
    """Load the trained MobileNetV2 model."""
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)  # Adjust for 5 classes
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Load the model
model = load_model()

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