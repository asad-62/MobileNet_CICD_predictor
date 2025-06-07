import torch
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.hub

# Define class names for face types
CLASS_NAMES = ['heart', 'long', 'oval', 'round', 'square']

def load_model(model_path="face_type_classifier_final.pth"):
    """Load the trained MobileNetV2 model."""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 5)  # Adjust for 5 classes
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load the model
model = load_model()

def preprocess(img):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def predict(image_bytes):
    """Run inference and decode prediction."""
    try:
        # Open and convert image to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Preprocess image
        image_tensor = preprocess(img)
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item() * 100
        label = f"{predicted_class} ({confidence_score:.2f}%)"
        return label
    except Exception as e:
        print("Prediction error:", e)
        return "error"