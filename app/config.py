from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"


# Model config
MODEL_NAME = "mobilenetv2-face-type-classifier"
MODEL_VERSION = "1.0.0"
MODEL_PATH = MODEL_DIR / "face_type_classifier.keras"

# Inference config
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["heart", "long", "oval", "round", "square"]

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.80

## size check
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
