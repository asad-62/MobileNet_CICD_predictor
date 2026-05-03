import io

from fastapi.testclient import TestClient
from PIL import Image

from app.config import CLASS_NAMES, MODEL_NAME, MODEL_VERSION
from app.main import api


client = TestClient(api)


def test_predict_endpoint_with_valid_image():
    image = Image.new("RGB", (300, 300), color="white")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction_id" in data
    assert data["model_name"] == MODEL_NAME
    assert data["model_version"] == MODEL_VERSION
    assert data["face_type"] in CLASS_NAMES
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["confidence_percent"] <= 100.0
    assert data["confidence_label"] in ["low", "medium", "high"]
    assert isinstance(data["suggested_glasses"], list)
    assert isinstance(data["all_probabilities"], dict)
    assert data["latency_ms"] >= 0