from fastapi.testclient import TestClient

from app.config import CLASS_NAMES, MODEL_NAME, MODEL_VERSION
from app.main import api


client = TestClient(api)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_name"] == MODEL_NAME
    assert data["model_version"] == MODEL_VERSION


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == MODEL_NAME
    assert data["model_version"] == MODEL_VERSION
    assert data["classes"] == CLASS_NAMES


def test_invalid_file_upload_returns_400():
    response = client.post(
        "/predict",
        files={"file": ("README.md", b"not an image", "text/plain")},
    )

    assert response.status_code == 400

    data = response.json()

    assert "prediction_id" in data
    assert data["error"] == "Uploaded file must be an image."
    assert data["latency_ms"] is None