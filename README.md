# Face Type Classifier + Sunglasses Recommender

Production-style deep learning inference service for face type classification and sunglasses recommendation.

The system uses a MobileNetV2-based image classifier to predict face shape and applies a rule-based recommendation layer to suggest suitable sunglasses.

## Current Status

This project includes:

- FastAPI backend
- Gradio web UI
- TensorFlow/Keras inference
- Pydantic response schemas
- Prediction logging
- JSON metrics endpoint
- Prometheus metrics endpoint
- Docker support
- Docker Compose support
- Kubernetes manifests
- Helm chart
- Automated tests
- GitHub Actions CI
- Model card
- Offline evaluation script

## System Architecture

```text
User
  |
  v
Gradio UI          FastAPI API
  |                    |
  |                    v
  +-------------> Inference Service
                       |
                       v
              MobileNetV2 Keras Model
                       |
                       v
              Sunglasses Recommender
                       |
                       v
             Prediction Logs + Metrics
```

## Project Structure

```text
.
├── app/
│   ├── main.py              # FastAPI app and API endpoints
│   ├── gradio_app.py        # Gradio UI
│   ├── inference.py         # Model loading and prediction logic
│   ├── recommender.py       # Sunglasses recommendation rules
│   ├── schemas.py           # Pydantic response models
│   ├── monitoring.py        # Logging, JSON metrics, Prometheus metrics
│   ├── config.py            # Central configuration
│   └── __init__.py
├── models/
│   └── face_type_classifier.keras
├── configs/
│   └── model_config.yaml
├── reports/
│   ├── offline_evaluation.md
│   └── evaluation_results.json
├── scripts/
│   └── evaluate_model.py
├── tests/
│   ├── test_api.py
│   ├── test_predict_success.py
│   └── test_recommender.py
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── helm/
│   └── face-type-ai/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── model_card.md
└── README.md
```

## Model

The current model is a MobileNetV2-based Keras classifier.

Supported classes:

```text
heart
long
oval
round
square
```

Model path:

```text
models/face_type_classifier.keras
```

## Recommendation Logic

The recommender maps predicted face type to sunglasses styles:

```text
heart  -> Aviator, Cat-eye
long   -> Oversized, Square/Rectangular
oval   -> Square/Rectangular, Cat-eye
round  -> Square/Rectangular, Cat-eye
square -> Round/Oval, Aviator
```

The recommendation logic is separated from the model inference code.

## API Endpoints

### Root

```text
GET /
```

Returns basic service status.

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_name": "mobilenetv2-face-type-classifier",
  "model_version": "1.0.0"
}
```

### Model Info

```text
GET /model-info
```

Returns model name, version, and supported classes.

### Prediction

```text
POST /predict
```

Input:

```text
multipart/form-data image file
```

Example:

```bash
curl -s -X POST "http://localhost:7860/predict" \
  -F "file=@example.jpg" \
  | python -m json.tool
```

Example response:

```json
{
  "prediction_id": "uuid",
  "model_name": "mobilenetv2-face-type-classifier",
  "model_version": "1.0.0",
  "face_type": "long",
  "confidence": 0.755424,
  "confidence_percent": 75.54,
  "confidence_label": "medium",
  "suggested_glasses": [
    "Oversized",
    "Square/Rectangular"
  ],
  "all_probabilities": {
    "heart": 0.041651,
    "long": 0.755424,
    "oval": 0.031337,
    "round": 0.080222,
    "square": 0.091367
  },
  "latency_ms": 125.78,
  "warning": null
}
```

### JSON Metrics

```text
GET /metrics
```

Example response:

```json
{
  "total_requests": 9,
  "successful_predictions": 1,
  "errors": 8,
  "class_distribution": {
    "long": 1
  },
  "average_confidence": 0.755424,
  "average_latency_ms": 1039.77,
  "low_confidence_count": 0
}
```

### Prometheus Metrics

```text
GET /prometheus
```

Example:

```bash
curl -s http://localhost:7860/prometheus | grep face_
```

Exposed metrics include:

```text
face_predictions_total
face_prediction_errors_total
face_prediction_low_confidence_total
face_prediction_latency_seconds
face_prediction_class_total
```

### API Docs

```text
GET /docs
```

FastAPI-generated API documentation.

### Web UI

```text
GET /ui
```

Gradio user interface.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run checks:

```bash
make check
```

Start unified API + UI service:

```bash
make api
```

Open:

```text
http://localhost:7860/ui
http://localhost:7860/docs
http://localhost:7860/health
http://localhost:7860/prometheus
```

## Makefile Commands

```bash
make check
```

Runs syntax checks and tests.

```bash
make test
```

Runs automated tests.

```bash
make api
```

Starts the FastAPI service with Gradio mounted at `/ui`.

```bash
make health
```

Checks the `/health` endpoint.

```bash
make metrics
```

Checks the `/metrics` endpoint.

```bash
make eval
```

Runs offline evaluation on images under `data/eval/`.

```bash
make clean
```

Removes Python cache files.

## Run with Docker

Build image:

```bash
docker build -t face-type-recommender:latest .
```

Run container:

```bash
docker run --rm -p 7860:7860 face-type-recommender:latest
```

Open:

```text
http://localhost:7860/ui
http://localhost:7860/docs
http://localhost:7860/health
```

## Run with Docker Compose

Start service:

```bash
docker compose up --build
```

Stop service:

```bash
docker compose down
```

Docker Compose mounts local logs into the container:

```text
./logs:/app/logs
```

This keeps prediction logs persistent across container restarts.

## Kubernetes Deployment

Raw Kubernetes manifests are stored in:

```text
k8s/
```

Apply manifests:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check resources:

```bash
kubectl get pods -n face-type-ai
kubectl get svc -n face-type-ai
```

Port-forward locally:

```bash
kubectl port-forward -n face-type-ai svc/face-type-recommender 8000:7860
```

Open:

```text
http://localhost:8000/ui
http://localhost:8000/docs
http://localhost:8000/health
http://localhost:8000/prometheus
```

## Helm Deployment

Helm chart path:

```text
helm/face-type-ai
```

Render chart:

```bash
helm template face-type-ai ./helm/face-type-ai -n face-type-ai
```

Install chart:

```bash
helm install face-type-ai ./helm/face-type-ai -n face-type-ai --create-namespace
```

Check release:

```bash
helm list -n face-type-ai
kubectl get pods -n face-type-ai
kubectl get svc -n face-type-ai
```

Port-forward:

```bash
kubectl port-forward -n face-type-ai svc/face-type-recommender 8000:7860
```

Upgrade after changes:

```bash
helm upgrade face-type-ai ./helm/face-type-ai -n face-type-ai
```

Uninstall:

```bash
helm uninstall face-type-ai -n face-type-ai
```

## Minikube Notes

For local Minikube deployment, load the Docker image into Minikube:

```bash
docker build -t face-type-recommender:latest .
minikube image load face-type-recommender:latest
```

Then deploy with Helm:

```bash
helm install face-type-ai ./helm/face-type-ai -n face-type-ai --create-namespace
```

## Offline Evaluation

Expected evaluation data structure:

```text
data/eval/
├── heart/
├── long/
├── oval/
├── round/
└── square/
```

Run evaluation:

```bash
make eval
```

The evaluation script writes results to:

```text
reports/evaluation_results.json
```

Metrics include:

```text
accuracy
per-class accuracy
confusion matrix
average confidence
low-confidence count
failure examples
```

## Monitoring and Logging

Prediction metadata is logged to:

```text
logs/predictions.jsonl
```

Each log entry includes:

```text
prediction_id
timestamp_utc
model_name
model_version
face_type
confidence
confidence_label
latency_ms
warning
error, if applicable
```

Uploaded images are not stored. Only prediction metadata is logged.

## Tests

Run:

```bash
make test
```

Current tests cover:

```text
health endpoint
model info endpoint
invalid file upload handling
successful prediction pipeline
sunglasses recommendation rules
```

## CI

GitHub Actions CI runs on every push and pull request to `main`.

Workflow file:

```text
.github/workflows/ci.yml
```

CI steps:

```text
checkout repo
set up Python
install dependencies
run syntax check
run tests
```

## Production ML Concepts Covered

This project demonstrates:

```text
model serving
API contracts
input validation
error handling
model metadata
prediction logging
JSON metrics
Prometheus metrics
Dockerized deployment
Docker Compose
Kubernetes deployment
Helm packaging
automated tests
CI pipeline
offline evaluation
model documentation
```

## Limitations

- The model can still return a class for non-face images.
- No face detector is currently used before classification.
- The recommender is rule-based.
- Metrics are local unless connected to Prometheus/Grafana.
- No authentication is implemented.
- No live feedback loop is implemented yet.

## Future Improvements

- Add face detection before classification
- Add image quality checks
- Add Prometheus + Grafana dashboard
- Add model registry/version tracking
- Add user feedback collection
- Add retraining pipeline
- Add RAG or image embedding search as a second platform module
- Refactor later into multi-service layout:

```text
ai-platform/
├── services/
│   ├── face_classifier/
│   └── rag_service/
├── shared/
│   ├── monitoring/
│   ├── logging/
│   └── config/
├── k8s/
├── helm/
├── docker-compose.yml
└── README.md
```