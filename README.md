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
- Basic monitoring endpoint
- Docker support
- Docker Compose support
- Automated tests
- Health and model metadata endpoints

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