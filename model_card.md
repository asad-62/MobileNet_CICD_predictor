# Model Card: Face Type Classifier

## Model Overview

This model predicts face type from an uploaded face image and supports a sunglasses recommendation system.

The classifier is based on MobileNetV2 and is deployed as part of a FastAPI + Gradio inference service.

## Model Details

- Model name: mobilenetv2-face-type-classifier
- Model version: 1.0.0
- Framework: TensorFlow / Keras
- Input type: RGB image
- Input size: 224 x 224
- Output type: multi-class classification
- Deployment format: Keras `.keras` model

## Supported Classes

The model predicts one of five face types:

```text
heart
long
oval
round
square