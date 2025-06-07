# Face Type Classifier + Sunglasses Recommender 😎

This project uses a deep learning model based on MobileNetV2 to classify human face types from images. Based on the predicted face shape, it suggests the most suitable sunglasses styles.

## 🔍 Face Types Supported

- Heart
- Long
- Oval
- Round
- Square

## 😎 Sunglasses Recommendation Mapping

| Face Type | Suggested Sunglasses |
|-----------|----------------------|
| Heart     | Aviator, Cat-eye     |
| Long      | Oversized, Square/Rectangular |
| Oval      | Square/Rectangular, Cat-eye |
| Round     | Square/Rectangular, Cat-eye |
| Square    | Round/Oval, Aviator  |

## 🚀 Features

- FastAPI backend
- Web-based UI (upload and classify images)
- REST API endpoint for programmatic access
- Styled with modern dark UI
- Deployed on Railway

## 🖼 Demo

![demo screenshot](screenshot.png) <!-- Optional: replace with your actual screenshot -->

## 📦 Installation

```bash
git clone https://github.com/asad-62/MobileNet_CICD_predictor.git
cd MobileNet_CICD_predictor
pip install -r requirements.txt
