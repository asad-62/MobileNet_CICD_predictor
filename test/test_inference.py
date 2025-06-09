import pytest
from inference import load_model, preprocess_image, predict_face_type

@pytest.fixture(scope="module")
def model():
    """Load your Keras model once per test session."""
    return load_model('face_type_classifier_5classes.h5')

def test_preprocess_image(tmp_path):
    # Create a tiny RGB image and save
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_path = tmp_path / "gray.jpg"
    img.save(img_path)
  
    arr = preprocess_image(str(img_path))
    # Model expects shape (1,224,224,3) and normalized [0,1]
    assert arr.shape == (1, 224, 224, 3)
    assert arr.max() <= 1.0 and arr.min() >= 0.0

def test_predict_face_type_returns_known_class(model, tmp_path):
    # Reuse the gray image
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(200, 180, 150))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    face_type = predict_face_type(model, str(img_path))
    assert face_type in {"Heart", "Long", "Oval", "Round", "Square"}
