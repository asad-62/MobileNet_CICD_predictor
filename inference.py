# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the model
# model = load_model('mobilenetv2_imagenet.h5')

# # Load and preprocess image
# img_path = '/home/asad/asad_project/3d_reconstruction/images/1.jpg'  # replace with your image file
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# # Predict
# preds = model.predict(x)
# decoded = decode_predictions(preds, top=5)[0]

# # Print predictions
# for i, (imagenet_id, label, prob) in enumerate(decoded):
#     print(f"{i+1}: {label} ({prob:.4f})")
# inference.py
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

model = load_model("mobilenetv2_imagenet.h5")

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = preprocess(img)
    preds = model.predict(processed)
    class_idx = preds.argmax()
    return int(class_idx)
