import gradio as gr
from inference import predict

def gradio_predict(image):
    # Gradio gives you a PIL Image, so convert to bytes
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = predict(image_bytes)

    if "error" in result:
        return f"❌ Error: {result['error']}", None, None

    face_type = result["face_type"]
    confidence = f"{result['confidence']}%"
    glasses = ", ".join(result["suggested_glasses"])
    return face_type, confidence, glasses

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Label(label="Predicted Face Type"),
        gr.Label(label="Confidence"),
        gr.Label(label="Suggested Sunglasses"),
    ],
    title="Face Type Classifier + Sunglasses Recommender 😎",
    description="Upload an image and get face type + sunglasses recommendations using MobileNetV2."
)

if __name__ == "__main__":
    demo.launch()
