import io
import gradio as gr

from app.inference import predict


def gradio_predict(image):
    if image is None:
        return "No image uploaded.", "", "", "", ""

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = predict(image_bytes)

    if "error" in result:
        return (
            "Prediction failed",
            "",
            "",
            "",
            result["error"],
        )

    face_type = result["face_type"]
    confidence = f'{result["confidence_percent"]}% ({result["confidence_label"]})'
    glasses = ", ".join(result["suggested_glasses"])
    model_info = f'{result["model_name"]} v{result["model_version"]}'
    warning = result["warning"] or "No warning"

    return face_type, confidence, glasses, model_info, warning


custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    padding-top: 20px !important;
}

.main-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 6px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 20px;
}

.result-box {
    border-radius: 12px;
}
"""


with gr.Blocks(
    title="Face Type Classifier + Sunglasses Recommender",
) as demo:

    gr.HTML("""
        <div class="main-title">Face Type Classifier + Sunglasses Recommender</div>
        <div class="sub-title">
            Upload a frontal face image to predict face type and get sunglasses recommendations.
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload face image")
            predict_button = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            face_type_output = gr.Textbox(label="Predicted face type", elem_classes=["result-box"])
            confidence_output = gr.Textbox(label="Confidence", elem_classes=["result-box"])
            glasses_output = gr.Textbox(label="Suggested sunglasses", elem_classes=["result-box"])
            model_output = gr.Textbox(label="Model", elem_classes=["result-box"])
            warning_output = gr.Textbox(label="Warning", elem_classes=["result-box"])

    predict_button.click(
        fn=gradio_predict,
        inputs=image_input,
        outputs=[
            face_type_output,
            confidence_output,
            glasses_output,
            model_output,
            warning_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
    css=custom_css,
)