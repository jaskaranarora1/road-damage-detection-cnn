"""
Road Damage Detection - Web App (Gradio / Hugging Face Spaces)
Upload a road image and the trained custom CNN predicts whether it shows
a pothole, a crack, or a manhole.

Author: Jaskaran Singh
Pattern Recognition Project - Phase 3
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf

# -------------------------------------------------------------------
# Configuration - must match training (config.py)
# index 0 = pothole, 1 = crack, 2 = manhole
# -------------------------------------------------------------------
IMG_SIZE = 128
CLASS_NAMES = ["pothole", "crack", "manhole"]
MODEL_PATH = "custom_cnn_model.keras"

CLASS_INFO = {
    "pothole": "A depression or hole in the road surface. Potholes are a major "
               "safety hazard and a common cause of vehicle damage.",
    "crack":   "A surface fracture or fissure in the pavement. Cracks often "
               "develop into larger damage if left unrepaired.",
    "manhole": "A manhole cover and the surrounding road surface. Displaced or "
               "damaged manholes can pose a risk to traffic.",
}

# Load the trained model once at startup
model = tf.keras.models.load_model(MODEL_PATH)


def predict(image):
    if image is None:
        return {}, "Please upload a road image to get a prediction."

    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf = preds[top_idx] * 100
    explanation = (
        f"### Prediction: {top_class.capitalize()}  ({top_conf:.1f}% confidence)\n\n"
        f"{CLASS_INFO[top_class]}"
    )
    return confidences, explanation


description = """
This web app uses a custom Convolutional Neural Network (CNN) trained on
18,674 road images to classify road damage into three categories:
**Pothole**, **Crack**, and **Manhole**. Upload a road image to see the
prediction and confidence scores.
"""

article = """
---
**About this project**

Built for the Pattern Recognition course (M.Sc. Software Engineering,
University of Europe for Applied Sciences) by Jaskaran Singh. The model is a
custom CNN with three convolutional blocks (32, 64, 128 filters) reaching
63.93% test accuracy, compared against a MobileNetV2 baseline (55.26%).

*Academic demonstration only — not intended for real-world infrastructure
decisions.*
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber")) as demo:
    gr.Markdown("# Road Damage Detection using CNN")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a road image")
            submit_btn = gr.Button("Detect damage", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Prediction confidence")
            text_output = gr.Markdown()

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, text_output])
    image_input.change(fn=predict, inputs=image_input, outputs=[label_output, text_output])

    gr.Markdown(article)


if __name__ == "__main__":
    demo.launch()