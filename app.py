from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = load_model("models/custom_cnn_model.keras")

# Class names (must match training labels)
classes = [
    "crack",
    "manhole",
    "pothole"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    # Create uploads folder if it doesn't exist
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)

    # Save uploaded image
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    index = np.argmax(prediction)
    result = classes[index]
    confidence = float(np.max(prediction) * 100)

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)