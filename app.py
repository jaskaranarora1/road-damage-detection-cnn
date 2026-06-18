from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Load trained model
model = load_model("models/custom_cnn_model.keras")

# Class names — order MUST match training (config.py CLASS_NAMES)
# Training used: index 0 = pothole, 1 = crack, 2 = manhole
classes = ["pothole", "crack", "manhole"]

# Short descriptions shown in the UI for the predicted class
class_info = {
    "pothole": "A depression or hole in the road surface. Potholes are a major "
               "safety hazard and a common cause of vehicle damage.",
    "crack": "A surface fracture or fissure in the pavement. Cracks often "
             "develop into larger damage if left unrepaired.",
    "manhole": "A manhole cover and the surrounding road surface. Displaced or "
               "damaged manholes can pose a risk to traffic.",
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def run_prediction(filepath):
    """Preprocess an image and return sorted (class, probability) pairs."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    # Pair each class with its probability (%) and sort high -> low
    results = [
        {"label": classes[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(classes))
    ]
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Save with a unique name so repeat uploads don't collide / cache
    ext = os.path.splitext(file.filename)[1] or ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    results = run_prediction(filepath)
    top = results[0]

    # AJAX request -> return JSON (used by the modern UI)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "prediction": top["label"],
            "confidence": top["confidence"],
            "description": class_info[top["label"]],
            "all_results": results,
            "image": "/" + filepath.replace("\\", "/"),
        })

    # Fallback: classic full-page render
    return render_template(
        "index.html",
        prediction=top["label"],
        confidence=top["confidence"],
        description=class_info[top["label"]],
        all_results=results,
        image="/" + filepath.replace("\\", "/"),
    )


if __name__ == "__main__":
    app.run(debug=True)