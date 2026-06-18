# Road Damage Detection Using Convolutional Neural Networks

A deep learning project that classifies road surface images into three damage
categories — **pothole**, **crack**, and **manhole** — using a custom
Convolutional Neural Network (CNN), compared against a MobileNetV2 transfer
learning baseline. The project includes the trained model, a research report, and
a live web application that runs the model directly in the browser.

**Author:** Jaskaran Singh
**Student ID:** 33633568
**Course:** Pattern Recognition — M.Sc. Software Engineering
**University:** University of Europe for Applied Sciences

**Live website:** _ADD YOUR NETLIFY URL HERE_

---

## Project Overview

Manual road inspection is expensive and time-consuming. This project explores
whether a CNN can automatically detect road damage from images, which could help
transportation authorities plan repairs more efficiently. A custom CNN is trained
and compared with MobileNetV2 to evaluate how a purpose-built model performs
against a pre-trained one.

## Dataset

- **Total images:** 18,674
- **Classes:** 3 (pothole, crack, manhole)
- **Split:** 13,148 training · 2,759 validation · 2,767 test
- **Image type:** JPEG, resized to 128×128 pixels

## Results

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| **Custom CNN** | **63.93%** | **0.8302** |
| MobileNetV2 | 55.26% | 1.0370 |

The custom CNN outperformed MobileNetV2 by about 8.7 percentage points. Potholes
were the most reliably detected class; cracks were the most challenging due to
high visual variability and fewer samples.

## Repository Structure

```
road-damage-detection-cnn/
├── README.md                  This file
├── requirements.txt           Python dependencies (for training)
├── .gitignore
├── main.py                    Training script (trains + saves models, figures)
├── model_results.csv          Final accuracy/loss results
│
├── src/                       Training source code
│   ├── config.py              Paths, image size, class names, hyperparameters
│   ├── data_preprocessing.py  Dataset loading + preprocessing
│   ├── cnn_model.py           Custom CNN architecture
│   ├── mobilenet_model.py     MobileNetV2 transfer learning model
│   └── evaluation.py          Plots: curves, confusion matrix, comparison
│
├── models/                    Saved trained models (.keras)
├── figures/                   Output figures (curves, confusion matrix, etc.)
├── notebooks/                 Kaggle notebook
├── proposal/                  Phase 2 proposal document
│
└── website/                   Live web app (runs the model in the browser)
    ├── index.html             Web interface (TensorFlow.js)
    └── tfjs_model/            Model converted to TensorFlow.js format
        ├── model.json
        └── group1-shard1of4.bin ... group1-shard4of4.bin
```

## How to Run the Training

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset under dataset/ following the structure in src/config.py
#    (train/val/test folders with images and labels)

# 3. Run training — this trains both models and saves figures + models
python main.py
```

Outputs are saved to `models/` (trained models) and `figures/` (plots).

## The Web Application

The web app classifies a road image as a pothole, crack, or manhole. It uses
**TensorFlow.js** to run the trained CNN entirely in the browser — no server
required. The Keras model was converted to TensorFlow.js format (the
`website/tfjs_model/` folder).

**Run it locally:**

```bash
cd website
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

The app is also deployed live (see the live website link above).

## Methodology Summary

1. **Preprocessing** — images resized to 128×128 and normalized to [0, 1].
2. **Augmentation** — random rotation (15°), zoom (20%), and horizontal flip on
   the training set only.
3. **Custom CNN** — three convolutional blocks (32, 64, 128 filters), each with
   ReLU activation and max pooling, followed by a dense layer (128 units),
   dropout (0.5), and a softmax output over three classes.
4. **Transfer learning** — MobileNetV2 pre-trained on ImageNet with a frozen base
   and a new classification head.
5. **Evaluation** — accuracy/loss curves, confusion matrix, classification report,
   and a model comparison.

## Evaluation Metrics

Accuracy, loss, precision, recall, F1-score, and confusion matrix.

## Tech Stack

Python · TensorFlow / Keras · NumPy · Pillow · scikit-learn · Matplotlib ·
Seaborn · TensorFlow.js (web app)

---

*This is an academic project. Predictions reflect the trained model and are not
intended for real-world infrastructure decisions.*