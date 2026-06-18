# Road Damage Detection Using Convolutional Neural Networks

A deep learning project that classifies road surface images into three damage categories: **pothole**, **crack**, and **manhole**. The project uses a custom Convolutional Neural Network (CNN) and compares its performance with MobileNetV2, a transfer learning model. The repository includes the trained model, research report, and a web application for predictions.

**Author:** Jaskaran Singh
**Student ID:** 33633568
**Course:** Pattern Recognition, M.Sc. Software Engineering
**University:** University of Europe for Applied Sciences

---

## Project Overview

Road inspections are usually carried out manually, which can be expensive and time-consuming. This project investigates whether a CNN can automatically identify road damage from images and help make road maintenance more efficient. A custom CNN model was developed and compared with MobileNetV2 to evaluate its performance against a pre-trained model.

## Dataset

* **Total images:** 18,674
* **Classes:** 3 (pothole, crack, manhole)
* **Split:** 13,148 training, 2,759 validation, 2,767 test
* **Image type:** JPEG, resized to 128 × 128 pixels
* **Dataset link:** *Add Kaggle dataset link here*

## Results

| Model          | Test Accuracy | Test Loss  |
| -------------- | ------------- | ---------- |
| **Custom CNN** | **63.93%**    | **0.8302** |
| MobileNetV2    | 55.26%        | 1.0370     |

The custom CNN achieved better results than MobileNetV2, improving test accuracy by approximately 8.7 percentage points. Potholes were detected most accurately, while cracks were the most difficult class because of their visual diversity and lower number of samples.

## Repository Structure

```text
road-damage-detection-cnn/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── model_results.csv
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── cnn_model.py
│   ├── mobilenet_model.py
│   └── evaluation.py
│
├── models/
├── figures/
├── notebooks/
├── proposal/
├── report/
│
├── app.py
├── templates/
│   └── index.html
└── static/uploads/
```

## How to Run the Training

```bash
# Install dependencies
pip install -r requirements.txt

# Place the dataset in the correct directory structure

# Train both models
python main.py
```

The trained models are saved in the `models/` directory, while plots and evaluation figures are saved in `figures/`.

## How to Run the Web App

The web application allows users to upload a road image and receive a prediction from the trained model.

```bash
# Ensure the trained model exists
python app.py
```

Open the URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

Upload an image and click **Detect Damage** to view the predicted class and confidence scores.

**Live Demo:** *Add deployment link here*

## Methodology

1. Images are resized to 128 × 128 pixels and normalized to the range [0,1].
2. Data augmentation includes rotation, zoom, and horizontal flipping.
3. The custom CNN uses three convolutional layers followed by dense and dropout layers.
4. MobileNetV2 is used as a transfer learning baseline with a frozen feature extraction layer.
5. Performance is evaluated using accuracy, loss, confusion matrix, and classification metrics.

## Evaluation Metrics

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Confusion Matrix

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pillow
* Scikit-learn
* Matplotlib
* Seaborn
* Flask

## Documents

* Proposal (Phase 2): `proposal/`
* Final Report (Phase 3): `report/`

---

This repository was created as part of an academic project for the Pattern Recognition module. The model is intended for research and educational purposes and should not be used as the sole basis for real-world infrastructure decisions.
git status