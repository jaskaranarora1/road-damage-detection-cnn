import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.config import *


def plot_accuracy(history, filename, title):
    plt.figure(figsize=(6, 4))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        os.path.join(FIGURE_DIR, filename)
    )

    plt.show()


def plot_loss(history, filename, title):
    plt.figure(figsize=(6, 4))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        os.path.join(FIGURE_DIR, filename)
    )

    plt.show()


def plot_confusion_matrix(y_test, y_pred_classes):
    cm = confusion_matrix(
        y_test,
        y_pred_classes
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("Custom CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "confusion_matrix.png"
        )
    )

    plt.show()


def plot_model_comparison(custom_accuracy, mobilenet_accuracy):
    plt.figure(figsize=(6, 4))

    plt.bar(
        ["Custom CNN", "MobileNetV2"],
        [custom_accuracy, mobilenet_accuracy]
    )

    plt.title("Model Comparison")
    plt.ylabel("Test Accuracy")

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "model_comparison.png"
        )
    )

    plt.show()