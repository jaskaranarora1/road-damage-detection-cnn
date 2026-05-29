import os
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

from src.config import *
from src.data_preprocessing import load_all_datasets
from src.cnn_model import build_custom_cnn
from src.mobilenet_model import build_mobilenet_model
from src.evaluation import (
    plot_accuracy,
    plot_loss,
    plot_confusion_matrix,
    plot_model_comparison
)


def main():
    print("Road Damage Detection Using CNNs")
    print("Project started")

    X_train, y_train, X_val, y_val, X_test, y_test = load_all_datasets()

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    print("Training Custom CNN")

    custom_cnn = build_custom_cnn()

    custom_history = custom_cnn.fit(
        datagen.flow(
            X_train,
            y_train,
            batch_size=BATCH_SIZE
        ),
        validation_data=(X_val, y_val),
        epochs=CUSTOM_CNN_EPOCHS
    )

    custom_loss, custom_accuracy = custom_cnn.evaluate(
        X_test,
        y_test
    )

    print("Custom CNN Test Accuracy:", custom_accuracy)

    plot_accuracy(
        custom_history,
        "custom_cnn_accuracy_curve.png",
        "Custom CNN Accuracy Curve"
    )

    plot_loss(
        custom_history,
        "custom_cnn_loss_curve.png",
        "Custom CNN Loss Curve"
    )

    y_pred = custom_cnn.predict(X_test)

    y_pred_classes = np.argmax(
        y_pred,
        axis=1
    )

    plot_confusion_matrix(
        y_test,
        y_pred_classes
    )

    print("Classification Report")
    print(
        classification_report(
            y_test,
            y_pred_classes,
            target_names=CLASS_NAMES
        )
    )

    custom_cnn.save(
        os.path.join(
            MODEL_DIR,
            "custom_cnn_model.keras"
        )
    )

    print("Training MobileNetV2")

    mobilenet_model = build_mobilenet_model()

    mobilenet_history = mobilenet_model.fit(
        datagen.flow(
            X_train,
            y_train,
            batch_size=BATCH_SIZE
        ),
        validation_data=(X_val, y_val),
        epochs=MOBILENET_EPOCHS
    )

    mobilenet_loss, mobilenet_accuracy = mobilenet_model.evaluate(
        X_test,
        y_test
    )

    print("MobileNetV2 Test Accuracy:", mobilenet_accuracy)

    plot_accuracy(
        mobilenet_history,
        "mobilenet_accuracy_curve.png",
        "MobileNetV2 Accuracy Curve"
    )

    mobilenet_model.save(
        os.path.join(
            MODEL_DIR,
            "mobilenet_model.keras"
        )
    )

    plot_model_comparison(
        custom_accuracy,
        mobilenet_accuracy
    )

    results = pd.DataFrame({
        "Model": ["Custom CNN", "MobileNetV2"],
        "Test Accuracy": [
            custom_accuracy,
            mobilenet_accuracy
        ],
        "Test Loss": [
            custom_loss,
            mobilenet_loss
        ]
    })

    results.to_csv(
        "model_results.csv",
        index=False
    )

    print("Project completed successfully")
    print("Figures saved in figures folder")
    print("Models saved in models folder")


if __name__ == "__main__":
    main()