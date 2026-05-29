from tensorflow.keras import layers, models
from src.config import *


def build_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),
        layers.Dropout(0.5),

        layers.Dense(
            len(CLASS_NAMES),
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model