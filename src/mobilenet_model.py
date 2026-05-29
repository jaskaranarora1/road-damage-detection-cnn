from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src.config import *


def build_mobilenet_model():
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
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