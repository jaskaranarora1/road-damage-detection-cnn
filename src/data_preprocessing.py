import os
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.config import *


def find_image_file(image_dir, label_file):
    base_name = os.path.splitext(label_file)[0]

    possible_extensions = [".jpg", ".jpeg", ".png"]

    for ext in possible_extensions:
        image_path = os.path.join(image_dir, base_name + ext)

        if os.path.exists(image_path):
            return image_path

    return None


def create_dataframe(image_dir, label_dir, split_name):
    data = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, "r") as file:
                first_line = file.readline().strip()

                if first_line:
                    class_id = int(first_line.split()[0])

                    image_path = find_image_file(image_dir, label_file)

                    if image_path is not None and class_id in LABEL_MAP:
                        data.append([image_path, LABEL_MAP[class_id]])

    df = pd.DataFrame(data, columns=["image", "label"])

    print(f"{split_name} dataset loaded")
    print("Total images:", len(df))
    print(df["label"].value_counts())
    print()

    return df


def load_images(df):
    X = []
    y = []

    for _, row in df.iterrows():
        img = load_img(
            row["image"],
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img = img_to_array(img)
        img = img / 255.0

        X.append(img)
        y.append(LABEL_TO_NUM[row["label"]])

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_all_datasets():
    train_df = create_dataframe(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        "Training"
    )

    val_df = create_dataframe(
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        "Validation"
    )

    test_df = create_dataframe(
        TEST_IMAGE_DIR,
        TEST_LABEL_DIR,
        "Testing"
    )

    X_train, y_train = load_images(train_df)
    X_val, y_val = load_images(val_df)
    X_test, y_test = load_images(test_df)

    print("All images loaded successfully")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test