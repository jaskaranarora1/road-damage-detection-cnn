import os

TRAIN_IMAGE_DIR = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"

VAL_IMAGE_DIR = "dataset/val/images"
VAL_LABEL_DIR = "dataset/val/labels"

TEST_IMAGE_DIR = "dataset/test/images"
TEST_LABEL_DIR = "dataset/test/labels"

FIGURE_DIR = "figures"
MODEL_DIR = "models"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16

CUSTOM_CNN_EPOCHS = 5
MOBILENET_EPOCHS = 3

LABEL_MAP = {
    0: "pothole",
    1: "crack",
    2: "manhole"
}

LABEL_TO_NUM = {
    "pothole": 0,
    "crack": 1,
    "manhole": 2
}

CLASS_NAMES = ["pothole", "crack", "manhole"]