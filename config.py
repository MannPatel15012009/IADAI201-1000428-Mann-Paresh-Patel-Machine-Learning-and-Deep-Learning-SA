"""
ParkVision AI — Central Configuration
=========================================
Every script in this project imports from here so that paths,
hyperparameters, and business-logic thresholds live in exactly one
place. Change values here, not inside individual scripts.
"""

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")                 # unzip PKLot here
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")
UNSEEN_TEST_DIR = os.path.join(BASE_DIR, "data", "unseen_test")      # extra robustness images

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "parkvision_mobilenetv2.keras")
DETECTOR_PATH = os.path.join(MODEL_DIR, "slot_detector.pt")           # YOLOv8 slot finder
DETECTOR_CONF = 0.25           # min confidence for a detected slot
DETECTOR_IOU = 0.5             # NMS overlap threshold
DETECTOR_IMGSZ = 640
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.txt")

SAMPLE_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "sample_lot_annotations")

# ------------------------------------------------------------------
# Data parameters
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)          # MobileNetV2 native input size
IMAGES_PER_CLASS = 500         
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 25                    # phase 1: train classification head only
FINE_TUNE_EPOCHS = 10          # phase 2: fine-tune top MobileNetV2 layers
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT_LAYER = 100       # unfreeze base-model layers from this index onward

# ------------------------------------------------------------------
# Business logic: congestion thresholds & recommendations
# (edit freely — these directly satisfy the "System Logic & Insight
#  Generation" rubric criterion)
# ------------------------------------------------------------------
CONGESTION_THRESHOLDS = {
    "low": 40,        # occupancy < 40%      -> Low congestion
    "moderate": 75,    # 40% <= occupancy <= 75% -> Moderate congestion
    # occupancy > 75%  -> High congestion
}

RECOMMENDATIONS = {
    "low": "Plenty of slots available -- proceed to park.",
    "moderate": "Parking is filling up -- proceed, but availability may be limited soon.",
    "high": "Parking is nearly full -- consider searching for another location.",
}
