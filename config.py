"""Central settings for ParkVision AI (paths, training hyper-parameters, business rules, logging)."""
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- data -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = (DATA_DIR / "raw") if (DATA_DIR / "raw").exists() else (BASE_DIR / "raw_dataset")
DATASET_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
SAMPLES_DIR = BASE_DIR / "samples"

CLASS_NAMES = ["empty", "occupied"]         # index 0 = empty, 1 = occupied (alphabetical)
SPLIT_RATIOS = (0.70, 0.15, 0.15)           # train / val / test

# ---- model paths ------------------------------------------------------
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "parkvision_mobilenetv2.keras"
STAGE1_PATH = MODEL_DIR / "_stage1_best.keras"
STAGE2_PATH = MODEL_DIR / "_stage2_best.keras"

# YOLO model paths (YOLOv8 slot classifier)
YOLO_MODEL_PATH = MODEL_DIR / "best.pt"
YOLO_LAST_PATH = MODEL_DIR / "last.pt"
YOLO_BASE_MODEL = "yolov8n-cls.pt"          # pre-trained lightweight nano backbone
YOLO_RUNS_DIR = BASE_DIR / "runs"

# ---- reports & logging ------------------------------------------------
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "classification_report.txt"
HISTORY_PLOT_PATH = REPORT_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
METRICS_PATH = REPORT_DIR / "metrics.json"
SUMMARY_MD_PATH = REPORT_DIR / "results_summary.md"
DATASET_SUMMARY_PATH = REPORT_DIR / "dataset_summary.json"
TRAINING_STATE_PATH = REPORT_DIR / "training_state.json"
LOG_FILE_PATH = REPORT_DIR / "training.log"

# ---- training hyper-parameters ----------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15                 # head training (MobileNet)
FINE_TUNE_EPOCHS = 10       # fine-tuning phase (MobileNet)
YOLO_EPOCHS = 25            # YOLO epochs (assignment asks 15-30 total)
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT_LAYER = 100    # MobileNetV2 has 154 layers; layers before this stay frozen
SEED = 42

# ---- business rules (assignment, Step 4) -------------------------------
LOW_CONGESTION_MAX = 40.0       # occupancy % below this  -> Low
HIGH_CONGESTION_MIN = 75.0      # occupancy % above this  -> High (40-75 inclusive = Moderate)
DEFAULT_THRESHOLD = 0.20        # P(occupied) >= threshold -> occupied (optimized for shaded outdoor lots)


def get_logger(name: str = "ParkVision") -> logging.Logger:
    """Create and configure a standard logger writing to both console and training.log."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

        # Console handler (safe for Windows CP1252)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(str(LOG_FILE_PATH), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
