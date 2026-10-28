"""State-aware YOLOv8 training and evaluation pipeline for ParkVision AI.

Supports:
- 70/15/15 dataset training and test-split evaluation.
- State-preserving auto-resumption: resumes seamlessly from `last.pt` if interrupted.
- Batch and epoch logging saved to both console and `reports/training.log`.
- Exports `models/best.pt`, `reports/metrics.json`, and evaluation plots.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from ultralytics import YOLO

import config
import prepare_dataset

logger = config.get_logger("TrainYOLO")


def ensure_dataset():
    """Ensure dataset is present and split into 70/15/15."""
    for split in ["train", "val", "test"]:
        for cls in config.CLASS_NAMES:
            p = config.DATASET_DIR / split / cls
            if not p.exists() or len(list(p.glob("*.jpg"))) == 0:
                logger.info(f"Dataset split '{split}/{cls}' missing or empty. Running dataset preparation...")
                prepare_dataset.prepare(
                    source=config.RAW_DIR,
                    out_dir=config.DATASET_DIR,
                    max_per_class=300,
                    min_per_class=100,
                    clean=False
                )
                return


def get_resume_checkpoint() -> Path | None:
    """Check if an interrupted training run can be resumed."""
    # Check in models/
    if config.YOLO_LAST_PATH.exists():
        return config.YOLO_LAST_PATH
    # Check in runs/
    last_pt_files = list(config.BASE_DIR.glob("runs/**/weights/last.pt"))
    if last_pt_files:
        # Return most recently modified last.pt
        last_pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return last_pt_files[0]
    return None


def evaluate_test_set(model: YOLO, test_dir: Path):
    """Run rigorous evaluation on the 15% held-out test split."""
    logger.info(f"Evaluating model on held-out test set: {test_dir}")
    empty_files = list((test_dir / "empty").glob("*.jpg"))
    occupied_files = list((test_dir / "occupied").glob("*.jpg"))

    test_files = [(f, 0) for f in empty_files] + [(f, 1) for f in occupied_files]
    if not test_files:
        logger.warning("No test files found for evaluation.")
        return {}

    y_true, y_pred, y_probs = [], [], []

    # Map class names
    names = getattr(model, "names", {0: "empty", 1: "occupied"})
    occ_idx = 1
    for k, v in names.items():
        if str(v).lower() == "occupied":
            occ_idx = int(k)
            break

    # Batch prediction for speed
    batch_size = 64
    for i in range(0, len(test_files), batch_size):
        batch = test_files[i:i + batch_size]
        batch_paths = [str(item[0]) for item in batch]
        batch_labels = [item[1] for item in batch]

        results = model.predict(batch_paths, imgsz=config.IMG_SIZE[0], verbose=False)
        for r, label in zip(results, batch_labels):
            prob = float(r.probs.data[occ_idx].cpu().numpy())
            pred = 1 if prob >= config.DEFAULT_THRESHOLD else 0
            y_probs.append(prob)
            y_pred.append(pred)
            y_true.append(label)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0)

    logger.info(f"\nTest Accuracy: {acc:.4%}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix [[TN FP][FN TP]]:\n{cm}")

    # Save reports
    config.REPORT_PATH.write_text(report)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("YOLOv8 Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        "model_architecture": "YOLOv8-cls",
        "accuracy": float(acc),
        "precision_occupied": float(prec),
        "recall_occupied": float(rec),
        "f1_occupied": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_images": len(y_true),
        "evaluated_at": datetime.now().isoformat(timespec="seconds")
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    summary_md = (
        "# ParkVision AI: YOLOv8 Evaluation Results\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Model** | YOLOv8 Nano Classifier |\n"
        f"| **Test Accuracy** | **{acc:.2%}** |\n"
        f"| **Precision (Occupied)** | {prec:.2%} |\n"
        f"| **Recall (Occupied)** | {rec:.2%} |\n"
        f"| **F1-Score (Occupied)** | {f1:.2%} |\n"
        f"| **Test Images Evaluated** | {len(y_true)} |\n"
    )
    config.SUMMARY_MD_PATH.write_text(summary_md)
    return metrics


def train(epochs: int = config.YOLO_EPOCHS, batch: int = config.BATCH_SIZE, resume: bool = True):
    """Train YOLOv8 with state tracking and resumption protection."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_dataset()

    resume_pt = get_resume_checkpoint() if resume else None

    if resume_pt and resume_pt.exists():
        logger.info(f"===========================================================")
        logger.info(f"RESUMING TRAINING from existing checkpoint: {resume_pt}")
        logger.info(f"Training will NOT restart from the beginning.")
        logger.info(f"===========================================================")
        model = YOLO(str(resume_pt))
        results = model.train(
            data=str(config.DATASET_DIR),
            epochs=epochs,
            imgsz=config.IMG_SIZE[0],
            batch=batch,
            resume=True
        )
    else:
        logger.info(f"===========================================================")
        logger.info(f"STARTING FRESH YOLOv8 TRAINING ({epochs} Epochs)")
        logger.info(f"Backbone: {config.YOLO_BASE_MODEL}")
        logger.info(f"Dataset: {config.DATASET_DIR}")
        logger.info(f"===========================================================")
        model = YOLO(config.YOLO_BASE_MODEL)
        results = model.train(
            data=str(config.DATASET_DIR),
            epochs=epochs,
            imgsz=config.IMG_SIZE[0],
            batch=batch,
            project=str(config.BASE_DIR / "runs"),
            name="parkvision_yolo",
            exist_ok=True,
            save=True
        )

    # Save state
    state = {
        "status": "completed",
        "epochs_requested": epochs,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_path": str(config.YOLO_MODEL_PATH)
    }
    config.TRAINING_STATE_PATH.write_text(json.dumps(state, indent=2))

    # Promote best weights to models/
    run_best = getattr(model.trainer, "best", None)
    if run_best and Path(run_best).exists():
        shutil.copy2(run_best, config.YOLO_MODEL_PATH)
        logger.info(f"Saved best model to {config.YOLO_MODEL_PATH}")
    elif Path("runs/parkvision_yolo/weights/best.pt").exists():
        shutil.copy2("runs/parkvision_yolo/weights/best.pt", config.YOLO_MODEL_PATH)

    run_last = getattr(model.trainer, "last", None)
    if run_last and Path(run_last).exists():
        shutil.copy2(run_last, config.YOLO_LAST_PATH)
        logger.info(f"Saved latest checkpoint to {config.YOLO_LAST_PATH}")

    # Evaluate on the 15% held-out test split
    eval_model = YOLO(str(config.YOLO_MODEL_PATH))
    evaluate_test_set(eval_model, config.TEST_DIR)

    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 parking occupancy model")
    parser.add_argument("--epochs", type=int, default=config.YOLO_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--no-resume", action="store_true", help="Force fresh start without resuming")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, resume=not args.no_resume)
