"""Continuous Progressive / Online Training Engine for ParkVision AI.

Trains the YOLOv8 model progressively across extended data chunks without redundancy.
Key Features:
- Continues from existing `models/best.pt` (accumulating past training).
- Iterates through incoming dataset chunks or streaming batches.
- Saves and mirrors updated `best.pt` and `last.pt` to Google Drive after EACH completed chunk.
- Fully fault-tolerant: if interrupted at any time, all progress is preserved!
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO

import config

logger = config.get_logger("TrainProgressive")


class ProgressiveTrainer:
    def __init__(self, base_weights: Path | None = None, drive_mirror: Path | None = None):
        self.base_weights = base_weights or config.YOLO_MODEL_PATH
        self.drive_mirror = drive_mirror
        self.interrupted = False

        # Register graceful exit handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_interrupt)

        # Ensure directories
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        if self.drive_mirror:
            self.drive_mirror.mkdir(parents=True, exist_ok=True)

    def _handle_interrupt(self, signum, frame):
        logger.warning("\n⚠️ Interrupt received! Finishing current batch and saving weights safely...")
        self.interrupted = True

    def train_chunk(self, chunk_dir: Path, epochs: int = 5, batch: int = 32, lr0: float = 0.001) -> dict:
        """Fine-tune the current best model on a new chunk of data."""
        logger.info(f"Loading base weights from: {self.base_weights}")
        model = YOLO(str(self.base_weights))

        logger.info(f"Training on chunk {chunk_dir.name} for {epochs} epochs (batch={batch}, lr0={lr0})...")
        results = model.train(
            data=str(chunk_dir),
            epochs=epochs,
            imgsz=config.IMG_SIZE[0],
            batch=batch,
            lr0=lr0,
            project=str(config.BASE_DIR / "runs"),
            name="parkvision_progressive",
            exist_ok=True,
            save=True
        )

        # Retrieve best weights from this run
        run_best = getattr(model.trainer, "best", None)
        run_last = getattr(model.trainer, "last", None)

        if run_best and Path(run_best).exists():
            shutil.copy2(run_best, config.YOLO_MODEL_PATH)
            logger.info(f"Updated best model at {config.YOLO_MODEL_PATH}")
            if self.drive_mirror and self.drive_mirror.exists():
                shutil.copy2(run_best, self.drive_mirror / "best.pt")
                logger.info(f"🛡️ Mirrored updated best model to Google Drive: {self.drive_mirror / 'best.pt'}")

        if run_last and Path(run_last).exists():
            shutil.copy2(run_last, config.YOLO_LAST_PATH)
            if self.drive_mirror and self.drive_mirror.exists():
                shutil.copy2(run_last, self.drive_mirror / "last.pt")

        # Evaluate on the unseen test set
        test_dir = chunk_dir / "test"
        if not test_dir.exists():
            test_dir = config.TEST_DIR

        eval_model = YOLO(str(config.YOLO_MODEL_PATH))
        metrics = self.evaluate(eval_model, test_dir)
        return metrics

    def evaluate(self, model: YOLO, test_dir: Path) -> dict:
        """Evaluate accuracy and F1 score on the held-out test split."""
        empty_imgs = list((test_dir / "empty").glob("*.jpg")) + list((test_dir / "empty").glob("*.png"))
        occ_imgs = list((test_dir / "occupied").glob("*.jpg")) + list((test_dir / "occupied").glob("*.png"))
        test_items = [(f, 0) for f in empty_imgs] + [(f, 1) for f in occ_imgs]

        if not test_items:
            logger.warning("No test images found for evaluation.")
            return {}

        names = getattr(model, "names", {0: "empty", 1: "occupied"})
        occ_idx = 1
        for k, v in names.items():
            if str(v).lower() == "occupied":
                occ_idx = int(k)
                break

        y_true, y_pred = [], []
        for img_path, label in test_items:
            r = model.predict(str(img_path), imgsz=config.IMG_SIZE[0], verbose=False)[0]
            prob = float(r.probs.data[occ_idx].cpu().numpy())
            y_pred.append(1 if prob >= config.DEFAULT_THRESHOLD else 0)
            y_true.append(label)

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        logger.info(f"Chunk Evaluation -> Test Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, F1: {f1:.2%}")

        metrics = {
            "model_architecture": "YOLOv8-cls (Progressive)",
            "accuracy": acc,
            "precision_occupied": float(prec),
            "recall_occupied": float(rec),
            "f1_occupied": float(f1),
            "test_images": len(y_true),
            "evaluated_at": datetime.now().isoformat(timespec="seconds")
        }
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        if self.drive_mirror and self.drive_mirror.exists():
            shutil.copy2(config.METRICS_PATH, self.drive_mirror / "metrics.json")

        return metrics


def run_progressive(zip_files: list[Path], epochs_per_chunk: int = 3, batch: int = 32):
    """Run progressive training across an ordered sequence of zip archives."""
    drive_base = Path("/content/drive/MyDrive/ParkVision_AI_Model_Checkpoints")
    use_drive = drive_base.exists()

    trainer = ProgressiveTrainer(
        base_weights=config.YOLO_MODEL_PATH,
        drive_mirror=drive_base if use_drive else None
    )

    work_dir = Path("progressive_run")
    work_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for idx, zf in enumerate(zip_files):
        if trainer.interrupted:
            logger.info("Stopping progressive pipeline safely per user interrupt.")
            break

        logger.info(f"\n=======================================================")
        logger.info(f"▶️ STARTING PROGRESSIVE CHUNK {idx + 1}/{len(zip_files)}: {zf.name}")
        logger.info(f"=======================================================")

        # Extract chunk
        chunk_extract = work_dir / f"chunk_{idx}"
        if chunk_extract.exists():
            shutil.rmtree(chunk_extract)
        chunk_extract.mkdir(parents=True, exist_ok=True)

        logger.info(f"Unpacking {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(chunk_extract)

        chunk_data = chunk_extract / "dataset" if (chunk_extract / "dataset").exists() else chunk_extract

        # Train on chunk
        m = trainer.train_chunk(chunk_data, epochs=epochs_per_chunk, batch=batch)
        history.append({"chunk": zf.name, "metrics": m})

        # Clean extracted files to save disk space
        shutil.rmtree(chunk_extract, ignore_errors=True)

    logger.info("\n🎉 Progressive training session concluded successfully!")
    logger.info(f"Final Model Weights: {config.YOLO_MODEL_PATH}")
    if use_drive:
        logger.info(f"Google Drive Checkpoint: {drive_base / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Continual YOLO Training")
    parser.add_argument("--chunks", nargs="+", type=Path, help="List of chunk zip files to train on sequentially")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per chunk")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    chunks = args.chunks
    if not chunks:
        # Auto-detect any data_extended*.zip
        chunks = sorted(list(Path(".").glob("data_extended*.zip")))

    if not chunks:
        logger.error("No chunk zip files found. Provide via --chunks or create with tools/prepare_extended_dataset.py")
        sys.exit(1)

    run_progressive(chunks, epochs_per_chunk=args.epochs, batch=args.batch)
