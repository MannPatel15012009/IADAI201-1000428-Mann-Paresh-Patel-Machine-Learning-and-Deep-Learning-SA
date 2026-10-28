"""Unified Slot Classifier supporting YOLOv8 (.pt), MobileNetV2 (.keras), and Demo Fallback."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import config

logger = config.get_logger("Predictor")


class SlotClassifier:
    """Wrapper around trained parking models: YOLOv8 (.pt) or MobileNetV2 (.keras).

    Auto-detects model type based on extension and available files.
    If no trained model is found, smoothly falls back to texture/edge heuristic ("demo mode").
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_type = "Demo Heuristic"
        self.is_demo = True
        self.status = ""
        self.names = {}
        self.occupied_idx = 1

        # Auto-detect model if path is not explicitly given
        if model_path is None:
            if config.YOLO_MODEL_PATH.exists():
                model_path = config.YOLO_MODEL_PATH
            elif config.MODEL_PATH.exists():
                model_path = config.MODEL_PATH
            else:
                model_path = config.YOLO_MODEL_PATH  # for error messaging

        model_path = Path(model_path)

        if not model_path.exists():
            self.status = f"Trained model not found at {model_path.name}; running in demo mode."
            logger.info(self.status)
            return

        # ---- Load YOLOv8 model (.pt) ---------------------------------
        if model_path.suffix.lower() == ".pt":
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                self.model_type = "YOLOv8"
                self.is_demo = False
                self.names = getattr(self.model, "names", {0: "empty", 1: "occupied"})
                # Find index of 'occupied'
                for k, v in self.names.items():
                    if str(v).lower() == "occupied":
                        self.occupied_idx = int(k)
                        break
                self.status = f"Loaded YOLOv8 model ({model_path.name})"
                logger.info(self.status)
                return
            except Exception as exc:
                logger.warning(f"Could not load YOLO model: {exc}")

        # ---- Load MobileNetV2 Keras model (.keras) -------------------
        if model_path.suffix.lower() in {".keras", ".h5"} or self.is_demo:
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model_type = "MobileNetV2"
                self.is_demo = False
                self.status = f"Loaded MobileNetV2 ({model_path.name})"
                logger.info(self.status)
                return
            except Exception as exc:
                self.status = f"Could not load model ({exc.__class__.__name__}: {exc}); running in demo mode."
                logger.warning(self.status)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_array(crop: Image.Image) -> np.ndarray:
        return np.asarray(crop.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR), dtype=np.float32)

    @staticmethod
    def _demo_score(crop: Image.Image) -> float:
        """Texture, edge, and color variance heuristic for fallback demo mode."""
        a = np.asarray(crop.convert("RGB").resize((64, 64)), dtype=np.float32)
        ring = np.concatenate([a[:5].reshape(-1, 3), a[-5:].reshape(-1, 3),
                                a[:, :5].reshape(-1, 3), a[:, -5:].reshape(-1, 3)])
        background = np.median(ring, axis=0)
        inner = a[10:-10, 10:-10]
        diff = np.abs(inner - background).max(axis=2)
        color_score = float(min(1.0, (diff > 30).mean() * 2.5))
        gray = a.mean(axis=2)
        std_val = float(np.std(gray))
        edge_score = float(np.clip((std_val - 16.0) / 28.0, 0.0, 1.0))
        combined = max(color_score, edge_score * 0.85)
        return float(np.clip(combined, 0.0, 1.0))

    def predict_proba(self, crops: List[Optional[Image.Image]]) -> np.ndarray:
        """Return P(occupied) for each crop (None crops count as 0.0 = empty)."""
        probs = np.zeros(len(crops), dtype=np.float32)
        valid = [i for i, c in enumerate(crops) if c is not None]
        if not valid:
            return probs

        if self.is_demo:
            for i in valid:
                probs[i] = self._demo_score(crops[i])
            return probs

        # ---- Inference using YOLOv8 ----------------------------------
        if self.model_type == "YOLOv8":
            valid_images = [crops[i].convert("RGB") for i in valid]
            try:
                # Ultralytics batch prediction
                results = self.model.predict(valid_images, imgsz=config.IMG_SIZE[0], verbose=False)
                for idx, r in zip(valid, results):
                    if hasattr(r, "probs") and r.probs is not None:
                        # Classification model
                        prob_vec = r.probs.data.cpu().numpy()
                        probs[idx] = float(prob_vec[self.occupied_idx])
                    elif hasattr(r, "boxes") and len(r.boxes) > 0:
                        # Detection model: car confidence
                        probs[idx] = float(r.boxes.conf.max().cpu().numpy())
                    else:
                        probs[idx] = 0.0
                return probs
            except Exception as e:
                logger.error(f"YOLO inference error ({e}); falling back to heuristic.")
                for i in valid:
                    probs[i] = self._demo_score(crops[i])
                return probs

        # ---- Inference using MobileNetV2 (Keras) ---------------------
        batch = np.stack([self._to_array(crops[i]) for i in valid])
        try:
            preds = np.asarray(self.model(batch, training=False)).ravel()
        except Exception:
            preds = self.model.predict(batch, batch_size=64, verbose=0).ravel()
        probs[valid] = preds
        return probs


def load_classifier(model_path: Optional[Path] = None) -> SlotClassifier:
    return SlotClassifier(model_path)


class AutoSlotDetector:
    """Detects parking slot boundaries directly from surveillance images using slot_detector.pt."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        if model_path is None:
            model_path = config.MODEL_DIR / "slot_detector.pt"
        model_path = Path(model_path)
        if model_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                logger.info(f"Loaded slot detector ({model_path.name})")
            except Exception as exc:
                logger.warning(f"Could not load slot detector: {exc}")

    def detect_slots(self, image: Image.Image, conf: float = 0.15):
        """Returns slot polygons automatically sorted in natural reading order."""
        if self.model is None:
            return None
        try:
            res = self.model.predict(image, conf=conf, verbose=False)[0]
            if hasattr(res, "boxes") and len(res.boxes) > 0:
                raw_boxes = res.boxes.xyxy.cpu().numpy()
                # Sort boxes naturally: row bands first, then x-coordinate
                sorted_boxes = sorted(raw_boxes, key=lambda b: (round(b[1] / 60), b[0]))
                polys = []
                for b in sorted_boxes:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    polys.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return polys
        except Exception as exc:
            logger.warning(f"Slot detection failed: {exc}")
        return None


def get_slot_detector(model_path: Optional[Path] = None) -> AutoSlotDetector:
    return AutoSlotDetector(model_path)

