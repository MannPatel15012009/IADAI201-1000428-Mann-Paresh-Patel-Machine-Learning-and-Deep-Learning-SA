"""
ParkVision AI — Step 4: Integration & Parking Insight Logic
=================================================================
Given a full parking-lot image and the pixel coordinates of every
slot, this module:
  1. Crops and classifies each slot with the trained model.
  2. Draws colour-coded bounding-box overlays (green = empty,
     red = occupied) directly onto the image.
  3. Converts raw predictions into business insights: total /
     occupied / available counts, occupancy %, a congestion level
     (low / moderate / high) and a human-readable recommendation.

This is the layer that turns a technical model into the "smart city
solution" the assignment brief asks for.
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

import config

_model_cache = {}
_detector_cache = {}


# ------------------------------------------------------------------
# Model loading & single-slot prediction
# ------------------------------------------------------------------
def load_model(model_path=config.MODEL_PATH):
    if model_path not in _model_cache:
        _model_cache[model_path] = tf.keras.models.load_model(model_path)
    return _model_cache[model_path]


def predict_slots_batch(model, pil_image: Image.Image, boxes, batch_size=64):
    """Crop every (x, y, w, h) box, classify all crops in batches and
    return a list of (label, confidence). One model call per batch
    instead of one per slot -- much faster for big lots."""
    rgb = pil_image.convert("RGB")
    W, H = rgb.size
    crops = []
    for x, y, w, h in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = rgb.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))).resize(config.IMG_SIZE)
        crops.append(np.asarray(crop, dtype="float32"))
    if not crops:
        return []
    probs = model.predict(np.stack(crops), batch_size=batch_size, verbose=0).flatten()
    out = []
    for p in probs:
        label = "Occupied" if p >= 0.5 else "Empty"
        out.append((label, float(p if label == "Occupied" else 1 - p)))
    return out


def predict_slot(model, pil_image: Image.Image, box):
    """Single-slot convenience wrapper around predict_slots_batch."""
    return predict_slots_batch(model, pil_image, [box])[0]


# ------------------------------------------------------------------
# Automatic slot detection (YOLO) -- no JSON needed by the end user
# ------------------------------------------------------------------
def detector_available(path=config.DETECTOR_PATH):
    return os.path.exists(path)


def load_detector(path=config.DETECTOR_PATH):
    if path not in _detector_cache:
        from ultralytics import YOLO  # imported lazily: only needed for auto-detect
        _detector_cache[path] = YOLO(path)
    return _detector_cache[path]


def _reading_order(slots):
    """Number slots row by row, left to right, so ids look sensible."""
    if not slots:
        return slots
    row_h = float(np.median([s["h"] for s in slots])) or 1.0
    slots = sorted(slots, key=lambda s: (round((s["y"] + s["h"] / 2) / row_h), s["x"]))
    for i, s in enumerate(slots, start=1):
        s["id"] = i
    return slots


def detect_slots(detector, pil_image: Image.Image,
                 conf=config.DETECTOR_CONF, iou=config.DETECTOR_IOU):
    """Find every parking space in the photo. Returns the same slot
    dicts the rest of the pipeline uses: {id, x, y, w, h}.
    agnostic_nms merges the detector's empty/occupied boxes so one
    physical space is never reported twice; the MobileNetV2 classifier
    then decides Empty vs Occupied for each box."""
    rgb = pil_image.convert("RGB")
    W, H = rgb.size
    res = detector.predict(rgb, conf=conf, iou=iou, imgsz=config.DETECTOR_IMGSZ,
                           agnostic_nms=True, max_det=300, verbose=False)[0]
    slots = []
    for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy().tolist():
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 - x1 > 2 and y2 - y1 > 2:
            slots.append({"id": 0, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return _reading_order(slots)


# ------------------------------------------------------------------
# Slot coordinate helpers
# ------------------------------------------------------------------
def load_slots_from_json(json_path_or_file):
    """Load slot boxes from a JSON file/stream:
    { "slots": [ {"id": 1, "x": 10, "y": 20, "w": 80, "h": 120}, ... ] }"""
    if hasattr(json_path_or_file, "read"):
        data = json.load(json_path_or_file)
    else:
        with open(json_path_or_file) as f:
            data = json.load(f)
    return data["slots"]


def scale_slots(slots, ref_size, image_size):
    """Rescale slot boxes from the reference image size they were
    drawn on to the size of the image actually uploaded."""
    (rw, rh), (iw, ih) = ref_size, image_size
    if (rw, rh) == (iw, ih) or not rw or not rh:
        return slots
    sx, sy = iw / rw, ih / rh
    return [{**s, "x": int(s["x"] * sx), "y": int(s["y"] * sy),
             "w": int(s["w"] * sx), "h": int(s["h"] * sy)} for s in slots]


def generate_grid_slots(image_size, rows=4, cols=6, margin=10):
    """Fallback slot generator: lays an evenly spaced grid over the
    image. Useful for quick demos when ground-truth slot coordinates
    are not available. For production use, replace with real
    coordinates (see sample_lot_annotations/) or a YOLO detector."""
    W, H = image_size
    cell_w = (W - 2 * margin) / cols
    cell_h = (H - 2 * margin) / rows
    slots = []
    slot_id = 1
    for r in range(rows):
        for c in range(cols):
            x = int(margin + c * cell_w)
            y = int(margin + r * cell_h)
            slots.append({
                "id": slot_id,
                "x": x, "y": y,
                "w": int(cell_w * 0.9), "h": int(cell_h * 0.9),
            })
            slot_id += 1
    return slots


# ------------------------------------------------------------------
# Full slot-level analysis
# ------------------------------------------------------------------
def analyze_parking_lot(model, pil_image: Image.Image, slots):
    boxes = [(sl["x"], sl["y"], sl["w"], sl["h"]) for sl in slots]
    preds = predict_slots_batch(model, pil_image, boxes)
    return [{**sl, "label": lab, "confidence": round(conf, 4)}
            for sl, (lab, conf) in zip(slots, preds)]


def draw_overlays(pil_image: Image.Image, results):
    """Colour-coded bounding boxes: green = empty, red = occupied."""
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for r in results:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        color = (46, 204, 113) if r["label"] == "Empty" else (231, 76, 60)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        tag = f"{r['id']}:{r['label'][0]}"
        tag_w = 8 * len(tag) + 4
        draw.rectangle([x, max(0, y - 16), x + tag_w, y], fill=color)
        draw.text((x + 2, max(0, y - 16)), tag, fill="white", font=font)

    return annotated


# ------------------------------------------------------------------
# Business insight logic (Step 4 core requirement)
# ------------------------------------------------------------------
def compute_metrics(results):
    total = len(results)
    occupied = sum(1 for r in results if r["label"] == "Occupied")
    empty = total - occupied
    occupancy_pct = round((occupied / total) * 100, 1) if total else 0.0
    return {
        "total_slots": total,
        "occupied_slots": occupied,
        "available_slots": empty,
        "occupancy_pct": occupancy_pct,
    }


def classify_congestion(occupancy_pct):
    if occupancy_pct < config.CONGESTION_THRESHOLDS["low"]:
        return "low"
    elif occupancy_pct <= config.CONGESTION_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"


def generate_recommendation(congestion_level):
    return config.RECOMMENDATIONS[congestion_level]


def full_report(model, pil_image, slots):
    """Runs the complete Step 4 pipeline; returns everything the
    Streamlit UI (or any other client) needs to render results."""
    results = analyze_parking_lot(model, pil_image, slots)
    annotated = draw_overlays(pil_image, results)
    metrics = compute_metrics(results)
    congestion = classify_congestion(metrics["occupancy_pct"])
    recommendation = generate_recommendation(congestion)
    return {
        "results": results,
        "annotated_image": annotated,
        "metrics": metrics,
        "congestion": congestion,
        "recommendation": recommendation,
    }
import cv2
import numpy as np

def check_image_quality(pil_image, sharpness_threshold=100):
    """Check if image is too blurry. Returns True if good, False if blurry"""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(cv_image, cv2.CV_64F).var()
    return laplacian_var > sharpness_threshold