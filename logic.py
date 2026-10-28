"""Slot geometry, parking insight logic and overlay drawing (no TensorFlow needed)."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

import config

EMPTY, OCCUPIED = "empty", "occupied"
COLOR_EMPTY = (0, 200, 0)      # green  (RGB)
COLOR_OCCUPIED = (220, 30, 30)  # red    (RGB)

Polygon = list  # list of (x, y) points


# --------------------------------------------------------------------------
# Insight logic (Step 4)
# --------------------------------------------------------------------------
def summarize(statuses):
    """Return total / occupied / available / occupancy % for a list of statuses."""
    total = len(statuses)
    occupied = sum(1 for s in statuses if s == OCCUPIED)
    pct = 100.0 * occupied / total if total else 0.0
    return {
        "total": total,
        "occupied": occupied,
        "available": total - occupied,
        "occupancy_pct": round(pct, 1),
    }


def congestion_level(pct):
    """Low (<40 %), Moderate (40-75 %), High (>75 %)."""
    if pct < config.LOW_CONGESTION_MAX:
        return "Low"
    if pct <= config.HIGH_CONGESTION_MIN:
        return "Moderate"
    return "High"


def recommendation(summary):
    """Return (message, severity) where severity is success / warning / error."""
    if summary["total"] == 0:
        return "No parking slots to analyse.", "warning"
    if summary["available"] == 0:
        return "Parking full - try another area.", "error"
    level = congestion_level(summary["occupancy_pct"])
    if level == "High":
        return (f"Only {summary['available']} slot(s) left - parking is almost full, "
                "consider trying another area."), "warning"
    return f"Slots available ({summary['available']} free) - proceed.", "success"


def classify(probs, threshold=config.DEFAULT_THRESHOLD):
    """Turn P(occupied) values into 'empty' / 'occupied' labels."""
    return [OCCUPIED if p >= threshold else EMPTY for p in probs]


# --------------------------------------------------------------------------
# Slot layouts
# --------------------------------------------------------------------------
def grid_slots(width, height, rows, cols, margin_x=0.04, margin_y=0.08, gap=6):
    """Evenly split the image into rows x cols rectangular slots (4-point polygons)."""
    x0, x1 = width * margin_x, width * (1 - margin_x)
    y0, y1 = height * margin_y, height * (1 - margin_y)
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    slots = []
    for r in range(rows):
        for c in range(cols):
            a, b = x0 + c * cw + gap / 2, y0 + r * ch + gap / 2
            e, f = x0 + (c + 1) * cw - gap / 2, y0 + (r + 1) * ch - gap / 2
            slots.append([(int(a), int(b)), (int(e), int(b)), (int(e), int(f)), (int(a), int(f))])
    return slots


def _scale(poly, sx, sy):
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in poly]


def parse_slots_file(data: bytes, filename: str, image_size, xml_ref_size=(1280, 720)):
    """Read slot polygons from a PKLot XML file or a JSON file.

    JSON formats accepted:  [[x1,y1,x2,y2], ...]  |  [[[x,y],[x,y],...], ...]
    or {"image_width":W, "image_height":H, "slots":[...]} (coordinates are rescaled).
    PKLot XML coordinates refer to 1280x720 images; they are rescaled to the upload.
    """
    W, H = image_size
    name = filename.lower()
    slots = []
    if name.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        sx = sy = 1.0
        if isinstance(obj, dict):
            if obj.get("image_width") and obj.get("image_height"):
                sx, sy = W / obj["image_width"], H / obj["image_height"]
            obj = obj.get("slots", [])
        for item in obj:
            if isinstance(item, dict) and "x" in item and "y" in item:
                x, y, w, h = item["x"], item["y"], item.get("w", 0), item.get("h", 0)
                poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            elif len(item) == 4 and not isinstance(item[0], (list, tuple)):
                x1, y1, x2, y2 = item
                poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                poly = [tuple(p) for p in item]
            slots.append(_scale(poly, sx, sy))
    elif name.endswith(".xml"):
        root = ET.fromstring(data)
        sx, sy = W / xml_ref_size[0], H / xml_ref_size[1]
        for space in root.iter("space"):
            pts = [(float(p.get("x")), float(p.get("y"))) for p in space.findall("./contour/point")]
            if len(pts) < 3:
                rr = space.find("rotatedRect")
                if rr is None:
                    continue
                cx, cy = float(rr.find("center").get("x")), float(rr.find("center").get("y"))
                w, h = float(rr.find("size").get("w")), float(rr.find("size").get("h"))
                ang = math.radians(float(rr.find("angle").get("d")))
                pts = []
                for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
                    pts.append((cx + dx * math.cos(ang) - dy * math.sin(ang),
                                cy + dx * math.sin(ang) + dy * math.cos(ang)))
            slots.append(_scale(pts, sx, sy))
    else:
        raise ValueError("Slot file must be .json or .xml")
    if not slots:
        raise ValueError("No slots found in the file.")
    return slots


def polygon_bbox(poly, width, height):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(width, max(xs)), min(height, max(ys))
    return int(x1), int(y1), int(x2), int(y2)


def crop_slots(image: Image.Image, slots):
    """Crop every slot (bounding box of its polygon). Degenerate boxes become None."""
    W, H = image.size
    crops = []
    for poly in slots:
        x1, y1, x2, y2 = polygon_bbox(poly, W, H)
        crops.append(image.crop((x1, y1, x2, y2)) if x2 - x1 >= 4 and y2 - y1 >= 4 else None)
    return crops


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------
def draw_overlay(image: Image.Image, slots, statuses, probs=None, show_ids=True):
    """Return an RGB numpy array with green (empty) / red (occupied) slot outlines."""
    canvas = np.array(image.convert("RGB"))
    thick = max(2, int(round(min(canvas.shape[:2]) / 250)))
    scale = max(0.4, min(canvas.shape[:2]) / 1100)

    # Single-pass overlay mask to eliminate O(N) full-resolution image copies
    overlay = canvas.copy()
    polys_by_color = {COLOR_EMPTY: [], COLOR_OCCUPIED: []}
    parsed_pts = []

    for poly, status in zip(slots, statuses):
        color = COLOR_OCCUPIED if status == OCCUPIED else COLOR_EMPTY
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        polys_by_color[color].append(pts)
        parsed_pts.append((pts, color))

    for color, poly_list in polys_by_color.items():
        if poly_list:
            cv2.fillPoly(overlay, poly_list, color)

    # Single composite alpha blend
    canvas = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)

    # Crisp outlines and slot ID numbers
    for i, (pts, color) in enumerate(parsed_pts):
        cv2.polylines(canvas, [pts], True, color, thick)
        if show_ids:
            x = pts[:, 0, 0].min() + 3
            y = pts[:, 0, 1].min() + int(28 * scale) + 2
            text = str(i + 1)
            cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
            cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick - 1 if thick > 2 else 1, cv2.LINE_AA)
    return canvas
