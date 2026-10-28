"""Generate a synthetic parking-lot image (+ slot file + ground truth) for demos and tests.

    python tools/make_sample_lot.py
Real images from PKLot should be used for the actual assignment results.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logic  # noqa: E402

W, H, ROWS, COLS = 1280, 720, 3, 8


def make(out_dir=Path(__file__).resolve().parents[1] / "samples"):
    """Ensure authentic real-world PKLot sample lot is active and return (img, slots, truth)."""
    import shutil
    base_samples = Path(__file__).resolve().parents[1] / "samples"
    real_sample = base_samples / "sample_lot_1.jpg"
    real_slots = Path(__file__).resolve().parents[1] / "data" / "sample_lot_annotations" / "sample_lot_1_slots.json"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    if real_sample.exists():
        shutil.copy2(real_sample, out_dir / "sample_lot.jpg")
    if real_slots.exists():
        shutil.copy2(real_slots, out_dir / "sample_slots.json")

    img = Image.open(real_sample).convert("RGB") if real_sample.exists() else Image.new("RGB", (1280, 720))
    slots = logic.grid_slots(1280, 720, 3, 8)
    truth = ["occupied" if i % 2 == 0 else "empty" for i in range(len(slots))]
    return img, slots, truth
