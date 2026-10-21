"""
ParkVision AI — Build a real slots.json from an existing Roboflow label
=============================================================================
You already have exact per-slot bounding-box coordinates for thousands of
real parking-lot photos sitting in your Roboflow export (the same labels
`prepare_from_roboflow.py` used to crop training images). This script picks
ONE full lot photo + its matching label file and converts it directly into
the slots.json format the Streamlit dashboard expects -- no manual
measuring needed.

It also copies the chosen image into sample_images/ so it's ready to
upload straight into the app.

Usage:
    python generate_slots_from_yolo.py --image_path data/raw/test/images/XXXX.jpg

(The matching label file is found automatically -- same filename, .txt
extension, inside the sibling "labels" folder.)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

import config


def find_label_path(image_path: Path):
    """Given .../images/NAME.jpg, return .../labels/NAME.txt"""
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / (image_path.stem + ".txt")


def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    return int(x), int(y), int(w * img_w), int(h * img_h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True,
                         help="Path to ONE full lot photo from your Roboflow export, "
                              "e.g. data/raw/test/images/some_image.jpg")
    parser.add_argument("--out_name", default="sample_lot_1",
                         help="Base filename (no extension) for the saved image + json")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_path = find_label_path(image_path)
    if not label_path.exists():
        raise FileNotFoundError(
            f"Could not find a matching label file at {label_path}. "
            "Make sure --image_path points at a file inside an 'images' folder "
            "that has a sibling 'labels' folder with a same-named .txt file."
        )

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    slots = []
    for i, line in enumerate(lines, start=1):
        parts = line.split()
        cx, cy, w, h = map(float, parts[1:5])
        x, y, bw, bh = yolo_to_pixel_box(cx, cy, w, h, img_w, img_h)
        if bw <= 2 or bh <= 2:
            continue
        slots.append({"id": i, "x": x, "y": y, "w": bw, "h": bh})

    if not slots:
        raise ValueError("No valid slots found in this label file -- try a different image.")

    # Save outputs
    os.makedirs("sample_images", exist_ok=True)
    os.makedirs(config.SAMPLE_ANNOTATIONS_DIR, exist_ok=True)

    out_image_path = os.path.join("sample_images", f"{args.out_name}.jpg")
    out_json_path = os.path.join(config.SAMPLE_ANNOTATIONS_DIR, f"{args.out_name}_slots.json")

    shutil.copy(image_path, out_image_path)
    with open(out_json_path, "w") as f:
        json.dump({
            "description": f"Real slot coordinates extracted from {image_path.name}'s Roboflow label.",
            "image_reference": out_image_path,
            "image_width": img_w,
            "image_height": img_h,
            "slots": slots,
        }, f, indent=2)

    print(f"Found {len(slots)} real parking slots in this image.")
    print(f"Saved image -> {out_image_path}")
    print(f"Saved slots.json -> {out_json_path}")
    print("\nNext step: run `streamlit run app.py`, choose "
          "'Upload slot-coordinates JSON', upload this image AND this json file.")


if __name__ == "__main__":
    main()
