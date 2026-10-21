"""
ParkVision AI — Roboflow -> PKLotSegmented-style converter
================================================================
If you downloaded PKLot from Roboflow (public.roboflow.com/object-
detection/pklot) instead of the original UFPR server, you get FULL
parking-lot photos + YOLO bounding-box labels (classes: space-empty,
space-occupied) rather than pre-cropped per-slot images.

This script crops every labelled slot out of every image and saves it
into the same Empty/Occupied folder layout that data_preprocessing.py
expects, so the rest of the pipeline needs zero changes.

Expected Roboflow export layout (YOLOv8 format):
    <export_root>/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Usage:
    python prepare_from_roboflow.py --export_root path/to/roboflow_export
"""

import argparse
import os
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

import config


def load_class_names(export_root):
    with open(os.path.join(export_root, "data.yaml")) as f:
        data = yaml.safe_load(f)
    names = data["names"]  # e.g. ['space-empty', 'space-occupied']
    return names


def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    return int(x), int(y), int(w * img_w), int(h * img_h)


def label_folder_name(class_name):
    name = class_name.lower()
    if "occup" in name:
        return "Occupied"
    return "Empty"


def process_split(export_root, split, class_names, out_root):
    img_dir = Path(export_root) / split / "images"
    lbl_dir = Path(export_root) / split / "labels"
    if not img_dir.exists():
        print(f"[skip] {img_dir} not found")
        return 0

    count = 0
    for img_path in tqdm(list(img_dir.glob("*.*")), desc=f"Cropping {split}"):
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x, y, bw, bh = yolo_to_pixel_box(cx, cy, w, h, img_w, img_h)
            if bw <= 2 or bh <= 2:
                continue  # skip degenerate boxes

            crop = img.crop((max(0, x), max(0, y), x + bw, y + bh))
            label_folder = label_folder_name(class_names[cls_id])
            out_dir = Path(out_root) / label_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"{img_path.stem}_{i:03d}.jpg", quality=95)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_root", required=True,
                         help="Path to the unzipped Roboflow export folder")
    parser.add_argument("--out_root", default=os.path.join(config.RAW_DATA_DIR, "PKLotSegmented_fromRoboflow"),
                         help="Where to write the Empty/Occupied crop folders")
    args = parser.parse_args()

    class_names = load_class_names(args.export_root)
    print("Classes found in data.yaml:", class_names)

    total = 0
    for split in ["train", "valid", "test"]:
        total += process_split(args.export_root, split, class_names, args.out_root)

    print(f"\nDone. Wrote {total} cropped slot images to:\n  {args.out_root}/Empty\n  {args.out_root}/Occupied")
    print("\nNext step: run `python data_preprocessing.py` as normal -- "
          "it will pick these up automatically since RAW_DATA_DIR is scanned recursively.")


if __name__ == "__main__":
    main()
