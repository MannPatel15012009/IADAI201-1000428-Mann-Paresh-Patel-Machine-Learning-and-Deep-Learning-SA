"""
ParkVision AI — Step 2: Data Collection & Preprocessing
============================================================
Builds a clean, balanced, resized, and split dataset from the PKLot
dataset (https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
for training a slot-level occupancy classifier (Empty vs Occupied).

WHY PKLot: it already ships as thousands of PRE-CROPPED, per-slot
images (one image = one parking space), captured across sunny, cloudy
and rainy weather, and labelled Empty/Occupied. This maps perfectly
onto the assignment's requirement to organise data into /empty and
/occupied folders and to train under real-world lighting/weather
variation.

Expected raw layout after downloading + unzipping PKLotSegmented:
    data/raw/PKLotSegmented/<LOT>/<WEATHER>/<DATE>/Empty/*.jpg
    data/raw/PKLotSegmented/<LOT>/<WEATHER>/<DATE>/Occupied/*.jpg

Usage:
    python data_preprocessing.py
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def collect_image_paths(raw_dir: str):
    """Recursively walk the raw PKLot tree and bucket images by label."""
    empty_paths, occupied_paths = [], []
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Download PKLotSegmented and unzip it into data/raw/ first "
            "(see README.md -> Dataset section)."
        )

    for path in raw_dir.rglob("*.jpg"):
        parent = path.parent.name.lower()
        if parent == "empty":
            empty_paths.append(str(path))
        elif parent == "occupied":
            occupied_paths.append(str(path))

    print(f"Found {len(empty_paths):,} Empty images and {len(occupied_paths):,} Occupied images.")
    return empty_paths, occupied_paths


def balanced_sample(paths, n, seed=config.SEED):
    """Randomly sample n images so both classes are perfectly balanced."""
    random.seed(seed)
    if len(paths) < n:
        raise ValueError(f"Not enough images ({len(paths)}) available to sample {n}.")
    return random.sample(paths, n)


def resize_and_save(src_path, dst_path, size=config.IMG_SIZE):
    """Load, convert to RGB, resize to a fixed dimension, and save."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)
        return True
    except Exception as e:  # corrupt / unreadable image -> skip gracefully
        print(f"[WARN] Skipping corrupt image {src_path}: {e}")
        return False


def split_and_organize(empty_paths, occupied_paths):
    """70/15/15 stratified split per class, resized copies written to
    data/processed/{train,val,test}/{empty,occupied}/."""
    for label, paths in [("empty", empty_paths), ("occupied", occupied_paths)]:
        train_paths, temp_paths = train_test_split(
            paths, train_size=config.TRAIN_SPLIT, random_state=config.SEED
        )
        rel_val = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        val_paths, test_paths = train_test_split(
            temp_paths, train_size=rel_val, random_state=config.SEED
        )

        for split_name, split_paths in [
            ("train", train_paths), ("val", val_paths), ("test", test_paths)
        ]:
            out_dir = os.path.join(config.PROCESSED_DATA_DIR, split_name, label)
            print(f"  -> {split_name}/{label}: {len(split_paths)} images")
            saved = 0
            for i, src in enumerate(tqdm(split_paths, desc=f"{split_name}/{label}", leave=False)):
                dst = os.path.join(out_dir, f"{label}_{i:05d}.jpg")
                if resize_and_save(src, dst):
                    saved += 1
            print(f"     saved {saved}/{len(split_paths)}")


def print_summary():
    print("\nFinal processed dataset:")
    for split in ["train", "val", "test"]:
        for label in ["empty", "occupied"]:
            d = os.path.join(config.PROCESSED_DATA_DIR, split, label)
            n = len(list(Path(d).glob("*.jpg"))) if os.path.exists(d) else 0
            print(f"  {split:>5}/{label:<9}: {n:>5} images")


def main():
    print("=" * 65)
    print("ParkVision AI — Data Collection & Preprocessing (Step 2)")
    print("=" * 65)

    if os.path.exists(config.PROCESSED_DATA_DIR):
        print(f"Clearing existing processed data at {config.PROCESSED_DATA_DIR} ...")
        shutil.rmtree(config.PROCESSED_DATA_DIR)

    empty_paths, occupied_paths = collect_image_paths(config.RAW_DATA_DIR)

    # Balance both classes; use at least IMAGES_PER_CLASS, or the max
    # available if the raw pool is smaller (with a hard floor check).
    n = min(len(empty_paths), len(occupied_paths))
    if n < config.IMAGES_PER_CLASS:
        raise ValueError(
            f"Only {n} images available per class; the brief requires at least "
            f"{config.IMAGES_PER_CLASS}. Download more PKLot data or lower "
            f"IMAGES_PER_CLASS in config.py (not recommended)."
        )
    n = min(n, config.IMAGES_PER_CLASS)  
    print(f"\nSampling {n} balanced images per class...")

    empty_sample = balanced_sample(empty_paths, n)
    occupied_sample = balanced_sample(occupied_paths, n)

    print("\nResizing to", config.IMG_SIZE, "| splitting 70/15/15 | organizing into folders...")
    split_and_organize(empty_sample, occupied_sample)

    print_summary()
    print("\nDone. Dataset ready at:", config.PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
