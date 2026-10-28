"""Extracts fresh, non-redundant image batches from PKLotSegmented_fromRoboflow
and packages them into balanced 70/15/15 chunks for progressive continual training.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger("PrepareExtended")

RAW_SEGMENTED_DIR = config.DATA_DIR / "raw" / "PKLotSegmented_fromRoboflow"
if not RAW_SEGMENTED_DIR.exists():
    fallback = Path("IDAI201-1000428-Mann-Paresh-Patel-Machine-Learning-and-Deep-Learning-SA/data/raw/PKLotSegmented_fromRoboflow")
    if fallback.exists():
        RAW_SEGMENTED_DIR = fallback


def get_existing_filenames() -> set[str]:
    """Collect filenames already trained in data/processed to avoid redundancy."""
    existing = set()
    data_dir = config.DATA_DIR / "processed"
    if data_dir.exists():
        for p in data_dir.rglob("*.jpg"):
            existing.add(p.name)
            # Also add stem in case of naming prefixes
            existing.add(p.stem)
    return existing


def export_chunk(
    output_zip: Path,
    chunk_size: int = 10000,
    offset: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> int:
    """Export a balanced, non-redundant chunk in 70/15/15 ratio directly into a zip file."""
    empty_dir = RAW_SEGMENTED_DIR / "Empty"
    occupied_dir = RAW_SEGMENTED_DIR / "Occupied"

    if not empty_dir.exists() or not occupied_dir.exists():
        logger.error(f"Source segmented directory missing at {RAW_SEGMENTED_DIR}")
        return 0

    existing = get_existing_filenames()
    logger.info(f"Existing dataset has {len(existing)} images registered. Ensuring ZERO redundancy.")

    per_class = chunk_size // 2
    n_train = int(per_class * ratios[0])
    n_val = int(per_class * ratios[1])
    n_test = per_class - n_train - n_val

    logger.info(f"Chunk Target: {chunk_size} total images ({per_class} per class: {n_train} train, {n_val} val, {n_test} test)")

    temp_chunk_dir = Path("temp_chunk")
    if temp_chunk_dir.exists():
        shutil.rmtree(temp_chunk_dir)

    for split in ["train", "val", "test"]:
        for cls in ["empty", "occupied"]:
            (temp_chunk_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def collect_class(src_dir: Path, target_count: int, skip_count: int, cls_name: str):
        collected = []
        skipped = 0
        for entry in os.scandir(src_dir):
            if not entry.name.lower().endswith((".jpg", ".png")):
                continue
            if entry.name in existing or Path(entry.name).stem in existing:
                continue
            if skipped < skip_count:
                skipped += 1
                continue
            collected.append(entry.path)
            if len(collected) >= target_count:
                break
        return collected

    logger.info(f"Collecting empty images (offset {offset // 2})...")
    empty_files = collect_class(empty_dir, per_class, offset // 2, "empty")
    logger.info(f"Collecting occupied images (offset {offset // 2})...")
    occ_files = collect_class(occupied_dir, per_class, offset // 2, "occupied")

    actual_per_class = min(len(empty_files), len(occ_files))
    if actual_per_class == 0:
        logger.warning("No new images available to export.")
        return 0

    def distribute(files, cls_name):
        tr = files[:n_train]
        va = files[n_train:n_train + n_val]
        te = files[n_train + n_val:n_train + n_val + n_test]

        for s_name, s_list in [("train", tr), ("val", va), ("test", te)]:
            dst = temp_chunk_dir / s_name / cls_name
            for src_p in s_list:
                shutil.copy2(src_p, dst / Path(src_p).name)

    distribute(empty_files[:actual_per_class], "empty")
    distribute(occ_files[:actual_per_class], "occupied")

    # Zip temp_chunk into output_zip
    logger.info(f"Compressing into {output_zip}...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in temp_chunk_dir.rglob("*"):
            if f.is_file():
                arcname = str(f.relative_to(temp_chunk_dir))
                zf.write(f, arcname=f"dataset/{arcname}")

    shutil.rmtree(temp_chunk_dir)
    total_packed = actual_per_class * 2
    logger.info(f"Created {output_zip} ({output_zip.stat().st_size / (1024*1024):.2f} MB) containing {total_packed} images.")
    return total_packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package non-redundant PKLot images into progressive chunks")
    parser.add_argument("--out", type=Path, default=Path("data_extended_10k.zip"), help="Output zip filename")
    parser.add_argument("--size", type=int, default=10000, help="Number of images in this chunk")
    parser.add_argument("--offset", type=int, default=0, help="Number of images to skip (for multiple chunks)")
    args = parser.parse_args()

    export_chunk(args.out, chunk_size=args.size, offset=args.offset)
