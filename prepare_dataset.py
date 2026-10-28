"""Step 2 - build a balanced, resized 70/15/15 dataset from raw labelled images.

Works with any folder tree where images sit under folders called `empty` / `occupied`
(case-insensitive) - including PKLotSegmented (lot/weather/date/Empty|Occupied/*.jpg).

    python prepare_dataset.py --source raw_dataset --max-per-class 300 --clean
    python prepare_dataset.py --source raw_dataset --group-split      # avoid near-duplicate leakage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import config

EXTS = {".jpg", ".jpeg", ".png"}
WEATHER = {"sunny", "cloudy", "rainy"}


@dataclass
class Item:
    path: Path
    label: str
    group: str      # folder that holds the class folder (e.g. one day of one lot)
    weather: str


def collect(source: Path):
    items = {"empty": [], "occupied": []}
    for p in source.rglob("*"):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        label_dir = next((a for a in p.parents if a.name.lower() in items), None)
        if label_dir is None:
            continue
        weather = next((part.lower() for part in p.parts if part.lower() in WEATHER), "unknown")
        items[label_dir.name.lower()].append(Item(p, label_dir.name.lower(), str(label_dir.parent), weather))
    return items


def round_robin(buckets):
    iters = [iter(b) for b in buckets.values()]
    while iters:
        for it in list(iters):
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)


def sample_class(items, n, rng):
    """Pick n unique (by content hash) images, spread evenly over weather conditions."""
    buckets = defaultdict(list)
    for it in items:
        buckets[it.weather].append(it)
    for b in buckets.values():
        rng.shuffle(b)
    chosen, seen = [], set()
    for it in round_robin(buckets):
        digest = hashlib.sha256(it.path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        try:
            with Image.open(it.path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            continue
        seen.add(digest)
        it.digest = digest
        chosen.append(it)
        if len(chosen) >= n:
            break
    return chosen


def random_split(items, ratios, rng):
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_tr, n_va = round(ratios[0] * n), round(ratios[1] * n)
    return [items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]]


def group_split(items, ratios, rng):
    """Whole groups (same lot/day) go to one split so near-identical frames never straddle splits."""
    groups = defaultdict(list)
    for it in items:
        groups[it.group].append(it)
    keys = list(groups)
    rng.shuffle(keys)
    targets = [r * len(items) for r in ratios]
    out = [[], [], []]
    for k in keys:
        i = max(range(3), key=lambda j: targets[j] - len(out[j]))
        out[i].extend(groups[k])
    return out


def save_image(item, dest: Path, size):
    with Image.open(item.path) as im:
        im.convert("RGB").resize(size, Image.BILINEAR).save(dest, format="JPEG", quality=95)


REAL_PKLOT_RAW = config.BASE_DIR / "data" / "raw" / "PKLotSegmented_fromRoboflow"


def prepare(source: Path, out_dir: Path, max_per_class=300, min_per_class=100,
            ratios=config.SPLIT_RATIOS, use_groups=False, seed=config.SEED, clean=False,
            size=config.IMG_SIZE, summary_path=None):
    rng = random.Random(seed)
    items = collect(source)
    counts = {k: len(v) for k, v in items.items()}

    # If source is empty, check real PKLot raw directory
    if min(counts.values(), default=0) < min_per_class and REAL_PKLOT_RAW.exists():
        print(f"Checking real PKLot dataset at {REAL_PKLOT_RAW}...")
        items = collect(REAL_PKLOT_RAW)
        counts = {k: len(v) for k, v in items.items()}

    print(f"Found real images: {counts}")
    if min(counts.values(), default=0) < min_per_class:
        raise FileNotFoundError(
            f"ERROR: Real PKLot dataset not found in '{source}' or '{REAL_PKLOT_RAW}'. "
            f"Expected at least {min_per_class} real photos per class (empty and occupied)."
        )
    n = min(max_per_class, *counts.values())
    if n < min_per_class:
        print(f"WARNING: only {n} images/class available (assignment asks for at least {min_per_class}).")
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    splits = {"train": {}, "val": {}, "test": {}}
    for label in items:
        picked = sample_class(items[label], n, rng)
        parts = (group_split if use_groups else random_split)(picked, ratios, rng)
        if any(len(p) == 0 for p in parts):
            print(f"Notice: group-split resulted in an empty split for '{label}'; falling back to stratified random split.")
            parts = random_split(picked, ratios, rng)
        for name, part in zip(splits, parts):
            splits[name][label] = part
    summary = {s: {l: len(v) for l, v in d.items()} for s, d in splits.items()}

    for name, per_label in splits.items():
        for label, part in per_label.items():
            folder = out_dir / name / label
            folder.mkdir(parents=True, exist_ok=True)
            for it in part:
                save_image(it, folder / f"{label}_{it.digest[:12]}.jpg", size)
    result = {"images_per_class_used": n, "available": counts, "split": summary,
              "size": list(size), "group_split": use_groups, "seed": seed}
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--min-per-class", type=int, default=100)
    ap.add_argument("--group-split", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete the output folder first")
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    prepare(a.source, a.out, a.max_per_class, a.min_per_class, config.SPLIT_RATIOS,
            a.group_split, a.seed, a.clean, config.IMG_SIZE, config.DATASET_SUMMARY_PATH)
