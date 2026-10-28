"""Small Streamlit tool to add correctly-labelled slot crops to dataset/<split>/<class>/.

    streamlit run upload_labeler.py

(For local use only - files saved on Streamlit Cloud disappear when the app restarts.)
"""
import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

import config

st.set_page_config(page_title="ParkVision AI - labeler", page_icon="🅿️")
st.title("🅿️ ParkVision AI - add labelled images")
st.write("Upload **cropped images showing one parking slot each**, pick the correct label and split.")

label = st.radio("Correct label", ["Empty", "Occupied"], horizontal=True)
split = st.selectbox("Save into split", ["train", "val", "test"],
                     help="Keep val/test images different from training images to avoid leakage.")
uploads = st.file_uploader("Choose image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


def known_hashes():
    """Content hashes of everything already in the dataset (all splits and classes)."""
    out = set()
    for s in ("train", "val", "test"):
        for c in config.CLASS_NAMES:
            for p in (config.DATASET_DIR / s / c).glob("*.jpg"):
                out.add(p.stem.split("_")[-1])
    return out


def save(raw, name):
    digest = hashlib.sha256(raw).hexdigest()
    if digest[:12] in known_hashes():
        return False, "Duplicate - this image is already in the dataset."
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return False, f"{name} is not a valid image."
    folder = config.DATASET_DIR / split / label.lower()
    folder.mkdir(parents=True, exist_ok=True)
    img.resize(config.IMG_SIZE, Image.BILINEAR).save(folder / f"{label.lower()}_{digest[:12]}.jpg", quality=95)
    return True, f"Saved to dataset/{split}/{label.lower()}/"


if uploads:
    if st.button(f"Save all {len(uploads)} image(s) as {label} -> {split}", type="primary"):
        for up in uploads:
            ok, msg = save(up.getvalue(), up.name)
            (st.success if ok else st.warning)(f"{up.name}: {msg}")
    cols = st.columns(4)
    for i, up in enumerate(uploads):
        try:
            cols[i % 4].image(Image.open(BytesIO(up.getvalue())), caption=up.name)
        except (UnidentifiedImageError, OSError):
            cols[i % 4].error(f"{up.name}: invalid image")

st.divider()
st.subheader("Current image counts")
rows = []
for s in ("train", "val", "test"):
    rows.append({"split": s, **{c: len(list((config.DATASET_DIR / s / c).glob("*.jpg"))) for c in config.CLASS_NAMES}})
st.table(rows)
st.caption("Check every label carefully - wrong labels teach the model the wrong pattern.")
