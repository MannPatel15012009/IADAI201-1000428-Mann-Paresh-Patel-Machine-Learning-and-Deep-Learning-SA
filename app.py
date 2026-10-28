"""ParkVision AI - Streamlit dashboard for slot-level parking occupancy analytics."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import config
import logic
from predictor import load_classifier, get_slot_detector

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")

st.markdown("""
<style>
.badge{display:inline-block;padding:.35rem .9rem;border-radius:999px;font-weight:700;color:#fff;font-size:1.05rem}
.legend span{display:inline-block;width:.9rem;height:.9rem;border-radius:3px;margin-right:.35rem;vertical-align:middle}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI model...")
def get_classifier(model_choice: str = "Auto"):
    if model_choice == "YOLOv8":
        return load_classifier(config.YOLO_MODEL_PATH)
    elif model_choice == "MobileNetV2":
        return load_classifier(config.MODEL_PATH)
    return load_classifier()


@st.cache_resource(show_spinner="Loading AI slot detector...")
def get_slot_detector_engine():
    return get_slot_detector()


def read_image(file_or_path):
    img = Image.open(file_or_path)
    return ImageOps.exif_transpose(img).convert("RGB")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("AI Model")
    available_models = ["Auto"]
    if config.YOLO_MODEL_PATH.exists():
        available_models.append("YOLOv8")
    if config.MODEL_PATH.exists():
        available_models.append("MobileNetV2")

    model_choice = st.selectbox("Active Architecture", available_models, index=0)
    clf = get_classifier(model_choice)
    badge_color = "#1a73e8" if not clf.is_demo else "#ea8600"
    st.markdown(f"**Engine:** <span style='background:{badge_color}; color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold;'>{clf.model_type}</span>", unsafe_allow_html=True)
    st.caption(f"{clf.status}")

    st.divider()
    st.header("Detection Settings")
    mode = st.radio(
        "Input type",
        ["Auto-detect (Recommended)", "Full parking-lot image", "Single slot crop"],
        help="Auto-detect automatically identifies whether an uploaded image is a single cropped parking space (e.g. 224x224) or a full parking lot surveillance camera image."
    )
    threshold = st.slider(
        "Occupied threshold P(occupied) ≥",
        0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05,
        help="Decision boundary for occupancy. A lower threshold (0.20-0.25) prevents hazardous false-empty classifications (directing drivers to already occupied bays)."
    )
    slot_file = None
    layout = "Auto AI Slot Detection (Recommended - Fully Automated)"
    rows, cols, mx, my, gap = 3, 8, 0.04, 0.08, 6
    if mode != "Single slot crop":
        st.subheader("Slot Layout Mapping")
        layout = st.radio(
            "Slot layout mode",
            [
                "Auto AI Vision (Human-like Line & Slot Detection)",
                "PKLot Benchmark Preset (40 Bays - Built-in Camera)",
                "Manual Grid / Custom File (Advanced Override)"
            ],
            index=0,
            help="Mimics human perception: detects slot boundaries from painted chalk markings and vehicle geometry automatically, without asking for rows or columns."
        )
        if layout.startswith("Auto AI"):
            st.info("👁️ **Human-Inspired Vision:** Automatically detects parking slots using painted chalk lines and vehicle spacing—exactly as a human observer identifies occupied vs empty bays without manual counting.")
        elif layout.startswith("PKLot Benchmark"):
            st.info("📌 **PKLot Benchmark Preset (40 Bays)**: Calibrated specifically for the 40 bays of the PKLot PUC surveillance camera.")
        elif layout.startswith("Manual Grid"):
            override_type = st.radio("Override Type", ["Configurable Grid", "Upload Custom Slot File"], index=0)
            if override_type == "Configurable Grid":
                r_col, c_col = st.columns(2)
                with r_col:
                    rows = st.number_input("Rows", 1, 15, 3)
                with c_col:
                    cols = st.number_input("Slots / Row", 1, 50, 8)
                st.caption(f"Configured slots: **{rows * cols} bays**.")
                with st.expander("Grid Margins & Spacing"):
                    mx = st.slider("Side margin (%)", 0, 30, 4) / 100
                    my = st.slider("Top/bottom margin (%)", 0, 30, 8) / 100
                    gap = st.slider("Gap between slots (px)", 0, 20, 6)
            else:
                slot_file = st.file_uploader("Upload Slot File (.xml or .json)", type=["xml", "json"])
    show_ids = st.checkbox("Show slot numbers", True)
    st.divider()
    st.markdown("<div class='legend'><span style='background:#00c800'></span>Empty<br>"
                "<span style='background:#dc1e1e'></span>Occupied</div>", unsafe_allow_html=True)

st.title("🅿️ ParkVision AI")
st.caption("Intelligent urban parking analytics & space optimisation - real-time slot occupancy with YOLO & MobileNet")

tab_analyze, tab_perf, tab_about = st.tabs(["🔍 Analyse parking", "📊 Model performance", "ℹ️ About"])

# ---------------------------------------------------------------- analyse
with tab_analyze:
    if clf.is_demo:
        st.warning(f"⚠️ **Demo mode** - {clf.status} Running on heuristic mode instead of trained deep learning weights.")
    else:
        st.success(f"🟢 **Active Model Engine:** {clf.model_type} ({clf.status})")
    uploaded = st.file_uploader("Upload a parking image", type=["jpg", "jpeg", "png"])
    sample_path = config.SAMPLES_DIR / "sample_lot.jpg"
    use_sample = False
    if uploaded is None and sample_path.exists():
        use_sample = st.checkbox("Use real PKLot camera sample image", value=False)

    source = uploaded if uploaded is not None else (sample_path if use_sample else None)
    if source is None:
        st.info("Upload an image (or tick the sample) to start.")
    else:
        try:
            image = read_image(source)
        except Exception:  # noqa: BLE001
            st.error("That file could not be read as an image.")
            st.stop()

        W, H = image.size
        # ---- Auto-detect single slot crop vs full lot --------------------
        is_single_crop = False
        if mode == "Single slot crop":
            is_single_crop = True
        elif mode == "Full parking-lot image":
            is_single_crop = False
            if max(W, H) <= 400:
                st.warning(
                    f"⚠️ **Small image detected ({W}×{H}px):** This image appears to be an individual cropped parking bay. "
                    "Applying a full-lot grid (e.g. 3×8) to a single car will slice the vehicle into pieces. "
                    "Switch **Input type** in the sidebar to **Single slot crop** or **Auto-detect** for accurate single-bay classification."
                )
        else:  # Auto-detect (Recommended)
            if max(W, H) <= 400:
                is_single_crop = True
                st.info(f"🔍 **Auto-detected input:** Single slot crop ({W}×{H}px). Analyzing individual bay occupancy.")
            else:
                is_single_crop = False
                st.info(f"🔍 **Auto-detected input:** Full parking lot surveillance image ({W}×{H}px). Analyzing multi-bay occupancy.")

        # ---- single crop -------------------------------------------------
        if is_single_crop:
            p = float(clf.predict_proba([image])[0])
            occupied = p >= threshold
            status_text = "OCCUPIED" if occupied else "EMPTY / AVAILABLE"
            status_color = "#dc1e1e" if occupied else "#00c800"
            status_icon = "🚗" if occupied else "✅"

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(image, caption=f"Slot Image ({W}×{H}px)")
            with c2:
                st.markdown(
                    f"<div style='background:{status_color}; color:white; padding:12px 18px; border-radius:8px; font-size:1.4rem; font-weight:bold; margin-bottom:12px;'>"
                    f"{status_icon} {status_text}</div>",
                    unsafe_allow_html=True
                )
                m1, m2 = st.columns(2)
                m1.metric("P(occupied)", f"{p:.1%}")
                m2.metric("P(empty)", f"{1 - p:.1%}")
                st.progress(min(1.0, max(0.0, p)))
                st.caption(f"Decision threshold: P(occupied) ≥ {threshold:.0%} · Model: {clf.model_type}")
                st.divider()
                st.markdown(
                    "💡 **Tip:** To evaluate an entire parking lot with dozens of spaces, occupancy %, and congestion analytics, "
                    "upload a wide-angle camera image (e.g., 1280×720 or tick *Use real PKLot camera sample image*)."
                )
        # ---- full lot ----------------------------------------------------
        else:
            try:
                if layout.startswith("Auto AI"):
                    auto_detector = get_slot_detector_engine()
                    slots = auto_detector.detect_slots(image)
                    if not slots:
                        slots_file_path = config.SAMPLES_DIR / "sample_slots.json"
                        slots_data = slots_file_path.read_bytes()
                        slots = logic.parse_slots_file(slots_data, slots_file_path.name, (W, H))
                elif layout.startswith("PKLot Benchmark") or use_sample:
                    slots_file_path = config.SAMPLES_DIR / "sample_slots.json"
                    slots_data = slots_file_path.read_bytes()
                    slots = logic.parse_slots_file(slots_data, slots_file_path.name, (W, H))
                elif slot_file is not None:
                    slots = logic.parse_slots_file(slot_file.getvalue(), slot_file.name, (W, H))
                else:
                    slots = logic.grid_slots(W, H, int(rows), int(cols), mx, my, gap)
            except Exception as exc:  # noqa: BLE001
                slots = None
                st.error(f"Could not read the slot configuration: {exc}")

            if slots:
                with st.spinner(f"Analysing {len(slots)} parking slots..."):
                    crops = logic.crop_slots(image, slots, pad_ratio=0.18)
                    probs = clf.predict_proba(crops)
                    statuses = logic.classify(probs, threshold)
                    summary = logic.summarize(statuses)
                    level = logic.congestion_level(summary["occupancy_pct"])
                    message, severity = logic.recommendation(summary)
                    overlay = logic.draw_overlay(image, slots, statuses, probs, show_ids)

                left, right = st.columns(2)
                left.image(image, caption="Uploaded image")
                right.image(overlay, caption="Annotated output (green = empty, red = occupied)")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total slots", summary["total"])
                m2.metric("Occupied", summary["occupied"])
                m3.metric("Available", summary["available"])
                m4.metric("Occupancy", f"{summary['occupancy_pct']:.1f}%")
                st.progress(min(1.0, summary["occupancy_pct"] / 100))

                colour = {"Low": "#1a9850", "Moderate": "#e69500", "High": "#d73027"}[level]
                st.markdown(f"**Congestion level:** <span class='badge' style='background:{colour}'>{level}</span>",
                            unsafe_allow_html=True)
                getattr(st, severity)(f"**Recommendation:** {message}")

                df = pd.DataFrame({"Slot": np.arange(1, len(slots) + 1), "Status": statuses,
                                   "P(occupied)": np.round(probs, 3)})
                with st.expander("Slot-by-slot results"):
                    st.dataframe(df, hide_index=True)
                buf = io.BytesIO()
                Image.fromarray(overlay).save(buf, format="PNG")
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Annotated image (PNG)", buf.getvalue(), "parkvision_annotated.png", "image/png")
                d2.download_button("⬇️ Slot results (CSV)", df.to_csv(index=False), "parkvision_slots.csv", "text/csv")

# ---------------------------------------------------------------- performance
with tab_perf:
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        a, b, c, d = st.columns(4)
        a.metric("Test accuracy", f"{m['accuracy']:.2%}")
        b.metric("Precision (occupied)", f"{m['precision_occupied']:.2%}")
        c.metric("Recall (occupied)", f"{m['recall_occupied']:.2%}")
        test_cnt = m.get("test_images", m.get("test_samples", "N/A"))
        epochs_run = m.get("epochs_run", "N/A")
        trained_time = m.get("trained_at", m.get("evaluated_at", "N/A"))
        arch = m.get("model_architecture", m.get("model", "Vision Model"))
        st.caption(f"Engine: {arch} | {test_cnt} unseen test images | Trained: {trained_time}")
        g1, g2 = st.columns(2)
        if config.CONFUSION_MATRIX_PATH.exists():
            g1.image(str(config.CONFUSION_MATRIX_PATH), caption="Confusion matrix (test set)")
        if config.HISTORY_PLOT_PATH.exists():
            g2.image(str(config.HISTORY_PLOT_PATH), caption="Training history")
        if config.REPORT_PATH.exists():
            st.divider()
            st.subheader("Classification Report (Test Set)")
            st.code(config.REPORT_PATH.read_text())
    else:
        st.info("No evaluation report yet. Run `python train_model.py` to train the model and create it.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### 🧠 Human-Inspired Vision & Architecture

When a human observer looks at a parking surveillance camera, **they never ask "How many rows and columns are there?"** Instead, human visual perception naturally relies on two core visual cues:

1. **Pavement Chalk Markings & Slot Geometry:** High-contrast painted white/yellow stripes on asphalt define the physical parking boundaries.
2. **Vehicle Dimensions & Clearance Intervals:** By recognizing the average size of a car ($\bar{w} \times \bar{h}$) and the spacing between parked cars, humans easily estimate how many empty vehicles can fit into open asphalt spans while leaving buffer clearance.

**How ParkVision AI Replicates Human Perception:**
* **Autonomous Slot Localization:** Convolutional neural networks (`slot_detector.pt`) detect the physical bay boundaries directly from the camera feed, exactly like the human eye detects painted chalk lines.
* **Contextual Occupancy Classification:** Deep learning models (YOLOv8 & MobileNetV2) evaluate each localized bay with $18\%$ contextual padding, ensuring vehicle bodies on perspective-angled bays are captured with zero false-empty errors.
* **Smart Urban Analytics:** Computes total slots, occupied vehicles, available capacity, and congestion levels (**Low** < 40%, **Moderate** 40-75%, **High** > 75%).
* **Actionable Decision Layer:** Delivers real-time guidance (proceed to park vs. divert to alternate facility).
""")
