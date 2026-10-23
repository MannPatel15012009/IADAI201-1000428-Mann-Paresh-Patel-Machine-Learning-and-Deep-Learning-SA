"""
ParkVision AI — Step 5: Web App (Streamlit Dashboard)
==========================================================
An interactive dashboard for UrbanFlow AI's ParkSmart AI system.
Upload a parking-lot photo, get instant slot-level occupancy
detection, live availability metrics, a congestion gauge, and a
plain-language recommendation.
"""

import json
import os
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

import config
from parking_utils import (load_model, generate_grid_slots, full_report,
                           detector_available, load_detector, detect_slots,
                           scale_slots)

st.set_page_config(page_title="ParkVision AI", page_icon="P", layout="wide")

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.title("ParkVision AI")
st.sidebar.caption("Intelligent Urban Parking Analytics & Space Optimisation")
st.sidebar.caption("A Project by UrbanFlow AI Pvt. Ltd.")

st.sidebar.markdown("### Slot Detection")
MODE_AUTO = "Automatic (AI finds the slots)"
MODE_JSON = "Advanced: upload slots.json"
MODE_GRID = "Demo: even grid"
slot_mode = st.sidebar.radio("How should parking slots be found?",
                              [MODE_AUTO, MODE_JSON, MODE_GRID])

rows, cols = 4, 6
det_conf = config.DETECTOR_CONF
slots_json_file = None
if slot_mode == MODE_AUTO:
    det_conf = st.sidebar.slider(
        "Detection sensitivity", 0.05, 0.90, float(config.DETECTOR_CONF), 0.05,
        help="Lower = finds more slots (may add false ones). Higher = stricter.")
    st.sidebar.caption("Just upload a photo -- the AI locates every parking space "
                        "and then checks whether each one is empty or occupied.")
elif slot_mode == MODE_JSON:
    slots_json_file = st.sidebar.file_uploader("slots.json", type=["json"])
    st.sidebar.caption("For fixed cameras where you already have exact slot "
                        "coordinates. Not needed for normal use.")
else:
    rows = st.sidebar.slider("Grid rows", 2, 10, 4)
    cols = st.sidebar.slider("Grid cols", 2, 12, 6)
    st.sidebar.caption("Only lays a grid over the image -- boxes will NOT line up "
                        "with real parking spaces. For demos only.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.write(f"`{os.path.basename(config.MODEL_PATH)}`")
st.sidebar.caption("MobileNetV2 transfer-learning classifier -- Empty vs Occupied, "
                    "fine-tuned on the PKLot dataset.")

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
st.title("ParkVision AI -- Smart Parking Occupancy Dashboard")
st.write(
    "Upload a parking-lot image to instantly detect slot-level occupancy, "
    "view real-time availability metrics, and get a smart recommendation."
)

input_source = st.radio("Image source", ["Upload a photo", "Use live camera"], horizontal=True)
if input_source == "Upload a photo":
    uploaded_image = st.file_uploader("Upload a parking-lot image", type=["jpg", "jpeg", "png"])
else:
    uploaded_image = st.camera_input("Point your camera at the parking lot and take a snapshot")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_detector():
    return load_detector()


if uploaded_image is not None:
    pil_image = Image.open(uploaded_image).convert("RGB")

    from parking_utils import check_image_quality
    if not check_image_quality(pil_image):
        st.warning("Image is too blurry. Please upload a clear photo.")
        st.stop()

    
    # Continue with processing...
    pil_image = Image.open(uploaded_image).convert("RGB")

    with st.spinner("Loading model..."):
        model = get_model()

    if slot_mode == MODE_AUTO:
        if not detector_available():
            st.error("The slot detector hasn't been trained yet (missing "
                     f"`{os.path.basename(config.DETECTOR_PATH)}`). Run "
                     "`python train_detector.py --export_root <roboflow_export>` "
                     "once, or pick another mode in the sidebar.")
            st.stop()
        with st.spinner("Finding parking slots..."):
            slots = detect_slots(get_detector(), pil_image, conf=det_conf)
        if not slots:
            st.warning("No parking slots were detected. Try lowering the detection "
                       "sensitivity in the sidebar, or upload a clearer, wider photo "
                       "of the lot.")
            st.stop()
    elif slot_mode == MODE_JSON:
        if slots_json_file is None:
            st.warning("Upload a slots.json in the sidebar, or switch to "
                       "'Automatic' mode so no file is needed.")
            st.stop()
        data = json.load(slots_json_file)
        slots = data["slots"]
        if data.get("image_width") and data.get("image_height"):
            slots = scale_slots(slots, (data["image_width"], data["image_height"]),
                                pil_image.size)
    else:
        slots = generate_grid_slots(pil_image.size, rows=rows, cols=cols)

    with st.spinner("Analyzing parking slots..."):
        report = full_report(model, pil_image, slots)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Annotated Output")
        st.image(report["annotated_image"], use_container_width=True,
                  caption="Green = Empty   |   Red = Occupied")

    with col2:
        st.subheader("Real-Time Metrics")
        m = report["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Total Slots", m["total_slots"])
        c2.metric("Occupied", m["occupied_slots"])
        c1.metric("Available", m["available_slots"])
        c2.metric("Occupancy %", f"{m['occupancy_pct']}%")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=m["occupancy_pct"],
            title={"text": "Occupancy Level"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 40], "color": "#2ecc71"},
                    {"range": [40, 75], "color": "#f1c40f"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
            },
        ))
        gauge.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10))
        st.plotly_chart(gauge, use_container_width=True)

        congestion = report["congestion"]
        st.markdown(f"**Congestion Level:** `{congestion.capitalize()}`")
        if congestion == "low":
            st.success(report["recommendation"])
        elif congestion == "moderate":
            st.warning(report["recommendation"])
        else:
            st.error(report["recommendation"])

    with st.expander("Per-slot prediction details"):
        df = pd.DataFrame(report["results"])[["id", "label", "confidence"]]
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(df, use_container_width=True)

    csv = pd.DataFrame(report["results"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download slot report (CSV)", csv, "parking_report.csv", "text/csv")

else:
    st.info("Upload an image or take a camera snapshot to get started. Sample images are available in sample_images/.")

st.markdown("---")
st.caption("Built for the WACP AI CRS -- Machine Learning & Deep Learning Summative Assessment "
           "(Scenario 1: ParkVision AI).")
