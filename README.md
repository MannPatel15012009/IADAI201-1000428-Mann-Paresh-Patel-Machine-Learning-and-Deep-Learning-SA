# ParkVision AI
### Intelligent Urban Parking Analytics & Space Optimisation Platform

> **CRS:** Artificial Intelligence &nbsp;|&nbsp; **Course:** Machine Learning and Deep Learning &nbsp;|&nbsp; **Scenario 1:** ParkVision AI

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](PASTE_YOUR_STREAMLIT_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

**Live app:** [Streamlit Link](https://3rcxhnehamg7qw2gfwunce.streamlit.app/) &nbsp;|
**Author:** Mann Paresh Patel &nbsp;|&nbsp; **Candidate Registration Number:** 1000428 &nbsp;|&nbsp; **School:** Aspee Nutan Academy

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [System Overview](#2-system-overview)
3. [Dataset](#3-dataset)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Development](#5-model-development)
6. [Results & Metrics](#6-results--metrics)
7. [Parking Insight Logic](#7-parking-insight-logic)
8. [Web App (Streamlit Dashboard)](#8-web-app-streamlit-dashboard)
9. [Testing & Feedback](#9-testing--feedback)
10. [Deployment](#10-deployment)
11. [How to Run Locally](#11-how-to-run-locally)
12. [Project Structure](#12-project-structure)
13. [References](#13-references)

---

## 1. Problem Statement

Urban drivers routinely lose several minutes -- and burn unnecessary
fuel -- searching for an open parking space, worsening traffic
congestion in cities. **UrbanFlow AI Pvt. Ltd.** commissioned
**ParkSmart AI**: a computer-vision system that looks at a parking
lot image, tells the user exactly which slots are free, and gives a
plain-language recommendation.

- **Input:** a photo of a parking lot (single camera view).
- **Output:** slot-by-slot Occupied/Empty status, an annotated image,
  live availability metrics, a congestion classification, and an
  actionable recommendation.
- **Real-world relevance:** this mirrors how smart-city parking
  sensors and camera-based occupancy systems (e.g. municipal smart
  parking apps) work today -- camera in, structured decision-support
  data out.

## 2. System Overview

```
                 ┌───────────────────────┐
   Parking Lot   │                       │   Slot-wise            Occupancy %,
   Image  ─────► │  ParkVision AI Model  │ ─► Empty/Occupied ──► Congestion Level,
                 │  (MobileNetV2 CNN)    │      + boxes           Recommendation
                 └───────────────────────┘
                            │
                            ▼
                 Streamlit Dashboard (upload → annotated image →
                 metrics → gauge → recommendation → CSV export)
```

**Approach chosen:** crop-based **classification** (rather than
full-image object detection). Each parking slot is treated as an
independent 224x224 image patch and classified Empty/Occupied by a
MobileNetV2 transfer-learning model. This is well suited to
fixed/semi-fixed CCTV-style parking cameras, is far more
data-efficient than training a detector from scratch, and lets every
slot be evaluated with its own confidence score.

| Aspect | Decision |
|---|---|
| Input → Output | Parking lot image → slot-wise status (Occupied/Empty) |
| Parking structure | Fixed slots, coordinates supplied via `slots.json` (or auto-grid for demos) |
| Modelling approach | Crop-based classification (MobileNetV2) |
| Real-world challenges handled | Shadows/lighting & weather variation (via PKLot's multi-weather images + augmentation); partial occlusion (via fine-tuning); generalisation (via train/val/test split + unseen-image testing) |
| Final outputs | Total / Occupied / Available slots, occupancy %, congestion level, recommendation |

## 3. Dataset

**[PKLot Dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)**
(segmented release) -- thousands of parking-lot images captured
across **sunny, cloudy, and rainy** weather conditions, already
pre-cropped per parking space and labelled `Empty` / `Occupied`.

- Download link: see `Sources of information` in the assignment
  brief, or the PKLot project page above.
- After downloading, unzip so the raw data sits at:
  `data/raw/PKLotSegmented/<LOT>/<WEATHER>/<DATE>/{Empty,Occupied}/*.jpg`

## 4. Data Preprocessing

Handled end-to-end by **`data_preprocessing.py`**:

1. **Collection:** recursively scans the raw PKLot tree and buckets
   every image into `Empty` / `Occupied`.
2. **Balancing:** randomly samples an equal number of images per
   class -- **at least 100 images/class** as required (the script
   uses the full balanced pool if more is available, which improves
   accuracy).
3. **Resizing:** every image is resized to a fixed **224x224**
   dimension for uniform model input.
4. **Splitting:** a stratified **70% train / 15% validation / 15%
   test** split, seeded for reproducibility.
5. **Folder organisation:** processed images are written to
   `data/processed/{train,val,test}/{empty,occupied}/` -- directly
   compatible with `tf.keras.utils.image_dataset_from_directory`.
6. **Augmentation:** rotation, horizontal flip, brightness and
   contrast jitter, and zoom are applied **on-the-fly, inside the
   model graph, to the training stream only** (see `train_model.py`).
   This satisfies the augmentation requirement while avoiding any
   leakage into validation/test data.

Run it:
```bash
python data_preprocessing.py
```

> **TODO (fill in before submission):** run the script and paste the
> printed class counts here, e.g.
> `train/empty: 140 | train/occupied: 140 | val/empty: 30 | ...`

## 5. Model Development

Handled by **`train_model.py`**.

- **Backbone:** MobileNetV2 pretrained on ImageNet -- lightweight and
  fast enough for free-tier Streamlit Cloud deployment.
- **Head:** GlobalAveragePooling → Dropout(0.3) → Dense(128, ReLU) →
  Dropout(0.2) → Dense(1, sigmoid).
- **Two-phase training:**
  - *Phase 1:* backbone frozen, train the new head for up to
    **25 epochs** (early stopping on validation loss).
  - *Phase 2:* unfreeze the top backbone layers and fine-tune at a
    low learning rate (`1e-5`) for up to **10 epochs**.
- **Batch size:** 32. **Optimizer:** Adam. **Loss:** binary
  cross-entropy.
- **Callbacks:** `ModelCheckpoint` (best val accuracy),
  `EarlyStopping`, `ReduceLROnPlateau`.
- **Evaluation:** accuracy, precision, recall, and a confusion matrix
  on the held-out test split.
- **Iterative improvement levers** (used if initial accuracy is
  low): add more training images for the weaker class, inspect and
  relabel any noisy PKLot crops, adjust dropout/learning rate, or
  extend fine-tuning epochs.

Run it:
```bash
python train_model.py
```

Artifacts produced:
- `models/parkvision_mobilenetv2.keras` -- the trained model
- `models/training_history.png` -- accuracy/loss curves
- `models/confusion_matrix.png` -- confusion matrix on test data
- `models/classification_report.txt` -- precision/recall/F1 report

## 6. Results & Metrics

> **TODO (fill in after training on your machine):**

| Metric | Value |
|---|---|
| Test Accuracy | `__.__%` |
| Precision (Occupied) | `__.__%` |
| Recall (Occupied) | `__.__%` |
| F1-score | `__.__` |

**Training curves:**
`models/training_history.png`
*(embed with `![training curves](models/training_history.png)`)*

**Confusion matrix:**
`models/confusion_matrix.png`
*(embed with `![confusion matrix](models/confusion_matrix.png)`)*

## 7. Parking Insight Logic

Handled by **`parking_utils.py`** -- this is the layer that turns raw
predictions into a decision-support tool:

1. **Counting:** total slots, occupied slots, available slots.
2. **Occupancy %:** `occupied / total * 100`.
3. **Congestion classification:**
   - `< 40%` → **Low**
   - `40% – 75%` → **Moderate**
   - `> 75%` → **High**
4. **Recommendation engine:**
   - Low/Moderate → *"Slots available -- proceed to park."* style message
   - High → *"Parking full -- try another area."* style message
5. **Visual overlays:** green boxes for empty slots, red boxes for
   occupied slots, each labelled with its slot ID -- drawn directly
   onto the uploaded image with Pillow.

## 8. Web App (Streamlit Dashboard)

Handled by **`app.py`**. Features:

- Drag-and-drop image upload.
- Two slot-definition modes: an **auto-grid** for instant demos, or
  uploading a **`slots.json`** file with real per-slot coordinates
  for accurate production use (template in
  `data/sample_lot_annotations/sample_slots.json`).
- Annotated image with colour-coded bounding boxes.
- Live metric cards (Total / Occupied / Available / Occupancy %).
- An occupancy gauge (green/yellow/red) built with Plotly.
- Congestion level + recommendation banner (success/warning/error
  styled by severity).
- A per-slot results table with confidence scores, downloadable as
  CSV.
- Clean, responsive layout designed for non-technical users (store
  managers / city operators).

Run it locally:
```bash
streamlit run app.py
```

> **TODO:** paste 2-3 screenshots of the running dashboard here
> (upload screen, annotated result, metrics/gauge panel).

## 9. Testing & Feedback

Handled by **`evaluate_model.py`**:

- Re-evaluates the model on the **held-out test split** (precision,
  recall, F1, confusion matrix).
- Optionally evaluates on a separate **`data/unseen_test/`** folder
  of real-world images you collect yourself (different lighting,
  angle, or weather than PKLot) to check robustness before
  deployment -- exactly as the brief's Step 6 asks.
- Lists every misclassified file path so errors can be inspected and
  the dataset/model iteratively improved (add more data, fix labels,
  retrain).

Run it:
```bash
python evaluate_model.py
```

> **TODO:** summarise what you found when testing on unseen images
> (e.g. "accuracy dropped slightly to __% under low-light conditions;
> added N more low-light training images and retrained, recovering
> to __%").

## 10. Deployment

1. **Push to GitHub** -- commit all code (`.py` files), the trained
   model (`models/parkvision_mobilenetv2.keras`), `requirements.txt`,
   this `README.md`, and the processed dataset (or a link to it if
   too large for GitHub).
2. **Deploy on Streamlit Community Cloud:**
   - Go to <https://streamlit.io/cloud> and sign in with GitHub.
   - Select this repository and set the main file to `app.py`.
   - Streamlit Cloud installs `requirements.txt` automatically and
     gives you a public URL.
3. **Test the live link** end-to-end (upload an image, confirm
   results render) and paste the working URL at the top of this
   README and in your submission PDF.
4. Grant repository access to `ai.assignments@wacpinternational.org`
   as required by the checklist.

**Repository naming convention:** `IADAI201<StudentID>-studentname`

## 11. How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<you>/<repo>.git
cd <repo>

# 2. Create an environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download PKLot and unzip into data/raw/

# 4. Run the pipeline
python data_preprocessing.py    # Step 2
python train_model.py           # Step 3
python evaluate_model.py        # Step 6

# 5. Launch the dashboard
streamlit run app.py            # Step 5
```

## 12. Project Structure

```
parkvision-ai/
├── app.py                     # Streamlit dashboard (Step 5)
├── config.py                  # Central configuration
├── data_preprocessing.py      # Step 2: data collection & preprocessing
├── train_model.py             # Step 3: model development & evaluation
├── parking_utils.py           # Step 4: integration & insight logic
├── evaluate_model.py          # Step 6: testing on held-out/unseen data
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/                   # (gitignored) raw PKLot download
│   ├── processed/             # train/val/test, empty/occupied
│   ├── unseen_test/           # optional real-world robustness images
│   └── sample_lot_annotations/
│       └── sample_slots.json  # example slots.json template
├── sample_images/             # demo images for the app
└── models/
    ├── parkvision_mobilenetv2.keras
    ├── training_history.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## 13. References

- PKLot: A Robust Dataset for Parking Lot Classification
- Deep Learning Based Smart Parking Occupancy Detection using Computer Vision
- Vision-Based Parking Slot Detection using Deep Learning
- Ultralytics YOLO Object Detection Documentation
- TensorFlow / Keras `MobileNetV2` transfer-learning guide
- Streamlit Official Documentation
- OpenCV Official Documentation

*(Replace with the exact papers/tutorials you actually consulted, as
required by the "GitHub Repository & Documentation" rubric item.)*

---

## Rubric Coverage Map

| Rubric Criterion (Marks) | Where it's addressed |
|---|---|
| Understanding the Problem & Requirements (5) | Section 1, 2 |
| Data Collection & Preprocessing (10) | Section 3, 4; `data_preprocessing.py` |
| Model Development (10) | Section 5, 6; `train_model.py` |
| System Logic & Insight Generation (10) | Section 7; `parking_utils.py` |
| Web App Interface with Streamlit (10) | Section 8; `app.py` |
| Testing, Deployment & App Accessibility (10) | Section 9, 10; `evaluate_model.py` |
| GitHub Repository & Documentation (5) | This README + repo structure |
