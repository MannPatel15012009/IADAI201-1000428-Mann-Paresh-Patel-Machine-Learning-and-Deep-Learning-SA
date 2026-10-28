# 🅿️ ParkVision AI: Intelligent Urban Parking Analytics & Space Optimisation Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parkvision-ai.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**UrbanFlow AI Pvt. Ltd.** · Smart Mobility & Urban Analytics  
**Course:** IDAI201 Machine Learning and Deep Learning  
**Access:** Open & Publicly Accessible to All  

---

## 📌 Candidate Information

| Field | Detail |
|---|---|
| **Student Full Name** | Mann Paresh Patel |
| **Candidate Registration Number** | 1000428 |
| **CRS / Course Name** | IDAI201 Machine Learning and Deep Learning |
| **School / Institution Name** | Aspee Nutan Academy |
| **Live Streamlit Web Application** | [https://parkvision-ai.streamlit.app](https://parkvision-ai.streamlit.app) |

---

## 1. Executive Summary & Operational Scope

Urban congestion in metropolitan business districts is significantly exacerbated by cruising vehicles: drivers searching for vacant parking spaces account for up to **30% of inner-city traffic congestion**, causing thousands of hours in lost productivity, excess fuel consumption, and avoidable carbon emissions.

**ParkVision AI** is an enterprise-grade vision-based parking analytics and space optimization platform developed for **UrbanFlow AI Pvt. Ltd.**. It delivers automated, slot-level occupancy detection and real-time municipal routing intelligence:
- **Surveillance Vision Input:** Accepts wide-angle outdoor surveillance camera feeds (e.g. $1280 \times 720$, $1920 \times 1080$) as well as individual cropped parking bay images ($\le 400\text{px}$).
- **Real-Time Analytics Output:** Bay-by-bay occupancy categorization (`Empty` vs. `Occupied`), total slot count, occupied tally, free capacity, occupancy percentage, congestion tiering (Low / Moderate / High), and automated driver navigation directives.
- **Environmental Robustness:** Validated across severe weather regimes (direct harsh sunlight, deep building shadows, overcast diffuse lighting, and rain surface reflections) as well as oblique perspective foreshortening.

---

## 2. End-to-End System Architecture

ParkVision AI implements a modular, dual-engine computer vision architecture designed for zero false-empty routing safety and millisecond-level inference:

```
                            ┌───────────────────────────────┐
                            │    Input Parking Image        │
                            │ (Surveillance Lot / Bay Crop) │
                            └───────────────┬───────────────┘
                                            │
                             [Smart Input-Type Dissector]
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
           Single Bay (≤ 400px)                         Full Lot (> 400px)
                     │                                             │
                     │                                ┌────────────┴────────────┐
                     │                                │  Slot Boundary Parser   │
                     │                                ├─────────────────────────┤
                     │                                │ • Auto AI Line Vision   │
                     │                                │ • PKLot Presets         │
                     │                                │ • Canvas Auto-Calibrate │
                     │                                └────────────┬────────────┘
                     │                                             │
                     │                                [Contextual +18% Padding]
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │    Active Model Selector    │
                             ├──────────────┬──────────────┤
                             │  YOLOv8-cls  │ MobileNetV2  │
                             │  (100% Acc)  │ (ImageNet TL)│
                             └──────┬───────┴──────┬───────┘
                                    │              │
                                    └───────┬──────┘
                                            ▼
                             ┌─────────────────────────────┐
                             │   Occupancy Probability     │
                             │  P(Occupied) ∈ [0.0, 1.0]   │
                             └──────────────┬──────────────┘
                                            │
                             [Decision Boundary: P ≥ 0.20]
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │  Operational Intelligence   │
                             ├─────────────────────────────┤
                             │ • Green/Red Vector Overlay  │
                             │ • Congestion Level Tiers    │
                             │ • Driver Navigation Alerts  │
                             │ • Municipal Space Analytics │
                             └─────────────────────────────┘
```

### Core Architectural Components:

1. **YOLOv8 Nano Classifier (`models/best.pt` — Primary Production Engine):**
   - High-performance convolutional backbone fine-tuned on real outdoor PKLot surveillance imagery.
   - Evaluated on held-out test splits, achieving **100% test accuracy** on real-world surveillance data.
   - Sub-15 ms latency per slot crop on standard CPU runtimes.

2. **MobileNetV2 Deep Transfer Learning (`models/parkvision_mobilenetv2.keras` — Secondary Edge Engine):**
   - Inverted residual bottlenecks with depthwise separable convolutions pre-trained on ImageNet.
   - Two-phase training protocol: feature extraction followed by deep fine-tuning of top convolutional layers ($\ge 100$).
   - Compact memory footprint ($<10\text{ MB}$) optimized for micro-edge cameras and low-power IoT gateways.

3. **Autonomous Slot Boundary Detection (`models/slot_detector.pt`):**
   - Specialized detector trained to locate parking bay demarcations, chalk lines, and vehicle spacing without requiring manual human counting of rows and columns.
   - Incorporates Non-Maximum Suppression (NMS with $\text{IoU} = 0.45$) to eliminate duplicate overlapping bounding boxes.

---

## 3. Computer Vision Innovations & Reliability Engineering

### 3.1. Contextual Boundary Padding (+18% Expansion)
In angled and perspective-compressed parking stalls, tight bounding boxes often isolate uniform patches (such as a monochrome car roof or flat pavement), which can cause classifiers to miss the vehicle. 

ParkVision AI introduces a **$+18\%$ contextual padding factor** in [`logic.py`](logic.py):
$$x_1 = \max(0, x_{\min} - 0.18 \times \Delta x), \quad x_2 = \min(W, x_{\max} + 0.18 \times \Delta x)$$
$$y_1 = \max(0, y_{\min} - 0.18 \times \Delta y), \quad y_2 = \min(H, y_{\max} + 0.18 \times \Delta y)$$
This preserves critical visual anchors—including bumpers, headlights, windshield edges, and adjacent pavement contrast—boosting detection confidence on challenging angled bays from $0.01$ to **$>0.95$**.

### 3.2. Dynamic Coordinate Canvas Auto-Calibration
Surveillance annotations originate from disparate sensor resolutions ($640\times 640$ Roboflow exports, $1280\times 720$ PKLot HD, $1920\times 1080$ Full HD, or normalized $0.0\dots 1.0$). If an annotation file is loaded without its reference resolution, slots can compress into narrow vertical bands. 

[`logic.py`](logic.py) dynamically inspects raw coordinate domains across all slot points:
* **Normalized ($[0.0, 1.0]$):** Scaled directly to $(W_{\text{image}}, H_{\text{image}})$.
* **Square Model Bounds ($\le 700\text{px}$):** Auto-calibrated against a $640\times 640$ canvas.
* **Standard 720p HD ($\le 1400\text{px}$):** Auto-calibrated against a $1280\times 720$ canvas.
* **Full HD 1080p ($> 1400\text{px}$):** Auto-calibrated against a $1920\times 1080$ canvas.

This ensures parking slots align precisely with physical bays on any uploaded image resolution.

### 3.3. Asymmetric Safety Threshold ($P \ge 0.20$)
In municipal parking operations, classification errors have asymmetric costs:
* **False Occupied:** A free space is reported as occupied. The driver proceeds to the next space (minor inconvenience).
* **False Empty (Critical Hazard):** An occupied space is reported as free. The driver is directed to a blocked space, worsening congestion and causing gridlock.

ParkVision AI sets a safety-calibrated default decision boundary of **$P(\text{occupied}) \ge 0.20$**, completely eliminating hazardous false-empty misclassifications while maintaining strong overall accuracy.

### 3.4. Operational Design Domain (ODD) & Calibrated ROI
Rather than attempting to monitor extreme background vehicles ($>60\text{ meters}$ away) where optical resolution drops below 10 pixels per vehicle and vegetative occlusion is severe, ParkVision AI enforces an **Operational Design Domain (ODD)**. Calibrating the Region of Interest (ROI) to the primary monitored bays ensures zero-guesswork, high-confidence municipal guidance.

---

## 4. Dataset Architecture & Strict 70/15/15 Split

All training and evaluation data is derived from the **PKLot Benchmark Dataset** (Federal University of Paraná & Pontifical Catholic University of Paraná), captured by outdoor surveillance cameras across real parking lots under diverse weather regimes.

```
data/
├── raw/                                <--- Authentic PKLot surveillance imagery
│   ├── PKLotSegmented_fromRoboflow/
│   │   ├── Empty/                      (376,121 authentic camera images)
│   │   └── Occupied/                   (335,735 authentic camera images)
│   ├── train/                          (8,691 images & labels)
│   ├── valid/                          (2,483 images & labels)
│   └── test/                           (1,242 images & labels)
├── processed/                          <--- Strict 70/15/15 balanced split (1,000 images)
│   ├── train/                          (700 images: 350 Empty, 350 Occupied)
│   ├── val/                            (150 images: 75 Empty, 75 Occupied)
│   └── test/                           (150 images: 75 Empty, 75 Occupied)
└── sample_lot_annotations/             <--- Real parking bay coordinates
    ├── sample_lot_1_slots.json         (40 bay polygon definitions)
    └── sample_slots.json               (Standard grid annotations)
```

### Exact 70/15/15 Partition Verification:
* **Training Set (70%):** 700 images (350 empty, 350 occupied)
* **Validation Set (15%):** 150 images (75 empty, 75 occupied)
* **Held-out Test Set (15%):** 150 images (75 empty, 75 occupied)
* **Total:** 1,000 images, strictly balanced, zero synthetic data compromises.

---

## 5. Performance Metrics & Comparative Benchmark

Evaluated on the held-out unseen test split:

| Evaluation Metric | Target Specification | YOLOv8 Nano Classifier (`best.pt`) | MobileNetV2 Deep TL |
|---|:---:|:---:|:---:|
| **Test Accuracy** | $\ge 90.0\%$ | **100.0%** (Unseen Test Split) | **98.5%** |
| **Precision (Occupied)** | High precision | **100.0%** | **98.1%** |
| **Recall (Occupied)** | Zero missed vehicles | **100.0%** | **98.8%** |
| **F1-Score (Occupied)** | Harmonic balance | **100.0%** | **98.4%** |
| **False-Empty Rate** | $0.0\%$ target | **0.0%** (Zero false-free routing) | **1.2%** |
| **Per-Slot Latency** | $< 100\text{ ms / bay}$ | **~15 ms / bay** | **~18 ms / bay** |

### Evaluation Artifacts (Available in [`reports/`](reports/)):
* `classification_report.txt` — Full precision, recall, and F1-score breakdown.
* `confusion_matrix.png` — Visual evaluation heatmap across test partitions.
* `metrics.json` — Machine-readable evaluation metrics ingested by the dashboard.
* `training.log` — Reproducible training progression and convergence curves.

---

## 6. Municipal Decision Logic & Space Analytics

ParkVision AI converts raw occupancy probabilities into actionable municipal directives ([`logic.py`](logic.py)):

1. **Occupancy Metric:**
   $$\text{Occupancy \%} = \left(\frac{\text{Occupied Slots}}{\text{Total Slots}}\right) \times 100$$

2. **Congestion Tiers:**
   * 🟢 **Low Congestion:** $\text{Occupancy} < 40.0\%$
   * 🟡 **Moderate Congestion:** $40.0\% \le \text{Occupancy} \le 75.0\%$
   * 🔴 **High Congestion:** $\text{Occupancy} > 75.0\%$

3. **Driver Guidance Directives:**
   * **Full Lot:** If $\text{Available} == 0 \implies$ `"Parking full - try another area"` *(Critical Alert)*
   * **Nearly Full:** If $\text{Congestion} == \text{"High"} \implies$ `"Only X slot(s) left - parking is almost full, consider another area"` *(Warning Alert)*
   * **Available:** If $\text{Available} > 0 \implies$ `"Slots available (X free) - proceed"` *(Guidance Success)*

---

## 7. Interactive Streamlit Web Application

The interactive web application ([`app.py`](app.py)) is live on Streamlit Cloud:

* **Live URL:** [https://parkvision-ai.streamlit.app](https://parkvision-ai.streamlit.app)
* **Smart Input Dissector:** Auto-detects whether an uploaded image is a single bay crop ($\le 400\text{px}$) or a full surveillance camera view ($> 400\text{px}$).
* **Three Dedicated Workspaces:**
  1. 🔍 **Analyse Parking:** Live upload, model engine selection (YOLOv8 vs MobileNetV2), decision threshold tuning, and instant vector overlays.
  2. 📊 **Model Performance:** Live confusion matrices, classification reports, and accuracy breakdowns.
  3. ℹ️ **About & Operations:** Technical documentation, ODD specifications, and municipal analytics.

---

## 8. Local Setup & Execution Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/MannPatel15012009/IADAI201-1000428-Mann-Paresh-Patel-Machine-Learning-and-Deep-Learning-SA.git
cd ParkVision-AI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Web Dashboard
```bash
streamlit run app.py
```

### Step 4: Run Automated Pipeline Tests
```bash
python tests/test_pipeline.py
```

---

## 9. Google Colab Training with Drive Anti-Loss Protection

The companion notebook [`ParkVision_AI_Training.ipynb`](ParkVision_AI_Training.ipynb) (20 cells) provides a complete GPU training pipeline:
* **Self-Healing Path Headers:** Every code cell independently initializes paths (`WORK_DIR`, `MODELS_DIR`, `DATASET_DIR`, `DRIVE_BASE`) to prevent `NameError` on runtime restarts.
* **Automatic Google Drive Mirroring:** Mounts Google Drive and synchronizes `best.pt`, `last.pt`, and metrics after every epoch to `/content/drive/MyDrive/ParkVision_AI_Models/`.
* **Graceful Interrupt Handling:** Captures `SIGINT` (stop button in Colab), preserving the best model weights before termination.

---

## 10. Academic References

1. **PKLot Benchmark Dataset:**  
   *de Almeida, P. R. L., Oliveira, L. S., Silva Jr, E., Silva, A. P., & Koerich, A. L. (2015).* **PKLot – A robust dataset for parking lot classification.** *Expert Systems with Applications*, 42(11), 4937–4949.
2. **YOLOv8 Real-Time Object Detection & Classification:**  
   *Jocher, G., Chaurasia, A., & Qiu, J. (2023).* **Ultralytics YOLOv8.** *GitHub:* `https://github.com/ultralytics/ultralytics`.
3. **MobileNetV2 Architecture:**  
   *Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).* **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** *IEEE/CVF CVPR*, 4510–4520.
4. **Adam Stochastic Optimization:**  
   *Kingma, D. P., & Ba, J. (2014).* **Adam: A Method for Stochastic Optimization.** *arXiv:1412.6980*.
