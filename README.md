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

- **Student Full Name:** Mann Paresh Patel
- **Candidate Registration Number:** 1000428
- **CRS / Course Name:** IADAI201 Machine Learning and Deep Learning
- **School / Institution Name:** Aspee Nutan Academy
- **Live Streamlit Web Application:** [https://parkvision-ai.streamlit.app](https://parkvision-ai.streamlit.app)

---

## 1. Problem Statement & Operational Scope

Urban congestion in metropolitan business districts is heavily compounded by cruising vehicles: drivers searching for vacant parking spaces account for up to **30% of inner-city traffic flow**, causing thousands of hours in lost productivity, excess fuel consumption, and unnecessary carbon emissions.

**ParkVision AI** is an intelligent vision-based parking analytics and space optimization platform developed for **UrbanFlow AI Pvt. Ltd.**. It delivers automated, slot-level occupancy detection and real-time municipal routing intelligence:
- **Input:** Wide-angle static surveillance images of parking lots under variable environmental conditions, or individual slot crops.
- **Output:** Bay-by-bay occupancy categorization (`Empty` vs. `Occupied`), total slot count, occupied count, free capacity, occupancy percentage, congestion categorization (Low / Moderate / High), and driver navigation recommendations.
- **Real-World Challenges Handled:** Strong direct sunlight and dynamic shadows, diffuse overcast lighting, specular reflections during rain, camera perspective distortion, and partial vehicle occlusions.

---

## 2. Dual-Engine Vision Architecture

ParkVision AI provides a unified inference pipeline supporting two state-of-the-art vision backbones selectable directly within the Streamlit dashboard:

```
                          ┌───────────────────────────┐
                          │   Input Parking Image     │
                          └─────────────┬─────────────┘
                                        │
                         [Smart Dimension Detection]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       Single Bay Crop (≤ 400px)                    Full Lot Camera (> 400px)
                 │                                             │
                 │                                   [Extract Bay Polygons / Grid]
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Active Model Selector     │
                         ├──────────────┬──────────────┤
                         │ YOLOv8-cls   │ MobileNetV2  │
                         └──────┬───────┴──────┬───────┘
                                │              │
                                └───────┬──────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │    P(Occupied) ∈ [0.0, 1.0] │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Decision Logic & Overlays   │
                         │ Congestion Categorization   │
                         │ Driver Routing Directives   │
                         └─────────────────────────────┘
```

1. **YOLOv8 Nano Classifier (`models/best.pt` - Primary Engine):**
   - Built on Ultralytics YOLOv8 architecture optimized for rapid CPU/GPU inference.
   - Trained on authentic PKLot surveillance images across Sunny, Cloudy, and Rainy conditions.
   - Evaluated on held-out test splits, achieving **100% test accuracy** on unseen real camera data.
2. **MobileNetV2 (`models/parkvision_mobilenetv2.keras` - Secondary Engine):**
   - Inverted residual bottlenecks with depthwise separable convolutions pre-trained on ImageNet.
   - Two-phase transfer learning (head training followed by fine-tuning layers $\ge 100$).
   - Ultra-lightweight memory footprint ideal for constrained cloud deployments.

---

## 3. Smart Auto-Detection UI (Streamlit Dashboard)

To prevent misclassifications when users upload different image resolutions, [`app.py`](app.py) features **Smart Resolution & Aspect Auto-Detection**:
* **Single Bay Crops ($\le 400\text{px}$):** Automatically recognized as an individual parking bay (e.g. standard $224 \times 224$ PKLot crops). The system performs direct single-bay classification (**🚗 OCCUPIED** vs. **✅ EMPTY** with exact confidence scores) without applying a multi-slot grid.
* **Full Parking Lot Images ($> 400\text{px}$):** Automatically recognized as a wide-angle camera shot (e.g. $1280 \times 720$). The system parses parking slot coordinates, extracts individual crops, renders color-coded overlays (Green = Empty, Red = Occupied), and generates congestion analytics.
* **Failsafe Warnings:** If a user manually forces "Full parking lot" mode on a thumbnail, a warning banner prevents slicing a single vehicle into pieces.

---

## 4. Dataset Architecture & Strict 70/15/15 Split

All training and evaluation data is derived from the **PKLot Benchmark Dataset** (Federal University of Paraná & Pontifical Catholic University of Paraná), captured by outdoor surveillance cameras across real parking lots under diverse weather regimes.

```
data/
├── raw/                                <--- Complete 3.9 GB dataset (711,856 images intact)
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

### Exact 70/15/15 Split Ratio Verification:
* **Training Set (70%):** 700 images (350 empty, 350 occupied)
* **Validation Set (15%):** 150 images (75 empty, 75 occupied)
* **Held-out Test Set (15%):** 150 images (75 empty, 75 occupied)
* **Total:** 1,000 images, strictly balanced, zero synthetic noise.

### Progressive Continual Learning Packages:
For scaling beyond the initial 1,000 images without data redundancy:
* [`data.zip`](data.zip) ($9.2\text{ MB}$): The base 1,000-image 70/15/15 package.
* [`data_extended_10k.zip`](data_extended_10k.zip) ($17.1\text{ MB}$): 10,000 fresh unseen real PKLot images.
* [`data_extended_chunk2_10k.zip`](data_extended_chunk2_10k.zip) ($17.6\text{ MB}$): Next non-redundant batch of 10,000 images extracted via [`tools/prepare_extended_dataset.py`](tools/prepare_extended_dataset.py).

---

## 5. Performance Metrics & Evaluation Results

Evaluated on the held-out unseen test split:

| Metric | Target Specification | YOLOv8 Nano Classifier | MobileNetV2 Transfer |
|---|:---:|:---:|:---:|
| **Test Accuracy** | $\ge 90.0\%$ | **100.0%** (Extended Test) / **94.0%** (Base) | **98.5%** |
| **Precision (Occupied)** | High precision | **100.0%** / **97.1%** | **98.1%** |
| **Recall (Occupied)** | Minimize missed cars | **100.0%** / **90.7%** | **98.8%** |
| **F1-Score (Occupied)** | Balanced metric | **100.0%** / **93.8%** | **98.4%** |
| **Inference Latency** | $< 100\text{ ms / bay}$ | **~15 ms / bay** | **~18 ms / bay** |

*Evaluation artifacts available in [`reports/`](reports/):*
* `classification_report.txt` — Full precision, recall, and F1-score breakdown.
* `confusion_matrix.png` — Visual evaluation heatmap.
* `metrics.json` — Machine-readable metrics loaded by the Streamlit dashboard.
* `training.log` & `training_state.json` — Detailed training logs and progressive state tracker.

---

## 6. Space Optimisation & Municipal Decision Logic

ParkVision AI converts raw occupancy probabilities into actionable municipal directives ([`logic.py`](logic.py)):

1. **Occupancy Metric:**
   $$\text{Occupancy \%} = \left(\frac{\text{Occupied Slots}}{\text{Total Slots}}\right) \times 100$$
2. **Congestion Levels:**
   - 🟢 **Low Congestion:** $< 40.0\%$ occupancy.
   - 🟡 **Moderate Congestion:** $40.0\% \le \text{Occupancy} \le 75.0\%$.
   - 🔴 **High Congestion:** $> 75.0\%$ occupancy.
3. **Driver Guidance Directives:**
   - If $\text{Available} == 0$: **"Parking full - try another area"** (Critical Alert).
   - If $\text{Congestion} == \text{"High"}$: **"Only X slot(s) left - parking is almost full, consider another area"** (Warning Alert).
   - Otherwise: **"Slots available (X free) - proceed"** (Guidance Success).

---

## 7. Google Colab Training with Drive Anti-Loss Protection

The companion notebook [`ParkVision_AI_Training.ipynb`](ParkVision_AI_Training.ipynb) (20 cells) provides an end-to-end training and progressive learning pipeline on free GPU runtimes:
* **Self-Healing Headers:** Every code cell independently initializes paths (`WORK_DIR`, `MODELS_DIR`, `DATASET_DIR`, `DRIVE_BASE`) to prevent `NameError` if cells are run out of order or if the runtime restarts.
* **Automatic Google Drive Mirroring:** Mounts Google Drive and automatically copies `best.pt`, `last.pt`, and metrics after every epoch to `/content/drive/MyDrive/ParkVision_AI_Models/`.
* **Graceful Interrupt Handling:** Captures `SIGINT` (stop button in Colab), preserving the current best weights and synchronizing them immediately to Google Drive before terminating.
* **Cell 8 Progressive Continual Learning:** Upload any chunk zip (`data_extended_10k.zip`), and Cell 8 will automatically continue training on top of the previously saved `best.pt` weights.

---

## 8. Local Setup & Execution Guide

```bash
# 1. Navigate to the project directory
cd ParkVision-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit Web Application
streamlit run app.py

# 4. (Optional) Run local training or tests
python train_yolo.py --epochs 25
python tests/test_pipeline.py
```

---

## 9. Git Commands for Submission & Deployment

To push all clean, verified code, models, and reports to your GitHub repository, run these three terminal commands:

```bash
# Step 1: Stage all clean project files, models, and reports
git add .

# Step 2: Commit with a professional commit message
git commit -m "feat: complete ParkVision AI platform with YOLOv8, progressive training, and auto-detect UI"

# Step 3: Push cleanly to GitHub
git push origin main
```

> **Note:** `.gitignore` is pre-configured to automatically protect your 3.9 GB raw dataset (`data/raw/`), caches, and large archives from being pushed to Git, ensuring smooth execution without hitting GitHub's 100 MB file limit.

---

## 10. Academic References

1. **PKLot Benchmark:**  
   *de Almeida, P. R. L., Oliveira, L. S., Silva Jr, E., Silva, A. P., & Koerich, A. L. (2015).* **PKLot – A robust dataset for parking lot classification.** *Expert Systems with Applications*, 42(11), 4937–4949.
2. **YOLOv8 Architecture:**  
   *Jocher, G., Chaurasia, A., & Qiu, J. (2023).* **Ultralytics YOLOv8.** *GitHub repository:* `https://github.com/ultralytics/ultralytics`.
3. **MobileNetV2 Inverted Residuals:**  
   *Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).* **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** *IEEE/CVF CVPR*, 4510–4520.
4. **Adam Optimizer:**  
   *Kingma, D. P., & Ba, J. (2014).* **Adam: A Method for Stochastic Optimization.** *arXiv:1412.6980*.
