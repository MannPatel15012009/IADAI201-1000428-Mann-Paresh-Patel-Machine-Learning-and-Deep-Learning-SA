"""
ParkVision AI — Slot Detector Training (YOLOv8)
====================================================
Trains a YOLOv8 object detector that finds every parking space in a
full lot photo, so end users never have to supply slot coordinates.

It trains directly on the Roboflow YOLOv8 export you already have
(the same one prepare_from_roboflow.py used). The detector learns to
locate slots; the existing MobileNetV2 classifier then decides
Empty vs Occupied for each detected slot.

Usage:
    pip install ultralytics
    python train_detector.py --export_root path/to/roboflow_export

Output:
    models/slot_detector.pt
"""

import argparse
import os
import shutil

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_root", required=True,
                        help="Unzipped Roboflow YOLOv8 export (contains data.yaml)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base", default="yolov8n.pt",
                        help="Pretrained checkpoint to start from (n = smallest/fastest)")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = os.path.join(args.export_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found in {args.export_root}")

    model = YOLO(args.base)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        project=config.MODEL_DIR,
        name="slot_detector_run",
        exist_ok=True,
    )

    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    shutil.copy(best, config.DETECTOR_PATH)
    print(f"\nSaved slot detector -> {config.DETECTOR_PATH}")
    print("Next: run `streamlit run app.py` -- slots are now found automatically.")


if __name__ == "__main__":
    main()
