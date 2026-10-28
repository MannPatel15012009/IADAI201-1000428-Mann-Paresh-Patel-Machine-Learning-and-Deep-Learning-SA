"""Step 3 - train + evaluate the MobileNetV2 slot classifier (empty vs occupied).

    python train_model.py

Phase 1 trains a new head on a frozen MobileNetV2; phase 2 fine-tunes the upper layers at a
low learning rate. Total epochs = EPOCHS + FINE_TUNE_EPOCHS (config.py). The best model by
validation loss is saved to models/parkvision_mobilenetv2.keras and evaluated on the test set.
"""
import json
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow.keras import layers

import config


def load_split(path, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        path, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        label_mode="binary", class_names=config.CLASS_NAMES,   # fixes order: 0=empty, 1=occupied
        shuffle=shuffle, seed=config.SEED)
    return ds


def count_images(path):
    return {c: len([p for p in (path / c).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for c in config.CLASS_NAMES}


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),                                  # about +/- 14 degrees
        layers.RandomBrightness(0.2, value_range=(0, 255)),
        layers.RandomContrast(0.1),
    ], name="augmentation")                                           # active only during training
    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))             # raw pixels 0-255
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)                   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)                # P(occupied)
    return tf.keras.Model(inputs, outputs, name="parkvision_mobilenetv2"), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])


def callbacks(path):
    return [
        tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def plot_history(histories, boundary):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (key, title) in zip(ax, [("accuracy", "Accuracy"), ("loss", "Loss")]):
        a.plot(merged[key], label="train")
        a.plot(merged["val_" + key], label="validation")
        if boundary:
            a.axvline(boundary - 0.5, color="grey", ls="--", label="fine-tuning starts")
        a.set_title(title); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(config.HISTORY_PLOT_PATH, dpi=150); plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True); config.REPORT_DIR.mkdir(exist_ok=True)
    for name, d in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR), ("test", config.TEST_DIR)]:
        counts = count_images(d)
        print(f"{name:5s}: {counts}")
        if min(counts.values()) == 0:
            raise SystemExit(f"'{name}' split needs images in both classes - run prepare_dataset.py first.")

    train_ds, val_ds = load_split(config.TRAIN_DIR, True), load_split(config.VAL_DIR, False)
    test_ds = load_split(config.TEST_DIR, False)
    print("Class order:", train_ds.class_names)
    assert train_ds.class_names == config.CLASS_NAMES

    tc = count_images(config.TRAIN_DIR); n = sum(tc.values())
    class_weight = {i: n / (2 * tc[c]) for i, c in enumerate(config.CLASS_NAMES)}
    print("Class weights:", class_weight)
    AUTOTUNE = tf.data.AUTOTUNE
    train_p = train_ds.cache().prefetch(AUTOTUNE)
    val_p = val_ds.cache().prefetch(AUTOTUNE)

    model, base = build_model()

    # ---- phase 1: train the new head ----------------------------------
    compile_model(model, config.LEARNING_RATE)
    h1 = model.fit(train_p, validation_data=val_p, epochs=config.EPOCHS,
                   class_weight=class_weight, callbacks=callbacks(config.STAGE1_PATH))

    # ---- phase 2: fine-tune upper layers ------------------------------
    base.trainable = True
    for layer in base.layers[:config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    for layer in base.layers:                       # keep BatchNorm statistics stable on small data
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, config.FINE_TUNE_LR)
    h2 = model.fit(train_p, validation_data=val_p, epochs=config.FINE_TUNE_EPOCHS,
                   class_weight=class_weight, callbacks=callbacks(config.STAGE2_PATH))

    # ---- keep whichever phase had the lower validation loss -----------
    best1 = min(h1.history.get("val_loss", [float("inf")]))
    best2 = min(h2.history.get("val_loss", [float("inf")])) if "val_loss" in h2.history else float("inf")
    winner = config.STAGE2_PATH if (config.STAGE2_PATH.exists() and best2 <= best1) else config.STAGE1_PATH
    if not winner.exists():
        model.save(config.MODEL_PATH)
        print(f"Directly saved current model to {config.MODEL_PATH.name}")
    else:
        shutil.copy2(winner, config.MODEL_PATH)
        print(f"Best val_loss: phase1={best1:.4f} phase2={best2:.4f} -> saved {winner.name} as {config.MODEL_PATH.name}")

    for p in (config.STAGE1_PATH, config.STAGE2_PATH):
        try:
            if p.exists() and p != config.MODEL_PATH:
                p.unlink(missing_ok=True)
        except Exception as e:
            print(f"Note: Could not remove temporary file {p.name}: {e}")
    plot_history([h1, h2], len(h1.history["loss"]))

    # ---- evaluate on the held-out test set (deterministic single pass) -
    best = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    test_cached = test_ds.cache()
    y_true_list, probs_list = [], []
    for batch_x, batch_y in test_cached:
        batch_probs = np.asarray(best(batch_x, training=False)).ravel()
        probs_list.append(batch_probs)
        y_true_list.append(batch_y.numpy().ravel())
    y_true = np.concatenate(y_true_list).astype(int)
    probs = np.concatenate(probs_list)
    y_pred = (probs >= config.DEFAULT_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0)
    config.REPORT_PATH.write_text(report)
    print(report); print("Confusion matrix [[TN FP][FN TP]]:\n", cm)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Test-set confusion matrix")
    fig.tight_layout(); fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=150); plt.close(fig)

    metrics = {"accuracy": float(acc), "precision_occupied": float(prec), "recall_occupied": float(rec),
               "f1_occupied": float(f1), "confusion_matrix": cm.tolist(), "class_names": config.CLASS_NAMES,
               "test_images": int(len(y_true)), "train_counts": tc,
               "epochs_run": len(h1.history["loss"]) + len(h2.history["loss"]),
               "batch_size": config.BATCH_SIZE, "trained_at": datetime.now().isoformat(timespec="seconds")}
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    config.SUMMARY_MD_PATH.write_text(
        "| Metric | Value |\n|---|---|\n"
        f"| Test accuracy | {acc:.4f} |\n| Precision (occupied) | {prec:.4f} |\n"
        f"| Recall (occupied) | {rec:.4f} |\n| F1 (occupied) | {f1:.4f} |\n"
        f"| Test images | {len(y_true)} |\n| Epochs run | {metrics['epochs_run']} |\n")
    print(f"\nTest accuracy: {acc:.4f}\nSaved model, reports and metrics under models/ and reports/.")


if __name__ == "__main__":
    main()
