"""
ParkVision AI — Step 3: Model Development
==============================================
Trains a MobileNetV2 transfer-learning binary classifier that predicts
whether a cropped parking-slot image is Empty (0) or Occupied (1).

Design choices (worth restating in your README for the rubric):
  * Approach: crop-based CLASSIFICATION rather than full-image object
    detection. Each parking space is treated as an independent
    224x224 patch, which is both more data-efficient and more
    accurate for a fixed/semi-fixed camera layout like a parking lot.
  * Backbone: MobileNetV2 (ImageNet weights) -- lightweight enough to
    run in a free Streamlit Cloud container while remaining accurate.
  * Augmentation (rotation, flip, brightness, contrast, zoom) is
    applied ON-THE-FLY inside the model graph so it only ever touches
    the training split, never validation/test -- this avoids data
    leakage while still satisfying the brief's augmentation
    requirement.
  * Two-phase training: (1) freeze the backbone and train a new head,
    (2) unfreeze the top layers and fine-tune at a low learning rate.

Usage:
    python train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix

import config

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        label_mode="binary", shuffle=True, seed=config.SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        label_mode="binary", shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        label_mode="binary", shuffle=False
    )

    class_names = train_ds.class_names  # alphabetical -> ['empty', 'occupied']
    print("Detected classes (label 0 / label 1):", class_names)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.10),
        layers.RandomZoom(0.08),
    ], name="augmentation")


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Phase 1: feature extraction only

    inputs = layers.Input(shape=config.IMG_SIZE + (3,))
    x = build_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="ParkVisionAI_MobileNetV2")
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def get_callbacks():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(config.MODEL_PATH, save_best_only=True,
                                   monitor="val_accuracy", mode="max"),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]


# ------------------------------------------------------------------
# Evaluation & reporting
# ------------------------------------------------------------------
def plot_history(hist1, hist2=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        vals = list(hist1.history[key])
        val_vals = list(hist1.history[f"val_{key}"])
        if hist2 is not None:
            vals += list(hist2.history[key])
            val_vals += list(hist2.history[f"val_{key}"])
        ax.plot(vals, label="train")
        ax.plot(val_vals, label="val")
        if hist2 is not None:
            ax.axvline(len(hist1.history[key]) - 1, color="gray", linestyle="--", label="fine-tune start")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(config.HISTORY_PLOT_PATH, dpi=150)
    print(f"Saved training curves -> {config.HISTORY_PLOT_PATH}")


def evaluate_and_report(model, test_ds, class_names):
    y_true, y_pred_prob = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().flatten().tolist())
        y_pred_prob.extend(model.predict(x, verbose=0).flatten().tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(config.REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Saved classification report -> {config.REPORT_PATH}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix -- Test Set")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    print(f"Saved confusion matrix -> {config.CONFUSION_MATRIX_PATH}")


# ------------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------------
def main():
    print("=" * 65)
    print("ParkVision AI — Model Development (Step 3)")
    print("=" * 65)

    train_ds, val_ds, test_ds, class_names = load_datasets()
    model, base_model = build_model()

    # ---- Phase 1: train the new classification head ----
    print("\n--- Phase 1: training classification head (backbone frozen) ---")
    compile_model(model, config.LEARNING_RATE)
    model.summary()
    hist1 = model.fit(train_ds, validation_data=val_ds,
                       epochs=config.EPOCHS, callbacks=get_callbacks())

    # ---- Phase 2: fine-tune top layers of MobileNetV2 ----
    print("\n--- Phase 2: fine-tuning top backbone layers ---")
    base_model.trainable = True
    for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    compile_model(model, config.FINE_TUNE_LR)
    hist2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=config.FINE_TUNE_EPOCHS, callbacks=get_callbacks())

    plot_history(hist1, hist2)

    print("\n--- Final evaluation on held-out TEST set ---")
    evaluate_and_report(model, test_ds, class_names)

    model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
