"""
ParkVision AI — Step 6: Testing & Feedback
================================================
Evaluates the trained model on:
  1. The held-out TEST split created during preprocessing.
  2. Any additional real-world "unseen" images you drop into
     data/unseen_test/{empty,occupied}/ -- e.g. different lighting,
     weather, or camera angle than the training data -- to check
     robustness before deployment, as required by the brief.

Reports precision/recall/F1, a confusion matrix, and lists
misclassified files so you can inspect and iterate.

Usage:
    python evaluate_model.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config


def evaluate_on_dir(model, directory, title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    ds = tf.keras.utils.image_dataset_from_directory(
        directory, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )
    class_names = ds.class_names
    paths = ds.file_paths
    y_true, y_pred = [], []

    for x, y in ds:
        y_true.extend(y.numpy().flatten().tolist())
        preds = model.predict(x, verbose=0).flatten()
        y_pred.extend((preds >= 0.5).astype(int).tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    misclassified = [p for p, t, pr in zip(paths, y_true, y_pred) if t != pr]
    print(f"\n{len(misclassified)} misclassified out of {len(paths)} total images.")
    for p in misclassified[:20]:
        print("  -", p)
    if len(misclassified) > 20:
        print(f"  ... and {len(misclassified) - 20} more.")

    return y_true, y_pred, misclassified


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run train_model.py first."
        )

    model = tf.keras.models.load_model(config.MODEL_PATH)

    evaluate_on_dir(model, config.TEST_DIR, "Evaluating on held-out TEST split")

    if os.path.isdir(config.UNSEEN_TEST_DIR):
        evaluate_on_dir(model, config.UNSEEN_TEST_DIR,
                         "Evaluating on additional UNSEEN images (real-world robustness check)")
    else:
        print(f"\n(No unseen-test folder found at {config.UNSEEN_TEST_DIR} -- "
              "add images under empty/ and occupied/ subfolders to run this check.)")


if __name__ == "__main__":
    main()
