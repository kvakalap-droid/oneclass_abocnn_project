# eval_oneclass.py
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from tensorflow.keras.models import load_model
from config_oneclass import OneClassConfig
from data_oneclass import get_oneclass_eval_gen
from attention import ABOCNNAttention


def l2_normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def eval_oneclass():
    enc_path = os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.ENCODER_NAME)
    center_path = os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.CENTER_NAME)

    print(f"Loading encoder: {enc_path}")
    encoder = load_model(
        enc_path,
        custom_objects={"ABOCNNAttention": ABOCNNAttention},
        compile=False,
    )
    center = np.load(center_path)

    eval_gen = get_oneclass_eval_gen()

    n = eval_gen.samples
    steps = int(np.ceil(n / OneClassConfig.BATCH_SIZE))

    y_true = eval_gen.classes
    dists = []

    for _ in range(steps):
        x_batch, _ = next(eval_gen)
        z = encoder.predict(x_batch, verbose=0)

        z = l2_normalize(z)
        c = center / (np.linalg.norm(center) + 1e-8)

        d_batch = np.sum((z - c) ** 2, axis=1)
        dists.extend(d_batch.tolist())

    dists = np.array(dists)

    # Sweep thresholds
    thresholds = np.linspace(dists.min(), dists.max(), 200)
    best_f1, best_thr = -1, thresholds[0]

    for thr in thresholds:
        y_pred = (dists > thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"\nBest Threshold = {best_thr:.6f}   (F1 = {best_f1:.4f})")

    # Final predictions
    y_pred = (dists > best_thr).astype(int)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["healthy", "diseased"]))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    print("\nUse this threshold in predict.py")

    return best_thr


if __name__ == "__main__":
    eval_oneclass()
