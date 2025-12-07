import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from config_oneclass import OneClassConfig
from config_multiclass import MultiClassConfig
from preprocessing import abocnn_preprocessing
from attention import ABOCNNAttention
from layers_normalize import L2Normalize


# Load Stage-1 encoder + center
oneclass_encoder = load_model(
    os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.ENCODER_NAME),
    custom_objects={
        "ABOCNNAttention": ABOCNNAttention,
        "L2Normalize": L2Normalize,
    },
    compile=False,
)
center = np.load(os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.CENTER_NAME))

# Load Stage-2 multiclass model
multiclass_model = load_model(
    os.path.join(MultiClassConfig.SAVE_DIR, MultiClassConfig.MODEL_NAME),
    custom_objects={
        "ABOCNNAttention": ABOCNNAttention,
        "L2Normalize": L2Normalize,
    },
    compile=False,
)

CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]


def _prep(image_path, h, w):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (w, h))
    x = abocnn_preprocessing(img_rgb)
    return np.expand_dims(x, axis=0)


def stage1_decision(image_path, threshold):
    x = _prep(image_path, OneClassConfig.IMG_HEIGHT, OneClassConfig.IMG_WIDTH)
    z = oneclass_encoder.predict(x, verbose=0)[0]
    dist = np.sum((z - center) ** 2)
    is_diseased = dist > threshold
    return dist, is_diseased


def stage2_decision(image_path):
    x = _prep(image_path, MultiClassConfig.IMG_HEIGHT, MultiClassConfig.IMG_WIDTH)
    prob = multiclass_model.predict(x, verbose=0)[0]
    idx = np.argmax(prob)
    return CLASS_NAMES[idx], float(prob[idx])


def predict(image_path, threshold):
    print("=== Stage 1: One-Class Anomaly Detection ===")
    dist, diseased = stage1_decision(image_path, threshold)
    print(f"Distance to center: {dist:.4f}")
    print(f"Threshold:         {threshold:.4f}")

    if not diseased:
        print("\nFinal Decision: HEALTHY 🌿")
        return "Healthy", 1.0, dist

    print("\nAbnormal detected → Stage 2 disease classification...")
    disease, conf = stage2_decision(image_path)
    print("\n=== Stage 2: Disease Classification ===")
    print(f"Disease:    {disease}")
    print(f"Confidence: {conf:.4f}")
    return disease, conf, dist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Stage ABOCNN Predictor")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Best threshold from eval_oneclass.py",
    )
    args = parser.parse_args()

    predict(args.image, args.threshold)
