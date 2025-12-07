import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from config_multiclass import MultiClassConfig
from data_multiclass import get_multiclass_generators
from attention import ABOCNNAttention
from layers_normalize import L2Normalize


def eval_multiclass():
    _, val_gen = get_multiclass_generators()

    model_path = os.path.join(MultiClassConfig.SAVE_DIR, MultiClassConfig.MODEL_NAME)
    model = load_model(
        model_path,
        custom_objects={
            "ABOCNNAttention": ABOCNNAttention,
            "L2Normalize": L2Normalize,
        },
    )

    y_prob = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_gen.classes

    class_indices = val_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\nMulticlass Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    metrics_path = os.path.join(
        MultiClassConfig.METRICS_DIR, "multiclass_evaluation.txt"
    )
    with open(metrics_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"\nSaved multiclass evaluation to: {metrics_path}")


if __name__ == "__main__":
    eval_multiclass()
