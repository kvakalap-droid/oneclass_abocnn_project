# train_multiclass.py
import os
from config_multiclass import MultiClassConfig
from data_multiclass import get_multiclass_generators
from model_multiclass import build_multiclass_model_from_oneclass


def train_multiclass():
    train_gen, val_gen = get_multiclass_generators()
    model = build_multiclass_model_from_oneclass(shared_trainable=True)
    print(model.summary())

    # NEW SAFE MODE FOR TENSORFLOW 2.17+ / KERAS 3.x
    steps = train_gen.samples // MultiClassConfig.BATCH_SIZE
    val_steps = val_gen.samples // MultiClassConfig.BATCH_SIZE

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps,
        validation_steps=val_steps,
        epochs=MultiClassConfig.EPOCHS,
        verbose=1
    )

    save_path = os.path.join(MultiClassConfig.SAVE_DIR, MultiClassConfig.MODEL_NAME)
    model.save(save_path)
    print(f"Saved multiclass model to: {save_path}")


0
if __name__ == "__main__":
    train_multiclass()
