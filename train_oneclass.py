# train_oneclass.py
import os
import numpy as np
import tensorflow as tf

from config_oneclass import OneClassConfig
from data_oneclass import get_oneclass_train_gen
from model_oneclass import build_oneclass_encoder


def train_oneclass():
    print("\n=== Building Stage-1 Encoder ===")
    encoder = build_oneclass_encoder()
    print(encoder.summary())

    train_gen = get_oneclass_train_gen()
    steps = int(np.ceil(train_gen.samples / OneClassConfig.BATCH_SIZE))

    optimizer = tf.keras.optimizers.Adam(OneClassConfig.LR)

    print("\n=== Custom Deep-SVDD Warm-up Training ===")
    for epoch in range(OneClassConfig.EPOCHS):
        epoch_loss = []

        for _ in range(steps):
            batch = next(train_gen)

            with tf.GradientTape() as tape:
                z = encoder(batch, training=True)
                # TEMP CENTER = mean of batch
                batch_center = tf.reduce_mean(z, axis=0)
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - batch_center), axis=1))

            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_loss.append(loss.numpy())

        print(f"Epoch {epoch+1}/{OneClassConfig.EPOCHS} - Loss: {np.mean(epoch_loss):.6f}")

    # Save encoder
    enc_path = os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.ENCODER_NAME)
    encoder.save(enc_path)
    print(f"\nSaved trained encoder: {enc_path}")

    # Recompute REAL center (on full healthy dataset AFTER training)
    print("\n=== Computing FINAL Deep-SVDD center ===")

    train_gen = get_oneclass_train_gen()  # reset
    embeddings = []

    for _ in range(steps):
        batch = next(train_gen)
        z = encoder.predict(batch, verbose=0)
        embeddings.append(z)

    embeddings = np.vstack(embeddings)
    center = np.mean(embeddings, axis=0)

    center_path = os.path.join(OneClassConfig.SAVE_DIR, OneClassConfig.CENTER_NAME)
    np.save(center_path, center)

    print(f"Saved center: {center_path}")
    print("\nStage-1 training complete.\n")


if __name__ == "__main__":
    train_oneclass()
