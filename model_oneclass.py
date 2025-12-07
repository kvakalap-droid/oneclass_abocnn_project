# model_oneclass.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

from config_oneclass import OneClassConfig
from attention import ABOCNNAttention


def build_oneclass_encoder():
    """
    Stage-1 Encoder for One-Class Learning.
    FIXES APPLIED:
      ✓ Unfreeze last VGG block (block5)
      ✓ Remove L2 normalization from training
      ✓ Fully compatible with Deep SVDD
    """

    # Base CNN
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(
            OneClassConfig.IMG_HEIGHT,
            OneClassConfig.IMG_WIDTH,
            OneClassConfig.CHANNELS,
        ),
    )

    # Freeze everything except last 4 layers
    for layer in base.layers[:-4]:
        layer.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True

    inputs = layers.Input(
        shape=(
            OneClassConfig.IMG_HEIGHT,
            OneClassConfig.IMG_WIDTH,
            OneClassConfig.CHANNELS,
        )
    )

    x = base(inputs)
    x = ABOCNNAttention(name="shared_attention")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Embedding WITHOUT normalization
    x = layers.Dense(
        OneClassConfig.EMBEDDING_DIM,
        activation="relu",
        name="embedding_dense",
    )(x)

    encoder = models.Model(
        inputs=inputs,
        outputs=x,
        name="ABOCNN_OneClass_Encoder",
    )
    return encoder
