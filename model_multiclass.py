# model_multiclass.py  (SAFE + STABLE VERSION)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

from config_multiclass import MultiClassConfig
from config_oneclass import OneClassConfig

from attention import ABOCNNAttention
from layers_normalize import L2Normalize


def build_multiclass_model_from_oneclass(shared_trainable=True):
    """
    Stage-2 ABOCNN for 4-class classification.
    This version avoids TensorFlow graph freezing by:
       1. Building attention in a dummy model
       2. Setting weights safely
       3. Injecting attention into the real model
    """

    # ----------------------------------------------------------------------
    # Load Stage-1 Encoder and Extract Shared Attention Weights
    # ----------------------------------------------------------------------
    print("[INFO] Loading Stage-1 encoder...")
    oneclass_encoder = load_model(
        f"{OneClassConfig.SAVE_DIR}/{OneClassConfig.ENCODER_NAME}",
        custom_objects={
            "ABOCNNAttention": ABOCNNAttention,
            "L2Normalize": L2Normalize,
        },
        compile=False,
    )

    att_stage1 = oneclass_encoder.get_layer("shared_attention")
    pretrained_att_weights = att_stage1.get_weights()
    print(f"[INFO] Loaded Stage-1 attention weights ({len(pretrained_att_weights)} tensors).")

    # ----------------------------------------------------------------------
    # Dummy Model to Build Attention Layer Safely
    # ----------------------------------------------------------------------
    print("[INFO] Building dummy attention model (safe initialization)...")

    dummy_input = layers.Input(shape=(10, 10, 512))     # fake spatial size, fake channels
    dummy_att = ABOCNNAttention(name="shared_attention")
    _ = dummy_att(dummy_input)  # builds internal Conv2D weights

    dummy_att.set_weights(pretrained_att_weights)
    dummy_attention_layer = dummy_att  # keep reference

    # ----------------------------------------------------------------------
    # Build Actual Stage-2 Model
    # ----------------------------------------------------------------------
    print("[INFO] Building Stage-2 VGG16 backbone...")

    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(
            MultiClassConfig.IMG_HEIGHT,
            MultiClassConfig.IMG_WIDTH,
            MultiClassConfig.CHANNELS,
        ),
    )

    for layer in base.layers:
        layer.trainable = False

    inputs = layers.Input(
        shape=(
            MultiClassConfig.IMG_HEIGHT,
            MultiClassConfig.IMG_WIDTH,
            MultiClassConfig.CHANNELS,
        )
    )

    x = base(inputs)

    print("[INFO] Inserting pretrained shared attention...")
    att = dummy_attention_layer
    att.trainable = shared_trainable
    x = att(x)

    # ----------------------------------------------------------------------
    # Classification Head
    # ----------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        MultiClassConfig.NUM_CLASSES, activation="softmax"
    )(x)

    model = models.Model(inputs, outputs, name="ABOCNN_Multiclass_SharedAttention")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(MultiClassConfig.LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[INFO] Stage-2 model ready for training.")

    return model
