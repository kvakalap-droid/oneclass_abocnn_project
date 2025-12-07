import tensorflow as tf
from tensorflow.keras import layers


class ABOCNNAttention(layers.Layer):
    """
    Spatial attention: avg+max over channels → conv → sigmoid mask → residual refine.
    """

    def __init__(self, **kwargs):
        super(ABOCNNAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_mask = layers.Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        mask = self.conv_mask(concat)
        weighted = layers.Multiply()([inputs, mask])
        out = layers.Add()([inputs, weighted])
        return out

    def get_config(self):
        cfg = super().get_config()
        return cfg
