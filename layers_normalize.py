import tensorflow as tf
from tensorflow.keras import layers


class L2Normalize(layers.Layer):
    """
    Serializable L2 normalization layer, safe for saving/loading models.
    """

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
