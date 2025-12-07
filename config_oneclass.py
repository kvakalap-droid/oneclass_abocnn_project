import os
import numpy as np
import tensorflow as tf


class OneClassConfig:
    # Image shape
    IMG_HEIGHT = 300
    IMG_WIDTH = 300
    CHANNELS = 3

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-4
    EMBEDDING_DIM = 128

    # Paths
    DATA_DIR = "data/rice_oneclass"  # healthy/, diseased/
    SAVE_DIR = "saved_oneclass"
    ENCODER_NAME = "abocnn_oneclass_encoder.h5"
    CENTER_NAME = "abocnn_oneclass_center.npy"
    METRICS_DIR = "metrics_oneclass"

    SEED = 42

    @staticmethod
    def init():
        os.makedirs(OneClassConfig.SAVE_DIR, exist_ok=True)
        os.makedirs(OneClassConfig.METRICS_DIR, exist_ok=True)
        np.random.seed(OneClassConfig.SEED)
        tf.random.set_seed(OneClassConfig.SEED)


OneClassConfig.init()
