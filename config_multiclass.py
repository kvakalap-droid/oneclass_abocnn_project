# config_multiclass.py
import os
import numpy as np
import tensorflow as tf


class MultiClassConfig:
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3          # <-- ADD THIS LINE ✔️
    
    NUM_CLASSES = 4
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4

    DATA_DIR = "data/rice4class"
    SAVE_DIR = "saved_multiclass"
    MODEL_NAME = "abocnn_multiclass.h5"
    METRICS_DIR = "metrics_multiclass"

    SEED = 42

    @staticmethod
    def init():
        os.makedirs(MultiClassConfig.SAVE_DIR, exist_ok=True)
        os.makedirs(MultiClassConfig.METRICS_DIR, exist_ok=True)
        np.random.seed(MultiClassConfig.SEED)
        tf.random.set_seed(MultiClassConfig.SEED)


MultiClassConfig.init()
