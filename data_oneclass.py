from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config_oneclass import OneClassConfig
from preprocessing import abocnn_preprocessing


def _preprocess(img):
    return abocnn_preprocessing(img.astype("uint8"))


def get_oneclass_train_gen():
    """
    Train only on healthy:
      data/rice_oneclass/healthy/
    """
    datagen = ImageDataGenerator(preprocessing_function=_preprocess)

    train_gen = datagen.flow_from_directory(
        OneClassConfig.DATA_DIR,
        target_size=(OneClassConfig.IMG_HEIGHT, OneClassConfig.IMG_WIDTH),
        batch_size=OneClassConfig.BATCH_SIZE,
        classes=["healthy"],
        class_mode=None,
        shuffle=True,
        seed=OneClassConfig.SEED,
    )
    return train_gen


def get_oneclass_eval_gen():
    """
    Evaluate on healthy + diseased:
      healthy  → label 0
      diseased → label 1
    """
    datagen = ImageDataGenerator(preprocessing_function=_preprocess)

    eval_gen = datagen.flow_from_directory(
        OneClassConfig.DATA_DIR,
        target_size=(OneClassConfig.IMG_HEIGHT, OneClassConfig.IMG_WIDTH),
        batch_size=OneClassConfig.BATCH_SIZE,
        classes=["healthy", "diseased"],
        class_mode="binary",
        shuffle=False,
        seed=OneClassConfig.SEED,
    )
    return eval_gen
