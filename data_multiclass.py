from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config_multiclass import MultiClassConfig
from preprocessing import abocnn_preprocessing


def _preprocess(img):
    return abocnn_preprocessing(img.astype("uint8"))


def get_multiclass_generators():
    """
    Uses:
      data/rice4class/Bacterialblight/
                       Blast/
                       Brownspot/
                       Tungro/
    """
    datagen = ImageDataGenerator(
        preprocessing_function=_preprocess,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        MultiClassConfig.DATA_DIR,
        target_size=(MultiClassConfig.IMG_HEIGHT, MultiClassConfig.IMG_WIDTH),
        batch_size=MultiClassConfig.BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=MultiClassConfig.SEED,
    )

    val_gen = datagen.flow_from_directory(
        MultiClassConfig.DATA_DIR,
        target_size=(MultiClassConfig.IMG_HEIGHT, MultiClassConfig.IMG_WIDTH),
        batch_size=MultiClassConfig.BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=MultiClassConfig.SEED,
    )

    return train_gen, val_gen
