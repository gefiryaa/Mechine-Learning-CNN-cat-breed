import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(num_classes):

    model = models.Sequential([

        layers.Input(shape=(128, 128, 3)),

        layers.Rescaling(1.0 / 255),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),

        # Convolution Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        # Classification Layer
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.3),

        layers.Dense(
            num_classes,
            activation="softmax"
        )

    ], name="Cat_Breeds_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model