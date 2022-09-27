from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import (
    Add,
    ReLU,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
)

class Encoder(keras.Model):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = keras.Sequential([
            backbone,
            Conv2D(512, 1, strides=1, padding="same"),
            ReLU(),
            Conv2D(512, 1, strides=1, padding="same"),
            ReLU()
        ], name="encoder_head")

    def call(self, x):
        return self.encoder(x)