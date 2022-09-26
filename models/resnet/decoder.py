from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    ReLU,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
)
from src import config


class DecodeBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super().__init__()
        self.deconv1 = Conv2DTranspose(filters, kernel_size, strides, padding="same")
        self.bn1 = BatchNormalization()

    def call(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = ReLU()(x)

        return x


class Decoder(keras.Model):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = DecodeBlock(512, 4, strides=2)
        self.layer2 = DecodeBlock(512, 3, strides=1)
        self.layer3 = DecodeBlock(256, 4, strides=2)
        self.layer4 = DecodeBlock(256, 3, strides=1)
        self.layer5 = DecodeBlock(128, 4, strides=2)
        self.layer6 = DecodeBlock(128, 3, strides=1)
        self.layer7 = DecodeBlock(64, 4, strides=2)
        self.layer8 = DecodeBlock(64, 3, strides=1)
        self.layer9 = Conv2DTranspose(channels, 4, strides=2, padding="same", activation="sigmoid")

    def call(self, encoded):
        layer1 = self.layer1(encoded)
        layer2 = self.layer2(layer1)
        add1 = Add()([layer1, layer2])

        layer3 = self.layer3(add1)
        layer4 = self.layer4(layer3)
        add2 = Add()([layer3, layer4])

        layer5 = self.layer5(add2)
        layer6 = self.layer6(layer5)
        add3 = Add()([layer5, layer6])

        layer7 = self.layer7(add3)
        layer8 = self.layer8(layer7)
        add4 = Add()([layer7, layer8])

        decoded = self.layer9(add4)

        return decoded
