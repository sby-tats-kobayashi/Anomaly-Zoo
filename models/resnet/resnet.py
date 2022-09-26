import tensorflow as tf
import numpy as np
import os
import pathlib
from tensorflow.keras import layers
from tensorflow import keras
from typing import Union


class IdentifyLayer(layers.Layer):
    def __init__(self):
        super(IdentifyLayer, self).__init__()

    def call(self, x):
        return x


class ResBlock(keras.Model):
    def __init__(self, filters, downsample=False):
        super(ResBlock, self).__init__()
        if downsample:
            self.conv1 = layers.Conv2D(filters, 3, strides=2, padding="same")
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, 2, padding="same"),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(filters, 3, 1, padding="same")
            self.shortcut = IdentifyLayer()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, 3, 1, padding="same")

    def call(self, x):
        # skip the inputs
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.ReLU()(x)

        x = x + skip
        x = layers.ReLU()(x)

        return x


class ResNet18(keras.Model):
    def __init__(self, output_dim=128):
        super().__init__()
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 7, 2, padding="same"),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="layer0")

        self.layer1 = keras.Sequential([
            ResBlock(64, downsample=False),
            ResBlock(64, downsample=False),
        ], name="layer1")
        self.layer2 = keras.Sequential([
            ResBlock(128, downsample=True),
            ResBlock(128, downsample=False)
        ], name="layer2")
        self.layer3 = keras.Sequential([
            ResBlock(256, downsample=True),
            ResBlock(256, downsample=False)
        ], name="layer3")
        self.layer4 = keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False)
        ], name="layer4")
        # self.layer5 = keras.Sequential([
        #     keras.layers.GlobalAveragePooling2D(),
        #     keras.layers.Dense(output_dim)
        # ], name="layer5")

    def call(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        return x


class ResNet34(keras.Model):
    def __init__(self, output_dim=128):
        super().__init__()
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 7, 2, padding="same"),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="layer0")

        self.layer1 = keras.Sequential([
            ResBlock(64, downsample=False),
            ResBlock(64, downsample=False),
            ResBlock(64, downsample=False)
        ], name="layer1")
        self.layer2 = keras.Sequential([
            ResBlock(128, downsample=True),
            ResBlock(128, downsample=False),
            ResBlock(128, downsample=False),
            ResBlock(128, downsample=False),
        ], name="layer2")
        self.layer3 = keras.Sequential([
            ResBlock(256, downsample=True),
            ResBlock(256, downsample=False),
            ResBlock(256, downsample=False),
            ResBlock(256, downsample=False),
            ResBlock(256, downsample=False),
            ResBlock(256, downsample=False),
        ], name="layer3")
        self.layer4 = keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False),
            ResBlock(512, downsample=False),
        ], name="layer4")
        # self.layer5 = keras.Sequential([
        #     keras.layers.GlobalAveragePooling2D(),
        #     keras.layers.Dense(output_dim)
        # ], name="layer5")

    def call(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        return x


class ResBottleneckBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()

        self.downsample = downsample
        self.filters = filters
        if self.downsample:
            self.strides = 2
        else:
            self.strides = 1

        self.conv1 = layers.Conv2D(filters, 1, 1, padding="same")
        self.conv2 = layers.Conv2D(filters, 3, self.strides, padding="same")
        self.conv3 = layers.Conv2D(4 * filters, 1, 1, padding="same")

    def build(self, input_shape):
        if self.downsample or self.filters * 4 != input_shape[-1]:
            self.shortcut = keras.Sequential([
                layers.Conv2D(self.filters * 4, 1, self.strides, padding="same"),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = IdentifyLayer()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

    def call(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = layers.ReLU()(x)

        x = shortcut + x
        x = layers.ReLU()(x)

        return x


class ResNet(keras.Model):
    def __init__(self, repeat, outputs):
        super().__init__()
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 7, 2, padding="same"),
            layers.MaxPool2D(3, 2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="layer0")

        self.layer1 = keras.Sequential([
            ResBottleneckBlock(64, downsample=False) for _ in range(repeat[0])
        ], name="layer1")

        self.layer2 = keras.Sequential([
            ResBottleneckBlock(128, downsample=True)
        ] + [ResBottleneckBlock(128, downsample=False) for _ in range(1, repeat[1])
        ], name="layer2")

        self.layer3 = keras.Sequential([
            ResBottleneckBlock(256, downsample=True)
        ] + [ResBottleneckBlock(256, downsample=False) for _ in range(1, repeat[2])
        ], name="layer3")

        self.layer4 = keras.Sequential([
            ResBottleneckBlock(512, downsample=True)
        ] + [ResBottleneckBlock(512, downsample=False) for _ in range(1, repeat[3])
        ], name="layer4")

        # self.gap = layers.GlobalAveragePooling2D()
        # self.fc = layers.Dense(outputs)

    def call(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.gap(x)
        # x = self.fc(x)

        return x

class ResNet50(ResNet):
    def __init__(self):
        super().__init__(repeat=[3, 4, 6, 3])
    def call(self, x):
        return super().call(x)


class ResNet101(ResNet):
    def __init__(self):
        super().__init__(repeat=[3, 4, 23, 3])
    def call(self, x):
        return super().call(x)


class ResNet152(ResNet):
    def __init__(self):
        super().__init__(repeat=[3, 8, 36, 3])
    def call(self, x):
        return super().call(x)


def build_resnet(model: str, input_shape: Union[tuple, list], output_dim: int, get_map=True):

    model_type = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    assert model in model_type, f"Specified model type doesn't exist. These types of resnet are available: {model_type}"
    assert len(input_shape) == 4, f"input_shape: {input_shape} is required to be a tuple or list of size 4 \n ex. [Batch, Height, Width, Channel]"

    if model == "resnet18":
        resnet = ResNet18(output_dim)
        resnet.build(input_shape=input_shape)
        resnet.summary()

    if model == "resnet34":
        resnet = ResNet34(output_dim)
        resnet.build(input_shape=input_shape)
        resnet.summary()

    if model == "resnet50":
        resnet = ResNet50(output_dim)
        resnet.build(input_shape=input_shape)
        resnet.summary()

    if model == "resnet101":
        resnet = ResNet101(output_dim)
        resnet.build(input_shape=input_shape)
        resnet.summary()

    if model == "resnet152":
        resnet = ResNet152(output_dim)
        resnet.build(input_shape=input_shape)
        resnet.summary()

    if get_map:
        return resnet
    else:
        return resnet