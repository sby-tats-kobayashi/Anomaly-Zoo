import os
import pathlib

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import PIL
from PIL import Image
import config
import matplotlib.pyplot as plt

'''
Configで指定するパラメータは基本的に変えなくてよいものにしたい。
また、コンストラクタの引数で受け取るパラメータは利用者/日時によって変わるものにしたい。
Config:
    データの前処理パラメータ(VAL_SPLIT)
    
Constructor:
    
'''


class Preprocessor:
    def __init__(self, data_dir="../data/mvtec", batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = config.VAL_SPLIT
        self.rot_angle = config.ROT_ANGLE
        self.w_shift = config.W_SHIFT_RANGE
        self.h_shift = config.H_SHIFT_RANGE
        self.brightness_factor = config.BRIGHTNESS_RANGE
        self.image_size = config.IMG_SIZE

    def get_dataset(self):
        print("Start creating dataset... \n")
        data_dir = pathlib.Path(self.data_dir)

        # Get the list of the product type in the dataset.
        class_names = np.array(sorted([item for item in data_dir.glob("*") if item.is_dir()]))

        # Get the list of file path of a train image in the data_dir.
        list_train = list(data_dir.glob("*/train/good/*.png"))
        list_test = list(data_dir.glob("*/test/*/*.png"))
        num_train = len(list_train)  # the number of train images.
        num_test = len(list_test)

        # Create train/test dataset. Each element in the dataset is path of image file.
        list_train_ds = tf.data.Dataset.list_files(str(data_dir / "*/train/good/*.png"), shuffle=False)
        list_test_ds = tf.data.Dataset.list_files(str(data_dir / "*/test/*/*.png"), shuffle=False)

        # Shuffle these dataset once.
        list_train_ds = list_train_ds.shuffle(buffer_size=num_train, reshuffle_each_iteration=False)
        list_test_ds = list_test_ds.shuffle(buffer_size=num_test, reshuffle_each_iteration=False)

        # Split these dataset for following division of dataset(Refer to the README.md)
        # VAL_SPLIT is set 0.1 as default.
        train_size = int(num_train * (1 - self.val_split))
        fine_size = int((num_train + num_test) * self.val_split)
        test_size = int(num_test * (1 - self.val_split))

        # Create the trai/fine-tuning/test dataset.
        train_ds = list_train_ds.take(train_size)
        fine_ds = list_train_ds.skip(train_size).concatenate(list_test_ds.take(num_test - test_size))
        test_ds = list_test_ds.skip(num_test - test_size)

        print(f"The number of train data: {train_ds.cardinality().numpy()}")
        print(f"The number of fine-tuning data: {fine_ds.cardinality().numpy()}")
        print(f"The number of test data: {test_ds.cardinality().numpy()}")

        train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        finetune_ds = fine_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, finetune_ds, test_ds

    def resize_and_rescale(self, ds):
        resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.image_size, self.image_size),
            layers.Rescaling(1./255)
        ])
    def data_augmentation(self, ds):
        augments = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(self.rot_angle),
            layers.RandomTranslation(self.h_shift, self.w_shift,
                                     fill_mode="constant", fill_value=0.0),
            layers.RandomBrightness(self.brightness_factor, value_range=[0.0, 1.0]),
        ])

        aug_ds = ds.map(
            lambda x,y: (augments(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def get_label(file_path):
    # Convert the path to a list of path components. (ex. ['.'. 'data', 'mvtec', 'bottle'])
    parts = tf.strings.split(file_path, os.path.sep)
    label = int(parts[-2] == 'good')
    return label


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    return img


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the files as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label



preprocessor = Preprocessor()
train_ds, fine_ds, test_ds = preprocessor.get_dataset()

train_ds = preprocessor.resize_and_rescale(train_ds)
fine_ds = preprocessor.resize_and_rescale(fine_ds)
test_ds = preprocessor.resize_and_rescale(test_ds)

aug_ds = preprocessor.data_augmentation(train_ds)


