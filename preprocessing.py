import os
import pathlib

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import config

'''
Configで指定するパラメータは基本的に変えなくてよいものにしたい。
また、コンストラクタの引数で受け取るパラメータは利用者/日時によって変わるものにしたい。
Config:
    データの前処理パラメータ(VAL_SPLIT)
    
Constructor:
    
'''

class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.val_split = config.VAL_SPLIT
        self.rot_angle = config.ROT_ANGLE
        self.w_shift = config.W_SHIFT_RANGE
        self.h_shift = config.H_SHIFT_RANGE
        self.brightness_factor = config.BRIGHTNESS_RANGE
        self.image_size = config.IMG_SIZE
        self.train_filename_ds = None
        self.valid_filename_ds = None
        self.test_filename_ds = None
        self.test_classes = None
        self.test_size = None
        self.train_size = None
        self.valid_size = None

    def load_dataset(self):
        print("Start creating dataset... \n")
        data_dir = pathlib.Path(self.data_dir)

        # Get the list of the product type in the dataset.
        class_names = np.array(sorted([item for item in data_dir.glob("*") if item.is_dir()]))
        class_dict = dict(zip(class_names, range(len(class_names))))

        # Get the list of file path of a train image in the data_dir.
        list_train = list(data_dir.glob("*/train/good/*.png"))
        list_test = list(data_dir.glob("*/test/*/*.png"))
        num_train = len(list_train)  # the number of train images.
        num_test = len(list_test)

        # Create train/test dataset. Each element in the dataset is path of image file.
        list_train_ds = tf.data.Dataset.list_files(str(data_dir / "*/train/good/*.png"), shuffle=False)
        list_test_ds = tf.data.Dataset.list_files(str(data_dir / "*/test/*/*.png"), shuffle=False)

        # Shuffle train dataset for validation split
        list_train_ds = list_train_ds.shuffle(buffer_size=num_train, reshuffle_each_iteration=False)

        # Split these dataset for following division of dataset(Refer to the README.md)
        # VAL_SPLIT is set 0.1 as default.
        self.train_size = int(num_train * (1 - self.val_split))
        self.valid_size = int(num_train * self.val_split)
        self.test_size = int(num_test * (1 - self.val_split))

        # Create the train/fine-tuning/test dataset.
        self.train_filename_ds = list_train_ds.take(self.train_size)
        self.valid_filename_ds = list_train_ds.skip(self.train_size)
        self.test_filename_ds = list_test_ds.skip(num_test - self.test_size)

        # Create the class label for valid and test dataset
        self.test_classes = self.test_filename_ds.map(lambda x: tf.strings.split(x, os.path.sep)[-2])
        self.test_indices = range(len(self.test_filename_ds))

        print(f"The number of train data: {self.train_filename_ds.cardinality().numpy()}")
        print(f"The number of validation data: {self.valid_filename_ds.cardinality().numpy()}")
        print(f"The number of test data: {self.test_filename_ds.cardinality().numpy()}")

    def get_train_dataset(self, batch_size, color):
        """Get training dataset.

        Args:
            batch_size(int): batch size
            color(str): color mode of image, Supported 'gray', 'rgb'

        Returns:
            tf.data.Dataset: train dataset

        Note:
            This function applies data augmentation.
        """

        assert self.train_filename_ds, \
            "Maybe your dataset isn't loaded, Please add this line before call this function " \
            "'preprocessor.load_dataset()'"

        # cache train dataset
        train_ds = self.train_filename_ds.cache()

        # apply batch process and data augmentation
        train_ds = self._convert_path_to_image(train_ds)
        train_ds = self._resize_and_rescale(train_ds)
        train_ds = self._get_batch_dataset(train_ds, batch_size, shuffle=True)
        train_ds = self._data_augmentation(train_ds)
        train_ds = self._prepare_reconstruction_label(train_ds)

        if color == "gray":
            train_ds = self._rgb_to_gray(train_ds)

        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds

    def get_valid_dataset(self, batch_size, color):
        """Get validation dataset.

        Args:
            batch_size(int): batch size
            color(str): color mode of image, Supported 'gray', 'rgb'

        Returns:
            tf.data.Dataset: validation dataset

        Note:
            This function doesn't apply data augmentation.
        """

        assert self.valid_filename_ds, \
            "Maybe your dataset isn't loaded, Please add this line before call this function " \
            "'preprocessor.load_dataset()'"

        # apply batch transform and data augmentation
        valid_ds = self.valid_filename_ds.cache()
        valid_ds = self._convert_path_to_image(valid_ds)
        valid_ds = self._resize_and_rescale(valid_ds)
        valid_ds = self._get_batch_dataset(valid_ds, batch_size, shuffle=False)
        if color == "gray":
            valid_ds = self._rgb_to_gray(valid_ds)
        valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return valid_ds

    def get_test_dataset(self, batch_size, color):
        """Get test dataset.

        Args:
            batch_size(int): batch size
            color(str): color mode of image, Supported 'gray', 'rgb'

        Returns:
            tf.data.Dataset: test dataset

        Note:
            This function doesn't apply data augmentation.
        """

        assert self.test_filename_ds, \
            "Maybe your dataset isn't loaded, Please add this line before call this function " \
            "'preprocessor.load_dataset()'"

        # apply batch transform and data augmentation
        test_ds = self.test_filename_ds.cache()
        test_ds = self._convert_path_to_image(test_ds)
        test_ds = self._resize_and_rescale(test_ds)
        test_ds = self._get_batch_dataset(test_ds, batch_size, shuffle=False)
        if color == "gray":
            test_ds = self._rgb_to_gray(test_ds)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return test_ds

    @staticmethod
    def _get_batch_dataset(ds, batch_size, shuffle=False):
        """Get batch processed dataset.
        Args:
            ds(tf.data.Dataset): dataset that will be batch processed
            batch_size(int): batch size
            shuffle(bool): Whether to do shuffle on dataset

        Returns:
            tf.data.Dataset: batch processed dataset
        """
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        return ds.batch(batch_size)

    @staticmethod
    def _rgb_to_gray(ds):
        """Convert all images in the dataset to grayscale.
        Args:
            ds(tf.data.Dataset): dataset that will be converted

        Returns:
            tf.data.Dataset: batch processed dataset

        Notes:
            This function convert both of  element (Image, Image) to grayscale. If each element of dataset has (Image, label(not 4D Tensor)),
            then convert to only first Image to grayscale.
        """
        # If each element of dataset has (X, X)
        rgb_to_gray = tf.image.rgb_to_grayscale
        if len(next(ds.as_numpy_iterator())[1].shape) == 4:
            ds = ds.map(lambda x, y: (rgb_to_gray(x), rgb_to_gray(y)), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (rgb_to_gray(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    def _data_augmentation(self, ds):
        augments = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(self.rot_angle),
            layers.RandomTranslation(self.h_shift, self.w_shift,
                                     fill_mode="constant", fill_value=0.0),
            layers.RandomBrightness(self.brightness_factor, value_range=[0.0, 1.0]),
        ])

        ds = ds.map(
            lambda x, y: (augments(x, training=True)), num_parallel_calls=tf.data.AUTOTUNE
        )

        return ds

    @staticmethod
    def _prepare_reconstruction_label(ds):
        """Copy input image as reconstruction label.

        Returns:
            tf.data.Dataset: each element has (input, copy of input)

        """
        ds = ds.map(
            lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds

    def _resize_and_rescale(self, ds):
        resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.image_size, self.image_size),
            layers.Rescaling(1./255)
        ])

        ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    def _convert_path_to_image(self, ds):
        # Convert a file path to an image.
        ds = ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        return ds


    def _get_label(self, file_path):
        # Convert the path to a list of path components. (ex. ['.'. 'data', 'mvtec', 'bottle'])
        parts = tf.strings.split(file_path, os.path.sep)
        label = int(parts[-2] == 'good')
        return label

    def _decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)
        return img
    
    def _process_path(self, file_path):
        label = self._get_label(file_path)
        # Load the raw data from the files as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

    def get_preprocessing_function(architecture):
        """Get preprocessing function for passing certain architecture.

        Args:
            architecture(str): model architecture you want to use

        Returns:
            func: preprocessing function

        """
        if architecture in ['mvtecCAE', 'baselineCAE', 'inceptionCAE', 'resnetCAE']:
            preprocessing_function = None
        return preprocessing_function

    def get_number_train_images(self):
        return self.train_size
    def get_number_valid_images(self):
        return self.valid_size

    def get_number_test_images(self):
        return self.test_size

    def ds_to_ndarray(self, ds):
        """This function process dataset to a batch dataset before convert dataset to numpy ndarray.

        Args:
            ds(tf.data.Dataset): dataset

        Returns:
            Ndarray: numpy ndarray
        """

        # Dataset which has size 2 tuple is already batch processed (Image, Image).
        if len(ds) == 2:
            return next(ds.as_numpy_iterator())[0]
        else:
            return next(self._get_batch_dataset(ds, len(ds)).as_numpy_iterator())





preprocessor = Preprocessor("./data/mvtec")
preprocessor.load_dataset()

nb_valid = preprocessor.get_number_valid_images()
nb_test = preprocessor.get_number_test_images()

valid_ds = preprocessor.get_valid_dataset(nb_valid, "gray")
valid_input = preprocessor.ds_to_ndarray(valid_ds)
filenames_val = preprocessor.ds_to_ndarray(preprocessor.valid_filename_ds)

fine_ds = preprocessor.get_test_dataset(preprocessor.test_size, "rgb")
fine_input = next(fine_ds.as_numpy_iterator())[0]
filenames_test = preprocessor.ds_to_ndarray(preprocessor.test_filename_ds)
index_array = preprocessor.test_indices


print("end")

from sklearn.model_selection import train_test_split
classes = preprocessor.test_classes
classes = preprocessor.ds_to_ndarray(classes)
_, idx_ft, _, classes_ft = train_test_split(index_array, classes, stratify=classes, test_size=preprocessor.val_split, random_state=42)

y_ft_true = np.array(
    [0 if item == 'good' else 1 for item in classes_ft]
)

imgs_ft_input = fine_input[idx_ft]
filenames_ft = filenames_test[idx_ft]
print("**")




# print(next(preprocessor.test_classes.as_numpy_iterator()))
# path = next(iter(preprocessor.test_filename_ds))
# print(tf.strings.split(path, os.path.sep)[-2])

# for batch in valid_ds.as_numpy_iterator():
#     is_nan = np.any(np.isnan(batch))
#     print(is_nan)
#     print(np.min(batch), np.max(batch))
# img, label = next(iter(train_ds))
# print(img[0].shape, label[0].shape)
# print(f"min: {tf.reduce_min(img[0])}")
# print(f"max: {tf.reduce_min(img[0])}")

