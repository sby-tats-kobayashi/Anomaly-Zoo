import os
import pathlib
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import config

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

    def get_dataset(self, subset):

        print("Start creating dataset... \n")
        assert subset in ["training", ""]
        data_dir = pathlib.Path(self.data_dir)

        # Get the list of the product type in the dataset.
        class_names = np.array(sorted([item for item in data_dir.glob("*") if item.is_dir()]))

        # Get the list of file path of a train image in the data_dir.
        list_train = list(data_dir.glob("*/train/good/*.png"))
        list_test = list(data_dir.glob("*/test/*/*.png"))
        num_train = len(list_train)  # the number of train images.
        num_test = len(list_test)

        # Create train/test dataset. Each element in the dataset is path of image file.
        list_train_ds = tf.data.Dataset.list_files(str(data_dir/"*/train/good/*.png"), shuffle=False)
        list_test_ds = tf.data.Dataset.list_files(str(data_dir/"*/test/*/*.png"), shuffle=False)

        # Shuffle these dataset once.
        list_train_ds = list_train_ds.shuffle(buffer_size=num_train, reshuffle_each_iteration=False)
        list_test_ds = list_test_ds.shuffle(buffer_size=num_test, reshuffle_each_iteration=False)

        # Split these dataset for following division of dataset(Refer to the README.md)
        # VAL_SPLIT is set 0.1 as default.
        list_train_ds_s = list_train_ds.take(int(config.VAL_SPLIT * num_train))
        list_train_ds_l = list_train_ds.skip(int(config.VAL_SPLIT * num_train))
        list_test_ds_s = list_test_ds.take(int(config.VAL_SPLIT * num_test))
        list_test_ds_l = list_test_ds.skip(int(config.VAL_SPLIT * num_test))

        # Create the training/finetuning/evaluating dataset.
        train_ds = list_train_ds_l
        finetune_ds = list_train_ds_s.concatenate(list_test_ds_s)
        eval_ds = list_test_ds_l
        
        print(f"The number of training data: {train_ds.cardinality().numpy()}")
        print(f"The number of fine-tuning data: {finetune_ds.cardinality().numpy()}")
        print(f"The number of evaluating data: {eval_ds.cardinality().numpy()}")
    @staticmethod
    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = int(parts[-2] == b'good')
        return label
    def decode_img(self, img):

    def process_path(self, file_path):





preprocessor = Preprocessor()
preprocessor.get_dataset("training")