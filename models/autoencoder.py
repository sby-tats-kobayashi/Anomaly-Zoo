import tensorflow as tf
import keras
import pathlib
import numpy as np
import os
import datetime
from models import resnetCAE
from models import losses
from models import metrics

import config
from preprocessing import Preprocessor


class AutoEncoder:

    def __init__(
        self,
        data_dir,
        train_data,
        valid_data,
        architecture,
        color_mode,
        loss,
        batch_size=8,
        verbose=True
    ):
        self.data_dir = data_dir
        self.train_data = train_data
        self.valid_data = valid_data
        self.save_dir = None
        self.log_dir = None

        # model and data attributes
        self.architecture = architecture
        self.color_mode = color_mode
        self.loss = loss
        self.batch_size = batch_size

        # results attributes
        self.hist = None
        self.epochs_trained = None

        if architecture == "resnetCAE":
            self.model = resnetCAE.build_model("resnet18", color_mode)
            self.rescale = resnetCAE.RESCALE
            self.shape = resnetCAE.SHAPE
            self.preprocessing_function = resnetCAE.PREPROCESSING_FUNCTION
            self.preprocessing = resnetCAE.PREPROCESSING
            self.vmin = resnetCAE.VMIN
            self.vmax = resnetCAE.VMAX
            self.dynamic_range = resnetCAE.DYNAMIC_RANGE
            print(self.dynamic_range)

        # Training hyperparameters
        self.early_stopping = config.EARLY_STOPPING  # patience

        # verbosity
        self.verbose = verbose
        # if verbose:
        #     self.model.summary()

        # set loss function
        if loss == "ssim":
            self.loss_function = losses.ssim_loss(self.dynamic_range)
        elif loss == "mssim":
            self.loss_function = losses.mssim_loss(self.dynamic_range)
        elif loss == "l2":
            self.loss_function = losses.l2_loss

        # set metrics to monitor training
        if color_mode == "gray":
            self.metrics = [metrics.ssim_metric(self.dynamic_range)]
            self.hist_keys = ("loss", "val_loss", "ssim", "val_ssim")
        if color_mode == "rgb":
            self.metrics = [metrics.mssim_metric(self.dynamic_range)]
            self.hist_keys = ("loss", "val_loss", "mssim", "val_mssim")

        # create directory to save model and logs
        self.create_save_dir()

        # compile model
        self.model.compile(
            loss=self.loss_function,
            optimizer=tf.optimizers.Adam(),
            metrics=self.metrics
        )


    def fit(self):

        print("----test----")
        pred_x = next(iter(self.train_data))[0]
        true_x = next(iter(self.train_data))[0]
        metric = self.metrics[0](pred_x, true_x)
        print(metric)

        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1, write_graph=True,
            update_freq='epoch'
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping,
            restore_best_weights=True
        )

        terminate_nan_cb = tf.keras.callbacks.TerminateOnNaN()

        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )

        # checkpoint_cb = keras.callbacks.ModelCheckpoint(
        #     filepath=,
        #     monitor="val_loss",
        #     save_best_only=True
        # )

        # Print command to paste in browser for visualizing in Tensorboard
        print(
            "run the following command in a seperate terminal to monitor training on tensorboard:"
            + "\ntensorboard --logdir={}\n".format(self.log_dir)
        )
        
        self.hist = self.model.fit(
            x=self.train_data,
            batch_size=self.batch_size,
            epochs=100,
            verbose=self.verbose,
            callbacks=[tensorboard_cb, early_stopping_cb, reduce_lr_cb],
            validation_data=self.valid_data
        )
    def create_save_dir(self):
        # create a directory to save model
        now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_dir = os.path.join(
            os.getcwd(),
            "saved_models",
            self.data_dir,
            self.architecture,
            self.loss,
            now,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        # create a log directory for tensorboard
        log_dir = os.path.join(save_dir, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def create_model_name(self):
        epochs_trained = self.get_best_epoch()
        model_name = self.architecture + f"_b{self.batch_size}_e{epochs_trained}.hdf5"
        return model_name

    def save(self):
        # save model
        save_dir = os.path.join(self.save_dir, self.create_model_name())
        tf.saved_model.save(self.model, save_dir)
        return save_dir

    def get_history_dict(self):
        hist_dict = dict((key, self.hist.history[key]) for key in self.hist_keys)
        return hist_dict

    def get_best_epoch(self):
        hist_dict = self.get_history_dict()
        best_epoch = int(np.argmin(np.array(hist_dict["val_loss"])))
        return best_epoch

    def call(self, inputs):
        return self.model(inputs)

    def __call__(self, inputs):
        return self.call(inputs)
