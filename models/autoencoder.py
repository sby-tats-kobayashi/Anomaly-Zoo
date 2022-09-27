import tensorflow as tf
import keras
import pathlib
import os
import datetime
import resnetCAE
import losses
import metrics

from src import config
from src.preprocessing import Preprocessor

class AutoEncoder:

    def __init__(
        self,
        data_dir,
        architecture,
        color_mode,
        loss,
        batch_size=8,
        verbose=True
    ):
        self.data_dir = data_dir
        self.save_dir = None
        self.log_dir = None

        # model and data attributes
        self.architecture = architecture
        self.color_mode = color_mode
        self.loss = loss
        self.batch_size = batch_size
        val_split = config.VAL_SPLIT
        
        # get dataset object
        preprocessor = Preprocessor(
            data_dir=self.data_dir,
            batch_size=self.batch_size,)
        train_ds, _, _ = preprocessor.get_dataset()
        train_ds = preprocessor.get_batch_dataset(train_ds)
        train_ds = preprocessor.data_augmentation(train_ds)
        num_train_ds = int(len(train_ds) * val_split)
        self.train_ds = train_ds.take(num_train_ds)
        self.fine_ds = train_ds.skip(num_train_ds)

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
        if color_mode == "grayscale":
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
            optimizer="adam",
            metrics=self.metrics
        )


    def fit(self):
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1, write_graph=True,
            update_freq='epoch'
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping,
            restore_best_weights=True
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
        
        self.history = self.model.fit(
            x=self.train_ds,
            batch_size=self.batch_size,
            epochs=5,
            verbose=self.verbose,
            callbacks=[tensorboard_cb, early_stopping_cb],
            validation_data=self.fine_ds
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

autoencoder = AutoEncoder("../data/mvtec", "resnetCAE", "rgb", "l2")
autoencoder.fit()