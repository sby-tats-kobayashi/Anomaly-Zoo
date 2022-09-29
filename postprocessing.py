import os
import time
import numpy as np
import tensorflow as tf
from processing.utils import printProgressBar
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Segmentation Parameters

# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# float + L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# uint8 + L2 (generally uneffective combination)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1

class TensorImages:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        method,
        dtype="float64",
        filenames=None,
    ):

        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim", "mssim"]
        self.method = method
        self.dtype = dtype
        # pixel min and max values of input and reconstruction (pred)
        # depend on preprocessing function, which in turn depends on
        # the model used for training.
        self.vmin = vmin
        self.vmax = vmax
        self.filenames = filenames

        # if grayscale, reduce dim to (samples x length x width)
        if imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]
            self.cmap = "gray"
        # if RGB
        else:
            self.cmap = None

        # compute resmaps
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.scores, self.resmaps = calculate_resmaps(
            self.imgs_input, self.imgs_pred, method, dtype
        )

        # compute maximal threshold based on resmaps
        self.thresh_max = np.amax(self.resmaps)

        # set parameters for future segmentation of resmaps
        if dtype == "float64":
            self.vmin_resmap = 0.0
            self.vmax_resmap = 1.0
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.thresh_step = THRESH_STEP_FLOAT_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.thresh_step = THRESH_STEP_FLOAT_L2

        elif dtype == "uint8":
            self.vmin_resmap = 0
            self.vmax_resmap = 255
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.thresh_step = THRESH_STEP_UINT8_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_UINT8_L2
                self.thresh_step = THRESH_STEP_UINT8_L2

