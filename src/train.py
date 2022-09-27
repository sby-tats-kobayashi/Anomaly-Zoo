import os
import tensorflow as tf
from models.autoencoder import AutoEncoder
from preprocessing import Preprocessor
# from postprocessing import

def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return

def main(args):

    #