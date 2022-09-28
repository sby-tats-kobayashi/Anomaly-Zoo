import os
import argparse
import pathlib
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import distutils
import postprocessing
from preprocessing import Preprocessor
from preprocessing import preprocessing_function
from postprocessing import label_images
from utils import printProgressBar
from sklearn.model_selection import confusion_matrix
from test import predict_classes
import config
import logging

'''
Requirements:
    preprocessing
        get_preprocessing_function: get preprocessing function that is different over model or algorithm
        Preprocessor: modify arguments received by constructor
        get_valid_dataset: get validation dataset for specified batch size
    utils
        load_model: get metadata and load model
        progressbar: print progressbar
    test
    train.py
        save: implement save info
        
        
        
    postprocessing
'''

def calculate_largest_areas(resamps, thresholds):

    # initialize the largest areas to an empty list
    largest_areas = []

    # initialize progress bar
    printProgressBar(
        0, len(thresholds), prefix="Progress:", suffix="Complete", length=80
    )

    # それぞれの閾値についてlargest_areaを求める。
    for index, threshold in enumerate(thresholds):

        # segment (threshold) residual maps
        resamps_th = resamps > threshold

        # compute labeled coneccted componets
        _, areas_th = label_images(resamps_th)

        # retrieve the largest area of all resamps for current threshold
        areas_th_total = [item for sublist in areas_th for item in sublist]
        largest_area = np.amax(np.array(areas_th_total))
        largest_areas.append(largest_area)

        # print progress bar
        time.sleep(0.1)
        printProgressBar(
            index + 1, len(thresholds), prefix="Progress", suffix="Complete", length=80
        )

    return largest_areas

def main(args):

    # get validation arguments
    model_path = args.path
    method = args.method
    dtype = args.dtype

    # load model & preprocessing configuration
    model, info, _ = utils.load_model_HDF5(model_path)
    # set parameters
    input_directory = info["data"]["input_directory"]
    architecture = info["model"]["architecture"]
    loss = info["model"]["loss"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]
    nb_validation_images = info["data"]["nb_validation_images"]

    # get the correct preprocessing function
    preprocessing_function = get_preprocessing_function(architecture)

    # load and preprocess validation & finetuning images

    # initialize preprocessor
    preprocessor = Preprocessor(
        input_directory=input_directory,
        rescale=rescale,
        shape=shape,
        color_mode=color_mode,
        preprocessing_function=preprocessing_function
    )

    # get validation dataset
    valid_ds =

    # retrieve preprocessed validation images
