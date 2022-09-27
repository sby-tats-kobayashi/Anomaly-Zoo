import os
import argparse
import tensorflow as tf
from models.autoencoder import AutoEncoder
from preprocessing import Preprocessor
# from postprocessing import
import config

def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return

def main(args):
    # get parsed arguments from user
    data_dir = args.input_dir
    architecture = args.architecture
    color_mode = args.color
    loss = args.loss
    batch_size = args.batch
    val_split = config.VAL_SPLIT

    # generate dataset
    preprocessor = Preprocessor(data_dir, batch_size)
    preprocessor.load_dataset()
    train_ds, valid_ds = preprocessor.get_train_dataset(val_split)

    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(data_dir, train_ds, valid_ds, architecture, color_mode, loss,
                              batch_size)

    # train
    autoencoder.fit()

    # save model
    autoencoder.save()

    if args.inspect:

        sample_batch = valid_ds.take(1)
        print(len(sample_batch))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder on image",
        epilog="Example usage: python3 train.py -d ../data/mvtec -b 8 -l l2 -c rgb"
    )
    parser.add_argument(
        "-d",
        "--input_dir",
        type=str,
        required=True,
        help="directory containing training images"
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        metavar="",
        choices=["mvtecCAE", "baselineCAE", "inceptionCAE", "resnetCAE"],
        default="resnetCAE",
        help="architecture of the model to use for training: ['mvtecCAE', 'baselineCAE', 'inceptionCAE', 'resnetCAE', 'skipCAE']"
    )
    parser.add_argument(
        "-c",
        "--color",
        type=str,
        required=False,
        metavar="",
        choices=["rgb", "gray"],
        default="rgb",
        help="color mode for preprocessing images before training: ['rgb', 'grayscale']"
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=False,
        metavar="",
        choices=["mssim", "ssim", "l2"],
        default="mssim",
        help="loss function to use for training: ['mssim', 'ssim', 'l2']"
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        required=False,
        metavar="",
        default=8,
        help="batch size to use for training"
    )

    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="generate inspection plots after training. It can confirm at TensorBoard"
    )

    args = parser.parse_args()
    if tf.test.is_gpu_available():
        print("GPU was detected ...")
    else:
        print("No GPU was detected. Forward calculation can be slow")
    main(args)