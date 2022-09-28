import math

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        return 1 - K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return loss


# def mssim_loss(dynamic_range):
#     def loss(imgs_true, imgs_pred):
#         loss = 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
#         clipped_loss = tf.clip_by_value(loss, 0., 1)
#         return clipped_loss
#         # return 1 - tf.reduce_mean(
#         #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range))
#
#     return loss
def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        return 1 - K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return loss


def l2_loss(imgs_true, imgs_pred):
    return tf.nn.l2_loss(imgs_true - imgs_pred)
