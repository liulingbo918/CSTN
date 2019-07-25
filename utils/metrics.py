import numpy as np
from keras import backend as K
import tensorflow as tf
# from Dataset import channel_wise_max

_MAX = 241.0



def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def mape(y_true, y_pred):
    y_true = (y_true + 1) * _MAX / 2.0
    y_pred = (y_pred + 1) * _MAX / 2.0
    y_pred = tf.round(y_pred)
    mask = tf.greater_equal(y_true, K.constant(5.0))
    return tf.reduce_mean(tf.boolean_mask((tf.abs(y_true - y_pred)/y_true), mask))


def o_rmse(y_true, y_pred):
    y_true = K.sum(y_true, axis=1)
    y_pred = K.sum(y_pred, axis=1)
    return mean_squared_error(y_true, y_pred) ** 0.5

def o_mape(y_true, y_pred):
    y_true = (y_true + 1) * _MAX / 2.0
    y_pred = (y_pred + 1) * _MAX / 2.0
    y_true = K.sum(y_true, axis=1)
    y_pred = K.sum(y_pred, axis=1)
    y_pred = tf.round(y_pred)
    mask = tf.greater_equal(y_true, K.constant(5.0))
    return tf.reduce_mean(tf.boolean_mask((tf.abs(y_true - y_pred)/y_true), mask))

def shape1(y_true, y_pred):
    # batch
    return K.shape(y_pred)[0]


def shape2(y_true, y_pred):
    return K.shape(y_pred)[1]


def shape3(y_true, y_pred):
    return K.shape(y_pred)[2]


def shape4(y_true, y_pred):
    return K.shape(y_pred)[3]
