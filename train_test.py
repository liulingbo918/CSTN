# -*- coding: utf-8 -*-
""" 
Usage:
    python trainval.py -h
"""
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import cPickle as pickle
import pandas as pd
import math
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,LearningRateScheduler,Callback
from utils.dataset import load_data
import models
from keras.optimizers import Adam, SGD
from keras.utils import plot_model as plot
from keras.utils import multi_gpu_model

import utils
import utils.metrics as Metrics

from keras import backend as K

# uncomment followng to set fix random seed
np.random.seed(1337)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to train and eval')
parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seq_len', type=int, default=5, help='length of import sequence')
parser.add_argument('--pre_train', type=bool, default=False, help='whether to load weights file or not')
parser.add_argument('--weights', type=str, help='weights file to load')
parser.add_argument('--gpus', type=str, help='gpus to use, auto parallelize')


def get_tensorboard(path):
    tensorboard = TensorBoard(log_dir=path)
    return tensorboard

def save_file(file, path):
    rtcode = os.system(" ".join(["cp", file.replace(".pyc", ".py"), path]))
    assert rtcode == 0

def get_decay(base_lr):
    def step_decay_lr(epoch):
        if epoch < 200:
            return base_lr
        else:
            return base_lr * 0.1 

    return step_decay_lr


def show_score(odmax, score, stage):
    print(stage + ' score: %.6f rmse (real): %.6f mape: %.6f' %
          (score[0], score[1], score[2]))

    print('origin rmse (real): %.6f mape: %.6f' %
          (score[3], score[4]))

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = float(K.get_value(optimizer.lr))
        print('LR: {:.6f}'.format(lr))


def rmse(a, b):
    return Metrics.rmse(a, b) * 241.0 / 2.0
def o_rmse(a, b):
    return Metrics.o_rmse(a, b) * 241.0 / 2.0

def train(model, lr, batch_size, seq_len, pre_train, weights, DEMODEL):
    odmax = 241
    use_tensorboard = True
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    parallel = True if gpu_count != 1 else False

    nb_epoch = 200       # number of epoch at training stage
    nb_epoch_cont = 500  # number of epoch at continued training stage

    T = 48  # number of time intervals in one day
    m_patience = 20 # number of epoch to train 
    timestep = seq_len
    map_height, map_width = 15, 5  # grid size

    days_test = 60

    pt = datetime.now().strftime('%m_%d_%H_%M_%S')
    path_model = 'TRAIN/' + pt
    if os.path.isdir(path_model) is False:
        os.makedirs(path_model)
    print("Exp: " + path_model)

    # load data
    print("loading data...")
    '''
        expect:
        X = (sample, timestep, map_height * map_width, map_height, map_width)
        Y = (sample, map_height * map_width, map_height, map_width)
        weather = (sample, timestep, ?)
        meta = (sample, timestep, ?)

        The meta data is not used in this work, but we can explore its effect in future works. 
    '''
    X, Y, weather, meta = load_data(odmax, timestep)
    len_test = T * days_test



    print("nb_epoch: " + str(nb_epoch) + " nb_epoch_cont: " + str(nb_epoch_cont) + " batch_size: " + str(batch_size))
    print("patience: " + str(m_patience) + " lr: " + str(lr) + " seq_len: " + str(timestep))# + '-' + str(len_period) + '-' + str(len_trend))
    print("odmax: " + str(odmax))
    print("{} sample totally. {} for train, {} for test".format(X.shape[0], X.shape[0] - len_test, len_test))

    X_train, X_test = X[:-len_test], X[-len_test:]
    Y_train, Y_test = Y[:-len_test], Y[-len_test:]
    weather_train, weather_test = weather[:-len_test], weather[-len_test:]
    meta_train, meta_test = meta[:-len_test], meta[-len_test:]

    X_train = [X_train, weather_train, meta_train]
    X_test = [X_test, weather_test, meta_test]


    """********************************************************************************************"""
    """ Frist, we train our model with fixed learning rate                                         """
    """********************************************************************************************"""

    model_para = {
        "timestep": timestep,
        "map_height": map_height,
        "map_width": map_width,
        "weather_dim": weather.shape[2],
        "meta_dim": meta.shape[2],
    }
    # Build the model to train in parallel with multi-GPUs or only on GPU
    if parallel:
        model = DEMODEL.build_model(**model_para)
        plot(model, to_file=os.path.join(path_model,'networks.png'), show_shapes=True)
        model.summary()
        train_model = multi_gpu_model(model, gpu_count)

    else:
        model = DEMODEL.build_model(**model_para)
        plot(model, to_file=os.path.join(path_model,'networks.png'), show_shapes=True)
        model.summary()
        train_model = model

    # use the loss define in the model
    loss = DEMODEL.get_loss()
    optimizer = Adam(lr=lr)

    metrics = [ rmse, Metrics.mape,  \
                o_rmse, Metrics.o_mape,
                ]
    train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # load weights to the pre_train model after model compiled
    if pre_train == True:
        model.load_weights(weights, by_name=True, skip_mismatch=True)

    # define callbacks on training
    callbacks = []

    hyperparams_name = 'timestep{}.lr{}'.format(timestep, lr)
    fname_param = os.path.join(path_model, hyperparams_name + '.best.h5')
    lr_logger = SGDLearningRateTracker() # log out the learning rate after a epoch trained
    callbacks.append(lr_logger)
    callbacks.append(EarlyStopping(monitor='val_rmse', patience=m_patience, mode='min'))
    callbacks.append(ModelCheckpoint(
        fname_param, monitor='val_mape', verbose=0, save_best_only=True, mode='min'))

    if use_tensorboard:
        callbacks.append(get_tensorboard(path_model+"/tensorboard-1/"))

    print('=' * 10)
    print("training model...")
    history = train_model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        callbacks=callbacks,
                        verbose=1)

    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    train_model.load_weights(fname_param)
    model.save_weights(fname_param, overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_model, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('evaluating using the model that has the best model on the valid set')

    model.load_weights(fname_param)
    
    score = train_model.evaluate(X_train, Y_train, batch_size=batch_size, verbose=0)
    show_score(odmax, score, "train")
    score = train_model.evaluate(
        X_test, Y_test, batch_size=batch_size, verbose=0)
    show_score(odmax, score, "Test")

    print('=' * 10)


    """********************************************************************************************"""
    """ Second, we train our model with step_decay learning rate                                   """
    """********************************************************************************************"""

    # clear session to rebuild the model, in order to switch optimizor
    K.clear_session()
    DEMODEL.clear_graph()

    # rebuild the model
    if parallel:
        model = DEMODEL.build_model(**model_para)
        train_model = multi_gpu_model(model, gpu_count)
    else:
        model = DEMODEL.build_model(**model_para)
        train_model = model

    loss = DEMODEL.get_loss()
    optimizer = Adam(lr=lr)
    metrics = [ rmse, Metrics.mape, \
                o_rmse, Metrics.o_mape,
                ]
    train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.load_weights(fname_param)

    fname_param_step =  os.path.join(
        path_model, \
        hyperparams_name + '.cont.best.h5.{epoch:03d}-{val_mape:.4f}-{val_rmse:.4f}-{val_o_mape:.4f}-{val_o_rmse:.4f}')
    callbacks_cont = []
    #lr_logger = SGDLearningRateTracker()


    # callbacks_cont.append(lr_logger)
    callbacks_cont.append(LearningRateScheduler(get_decay(lr)))
    callbacks_cont.append(ModelCheckpoint(
        fname_param_step, monitor='val_mape', verbose=0, save_best_only=False, period=1, save_weights_only=True, mode='min'))
    if use_tensorboard:
        callbacks_cont.append(get_tensorboard(path_model+"/tensorboard-2/"))

    history = train_model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch_cont, 
                        batch_size=batch_size,
                        callbacks=callbacks_cont, 
                        validation_data=(X_test, Y_test),
                        verbose=1)

    pickle.dump((history.history), open(os.path.join(
        path_model, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        path_model, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    model.load_weights(fname_param)
    model.save_weights(fname_param, overwrite=True) # save the origin model weights instead of the paralleled one

    print('=' * 10)
    print('evaluating using the final model')
    score = train_model.evaluate(X_train, Y_train, batch_size=32, verbose=0)
    show_score(odmax, score, "train")
    score = train_model.evaluate(
        X_test, Y_test, batch_size=32, verbose=0)
    show_score(odmax, score, "test")

if __name__ == '__main__':
    args = parser.parse_args()
    model = args.model
    lr = args.lr
    batch_size = args.batch_size
    pre_train = args.pre_train
    weights = args.weights
    seq_len = args.seq_len
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    exec "import models.{} as DEMODEL".format(model)


    train(model, lr, batch_size, seq_len, pre_train, weights, DEMODEL)
