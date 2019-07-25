# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
from numpy import newaxis
import pandas as pd
import math
from datetime import datetime



np.random.seed(1337)  # for reproducibility

map_height = 15
map_width = 5
map_hw = map_height * map_width


def load_data(odmax, timestep):
    '''
        expectation:
        X = (sample, timestep, map_height * map_width, map_height, map_width)
        Y = (sample, map_height * map_width, map_height, map_width)
        weather = (sample, timestep, ?)
        meta = (sample, timestep, ?)
    '''
    oddata = "./NYC-TOD/oddata.npy"
    weather = "./NYC-TOD/weather.npy"
    meta = "./NYC-TOD/meta.npy"


    print("*************************")
    print("load data")
    print("*************************")
    oddata = np.load(oddata)[()]
    weather = np.load(weather)[()]
    meta = np.load(meta)[()]

    print("*************************")
    print("load data done")
    print("*************************")
    print("*************************")
    print("generate sequence")
    print("*************************")

    sets = len(oddata.keys())
    for i in oddata.keys():
        oddata[i] = oddata[i] * 2.0 / odmax - 1.0

    o = []
    w = []
    m = []
    y = []


    for i in oddata.keys():
        oddata_set = oddata[i]
        weather_set = weather[i]
        meta_set = meta[i]
        o.append( \
            np.concatenate([oddata_set[i:i-timestep, newaxis, ...] for i in range(timestep)], axis=1))
        y.append(oddata_set[timestep:,...])
        w.append( \
            np.concatenate([weather_set[i:i-timestep, newaxis, ...] for i in range(timestep)], axis=1))
        m.append( \
            np.concatenate([meta_set[i:i-timestep, newaxis, ...] for i in range(timestep)],axis=1))
    o = np.concatenate(o)
    y = np.concatenate(y)
    w = np.concatenate(w)
    m = np.concatenate(m)
    sample_num = o.shape[0]

    print("*************************")
    print("generate sequence done")
    print("*************************")
    print(o.shape)
    print(y.shape)
    print(w.shape)
    print(m.shape)
    return o, y, w, m

if __name__ == '__main__':
    load_data(241,5)
