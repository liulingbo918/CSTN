from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape,
    Concatenate,
    Multiply,
    Add
)
from keras.layers.convolutional import Convolution2D,SeparableConv2D, Conv2D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as plot


from keras import backend as K

from keras.layers import merge,ConvLSTM2D,Dot
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from models.channel_wise import TimeDistributed
from keras import regularizers

import tensorflow as tf


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("gcc_layers", {})
def gcc_block(block, H, W, inter_channel, feature_channel, output_channel):
    def _shortcut(input, residual):
        result = Add()([input, residual])
        return result
    def embedding2d(block, embedd_id, channels):
        layer = str(block) + "-" + str(embedd_id)
        if layer not in gcc_block.gcc_layers.keys():
            gcc_block.gcc_layers[layer] = Convolution2D(
                filters=channels, kernel_size=(1,1), padding="same", \
                name="NO."+str(block)+"-enbed2d-"+str(embedd_id), activation="relu")
        def f(input):
            input = gcc_block.gcc_layers[layer](input)
            return input
        return f
    def f(oddata_input, poi_input, his_input):
        inputs = []
        if poi_input != None:
            embed_poi = embedding2d(block, "poi", inter_channel)(poi_input)
            inputs.append(embed_poi)
        if his_input != None:
            embed_his = embedding2d(block, "his", inter_channel)(his_input)
            inputs.append(embed_his)
        embed_od = embedding2d(block, "oddata", inter_channel)(oddata_input)
        print("od embed {}".format(embed_od.shape))
        inputs.append(embed_od)
        # for i in inputs:
        #     print(i.shape)
        # raw_input("stop")
        if len(inputs) > 1:
            fusion = Concatenate(axis=1)(inputs)
        else:
            fusion = inputs[0]
        fusion1 = embedding2d(block, "fusion", inter_channel)(fusion)
        fusion1 = Reshape((inter_channel, -1))(fusion1)
        fusion2 = embedding2d(block, "fusion", inter_channel)(fusion)
        fusion2 = Reshape((inter_channel, -1))(fusion2)
        print(fusion.shape)
        sim_fusion = Dot(axes=[1,1], normalize=True)([fusion1, fusion2])
        print("sim_fusion: {}".format(sim_fusion.shape))
        softmax = TimeDistributed(Activation("softmax"))(sim_fusion)
        print(softmax.shape)
        embed_input = embedding2d(block, "input", feature_channel)(oddata_input)
        embed_input = Reshape((feature_channel,-1))(embed_input)

        
        output = Dot(axes=[2,2])([softmax, embed_input])
        embed_y = Permute((2,1))(output)
        embed_y = Reshape((feature_channel, H, W))(embed_y)

        embed_y = embedding2d(block, 'y', output_channel)(embed_y)
        return _shortcut(oddata_input, embed_y)
    return f


@static_var("resnet_count", 0)
def resnet(channels, map_z, map_height, map_width, nb_filter, kernel_size, repetations=1, use_bias=True, name=None):
    def _shortcut(input, residual, nn):
        result = Add()([input, residual])
        return result

    def _bn_relu_conv(nb_filter, kernel_size, block_name=None, bn=False):
        def f(input):
            if bn:
                binput = BatchNormalization(mode=0, axis=1, name=block_name+"batchnorm")(input)
                activation = Activation('relu', name=block_name+"relu")(binput)
            else:
                activation = Activation('relu', name=block_name+"relu")(input)
            return Conv3D(use_bias=use_bias, filters=nb_filter, kernel_size=kernel_size, \
                    padding="same", name=block_name+"conv")(activation)
        return f
    def residual_unit(nb_filter, block_name=''):
        def f(input):
            residual = _bn_relu_conv(nb_filter, kernel_size, block_name+"1_")(input)
            # residual = _bn_relu_conv(nb_filter, kernel_size, block_name+"2_")(residual)
            return _shortcut(input, residual, block_name)
        return f

    resnet.resnet_count += 1
    if name != None:
        model_name = "resnet-" + name
    else:
        model_name = "resnet-" + str(resnet.resnet_count)
    def build_model():
        input = Input(shape=(channels, map_z, map_height, map_width))
        output = Conv3D(
            filters=nb_filter, kernel_size=kernel_size, padding="same", \
            activation="relu", name="{}_first_conv".format(model_name))(input)
        for i in range(repetations):
            output = residual_unit(nb_filter=nb_filter, block_name="{}_block_{}_".format(model_name, i))(output)
        output = Activation("relu")(output)
        model = Model(inputs=input, outputs=output, name=model_name)
        return model
    return build_model()



def clear_graph():
    gcc_block.gcc_layers = {}
    resnet.resnet_count = 0
    return None

def build_model(timestep, map_height, map_width, weather_dim, meta_dim):
    K.set_image_data_format("channels_first")

    # define input data process operation and apply them

    map_hw = map_height * map_width
    main_inputs = []
    oddata_input = Input(shape=(timestep, map_hw, map_height, map_width), name="oddata_input_%d"%timestep)
    main_inputs.append(oddata_input)

    if weather_dim > 0:
        weather_input = Input(shape=(timestep, weather_dim), name="weather_input")
        main_inputs.append(weather_input)
    if meta_dim > 0:
        meta_input = Input(shape=(timestep, meta_dim), name="meta_input")
        main_inputs.append(meta_input)

    print(oddata_input.shape)
    oddate_times = [Lambda(lambda x: x[:,i,...])(oddata_input) for i in range(oddata_input.shape[1])]
    temporal_inputs = [oddate_times]
    if weather_dim > 0:
        weather_times = [Lambda(lambda x: x[:,i,...])(weather_input) for i in range(weather_input.shape[1])]
        temporal_inputs.append(weather_times)
    if meta_dim > 0:
        meta_times = [Lambda(lambda x: x[:,i,...])(meta_input) for i in range(meta_input.shape[1])]
        temporal_inputs.append(meta_times)


    # define LSC components and apply every time step

    od_encoded_seq = []
    od_time_feature = []


    # o-view CNN
    _o_stream = Reshape((1, map_hw, map_height, map_width))
    _o_encoder = resnet(1, map_hw, map_height, map_width, 16, kernel_size=(1,3,3), repetations=2, name="o_encoder")
    _o_shape = Reshape((-1, map_height, map_width))
    _o_embed = Conv2D(filters=32, kernel_size=(1,1), padding="same", activation="relu", name="o_embed")

    # d-view CNN
    _d_stream = Reshape((1, map_height, map_width, map_hw))
    _d_encoder = resnet(1, map_height, map_width, map_hw, 16, kernel_size=(3,3,1), repetations=2, name="d_encoded")
    _d_shape = Reshape((-1, map_height, map_width))
    _d_embed = Conv2D(filters=32, kernel_size=(1,1), padding="same", activation="relu", name="d_embed")

    # od-view fusion
    _concat = Concatenate(axis=1)
    _all_embed = Conv2D(filters=32, kernel_size=(1,1), padding="same", activation="relu", name="spatial_embed")

    # weather embedding
    _fc_wm_1 = Dense(32, activation="relu", name="wm_fc1")
    _fc_wm_2 = Dense(16, activation="relu", name="wm_fc2")
    _fc_wm_3 = Dense(8, activation="relu", name="wm_fc3")
    _wm_repeat = RepeatVector(map_hw)
    _wm_permute = Permute((2,1))
    _wm_reshape = Reshape((8, map_height,map_width))

    # fusion among od and weather
    _odwm_concat = Concatenate(axis=1)
    _odwm_embed = Convolution2D(filters=32, kernel_size=(1,1), padding="same", activation="relu", name="fusion_weather")
    _odwm_reshape = Reshape((1, -1, map_height, map_width))


    for step_data in zip(*temporal_inputs):
        # apply above component every time step input
        od = step_data[0]
        o_encoded = _o_stream(od)
        o_encoded = _o_encoder(o_encoded)
        o_encoded = _o_shape(o_encoded)
        o_encoded = _o_embed(o_encoded)

        d_encoded = _d_stream(od)
        d_encoded = _d_encoder(d_encoded)
        d_encoded = _d_shape(d_encoded)
        d_encoded = _d_embed(d_encoded)

        encoded = _concat([o_encoded, d_encoded])
        encoded = _all_embed(encoded)

        w_encode = step_data[1]
        w_encode = _fc_wm_1(w_encode)
        w_encode = _fc_wm_2(w_encode)
        w_encode = _fc_wm_3(w_encode)
        w_encode = _wm_repeat(w_encode)
        w_encode = _wm_permute(w_encode)
        w_encode = _wm_reshape(w_encode)

        encoded = _odwm_concat([encoded, w_encode])
        encoded = _odwm_embed(encoded)
        encoded = _odwm_reshape(encoded)

        od_encoded_seq.append(encoded)


    od_encoded = Concatenate(axis=1)(od_encoded_seq)
    # LSC modeling
    main_output = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=False, name="convlstm_encoder")(od_encoded)
    # fusion after modeling
    main_output = Convolution2D(filters=map_hw, kernel_size=(1,1), padding="same", activation="relu", name="regressor_first")(main_output)


    # GCC modeling
    gcc = gcc_block(1, map_height, map_width,64,75,map_hw)
    main_output = gcc(main_output, None, None)

    main_output = Conv2D(filters=map_hw, kernel_size=(1,1), padding="same", 
                                name="regressor_second" \
                                # activity_regularizer=regularizers.l1(0.01), \
                                # kernel_regularizer=regularizers.l1(0.01) \
                                )(main_output)

    # final regression    
    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)
    return model

def mse(y_true, y_pred):
    print("mse")
    return K.mean(K.square(y_pred - y_true))

def get_loss():
    return mse