import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

import keras
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D,BatchNormalization,Activation,AveragePooling2D
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers, models
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16, ResNet50, DenseNet121


def count_conv_params_flops(conv_layer):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    '''
    Arguments:
        conv layer 
    Return:
        Number of Parameters, Number of Flops
    '''

    out_shape = conv_layer.output_shape

    n_cells_total = np.prod(out_shape[1:-1])

    n_conv_params_total = conv_layer.count_params()
    # print(n_conv_params_total,len(conv_layer.get_weights()[0]),)
    conv_flops =  (n_conv_params_total * n_cells_total - len(conv_layer.get_weights()[1]) *n_cells_total)

 

    return n_conv_params_total, conv_flops


def count_dense_params_flops(dense_layer):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    '''
    Arguments:
      dense layer 
    Return:
        Number of Parameters, Number of Flops
    '''

    out_shape = dense_layer.output_shape
    n_cells_total = np.prod(out_shape[1:-1])

    n_dense_params_total = dense_layer.count_params()

    dense_flops =  (n_dense_params_total - len(dense_layer.get_weights()[1]) * n_cells_total)


    return n_dense_params_total, dense_flops




def count_model_params_flops(model,first_time):

    '''
    Arguments:
        model -> your model
        first_time -> boolean variable
        first_time = True => model is not pruned 
        first_time = False => model is pruned
    Return:
        Number of parmaters, Number of Flops
    '''
    total_params = 0
    total_flops = 0
    model_layers = model.layers
    for index,layer in enumerate(model_layers):
        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            params, flops = count_conv_params_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            params, flops = count_dense_params_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
    return total_params, int(total_flops)


def modMaxPool(model, mod_list):
    for i in range(1, len(model.layers)):
        if mod_list and isinstance(model.layers[-i], MaxPooling2D):
            model.layers[-i].pool_size = mod_list.pop(0)

    model = models.model_from_json(model.to_json())
    return model


def modConv(model, mod_list):
    for i in range(1, len(model.layers)):
        if mod_list and isinstance(model.layers[-i], Conv2D):
            kernel_size, filters = mod_list.pop(0)
            model.layers[-i].kernel_size = kernel_size
            model.layers[-i].filters = filters

    model = models.model_from_json(model.to_json())
    return model


def modFC(model, break_layer, mod_list):
    for i in range(1, len(model.layers)):
        if isinstance(model.layers[-i], break_layer):
            X = model.layers[-i].output
            break

    for t in mod_list[:-1]:
        if isinstance(t, tuple):
            units, dropout = t
            X = Dense(units, activation='relu')(X)
            X = Dropout(dropout)(X)
        else:
            X = Dense(t, activation='relu')(X)

    X = Dense(mod_list[-1], activation='softmax')(X)
    return Model(inputs=model.inputs, outputs=X)


if __name__ == "__main__":
    """
    model = keras.Sequential()
    model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=50, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=10, activation = 'softmax'))
    """
    print("Dataset: CalTech - 101")
    num_classes = 101
    model = VGG16(include_top=True)
    max_mod_list = [2]
    fc_mod_list = [(1024, 0.7), num_classes]
    model = modMaxPool(model, max_mod_list)
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(BO)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    model = VGG16(include_top=True)
    fc_mod_list = [(512, 0.6), num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(RS)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    model = VGG16(include_top=True)
    fc_mod_list = [4096, 4096, num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(Conventional) ({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    print("Dataset: CalTech - 256")
    num_classes = 256
    model = VGG16(include_top=True)
    max_mod_list = [3]
    fc_mod_list = [(1024, 0.6), num_classes]
    model = modMaxPool(model, max_mod_list)
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(BO)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    model = VGG16(include_top=True)
    max_mod_list = [3]
    fc_mod_list = [(1024, 0.3), num_classes]
    model = modMaxPool(model, max_mod_list)
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(RS)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    model = VGG16(include_top=True)
    fc_mod_list = [4096, 4096, num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(Conventional) ({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    print("Dataset: Stanford Dogs")
    num_classes = 120
    model = VGG16(include_top=True)
    fc_mod_list = [num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(BO)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()
    
    model = VGG16(include_top=True)
    fc_mod_list = [(1024, 0.3), (512, 0.5), num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(RS)({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")

    print()
    print()
    print("=" * 40)
    print()
    print()

    model = VGG16(include_top=True)
    fc_mod_list = [4096, 4096, num_classes]
    model = modFC(model, Flatten, fc_mod_list)
    total_parameters, total_flops = count_model_params_flops(model,True)
    print(f"\n\nVGG16(Conventional) ({num_classes}) =>", f"FLOPs: {total_flops};", f"Params: {total_parameters}")
