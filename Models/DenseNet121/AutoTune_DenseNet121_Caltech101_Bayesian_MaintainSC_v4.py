"""
tunes the hyperparameters of the conv and maxpool layers and varies the number of dense layers to be added to the model
with bayesian optimization
maintaining skip connections
"""

import time
import os
import math
import numpy as np
import pandas as pd
import GPyOpt
import keras
from itertools import product
from collections import OrderedDict
from keras.preprocessing import image
from keras import layers, models, optimizers, callbacks, initializers
from keras.applications import DenseNet121

reverse_list = lambda l: list(reversed(l))

DATA_FOLDER = "CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
EPOCHS = 1
RESULTS_PATH = os.path.join("AutoConv_DenseNet121_new1", "AutoConv_DenseNet121_log_" + DATA_FOLDER.split('/')[-1] + "_autoconv_bayes_opt_v1.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.densenet.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.5e-10)

# Creating a CSV file if one does not exist
try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'num_layers_tuned', 'num_fc_layers', 'num_neurons', 'dropouts', 'filter_sizes', 'num_filters', 'stride_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time_taken (s)'])
    log_df = log_df.set_index('index')


# function to modify architecture for current hyperparams for fully connected layers
def get_model_dense(model, dense_params):
    X = model.layers[-1].output

    for j in range(len(dense_params) // 2):
        params_dicts = OrderedDict(filter(lambda x: x[0].split('_')[-1] == str(j + 1), dense_params.items()))
        units, dropout = params_dicts.values()
        X = layers.Dense(int(units), activation='relu', kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(float(dropout))(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=model.inputs, outputs=X)


# function to modify architecture for current hyperparams for fully connected layers
def get_model_conv(model, index, architecture, conv_params, optim_neurons, optim_dropouts, acts):
    X = model.layers[index - 1].output

    for i in range(len(conv_params) // 3):
        global_index = index + i
        if architecture[i] == 'add':
            continue

        params_dicts = OrderedDict(filter(lambda x: x[0].startswith(architecture[i]) and x[0].split('_')[-1] == str(-global_index), conv_params.items()))
        filter_size, num_filters, stride_size = [x for x in params_dicts.values()]

        if architecture[i] == 'conv':
            assert type(model.layers[global_index]) == layers.Conv2D
            model.layers[global_index].trainable = True
            model.layers[global_index].filters = num_filters
            model.layers[global_index].kernel_size = (filter_size, filter_size)
            model.layers[global_index].strides = (stride_size, stride_size)
            model.layers[global_index].kernel_initializer = initializers.he_normal()
        elif architecture[i] == 'maxpool':
            assert type(model.layers[global_index]) == layers.MaxPooling2D
            model.layers[global_index].trainable = True
            model.layers[global_index].pool_size = filter_size
        elif architecture[i] == 'zeropad':
            assert type(model.layers[global_index]) == layers.ZeroPadding2D
            model.layers[global_index].trainable = True
            model.layers[global_index].padding = filter_size
        elif architecture[i] == 'globalavgpool':
            assert type(model.layers[global_index]) == layers.GlobalAveragePooling2D
        elif architecture[i] == 'batch':
            assert type(model.layers[global_index]) == layers.BatchNormalization
        elif architecture[i] == 'activation':
            assert type(model.layers[global_index]) == layers.Activation

    new_model = models.model_from_json(model.to_json())
    X = new_model.layers[-1].output

    for units, dropout in zip(optim_neurons, optim_dropouts):
        X = layers.Dense(units, kernel_initializer='he_normal', activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(dropout)(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=new_model.inputs, outputs=X)


# training the original model
base_model = DenseNet121(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
X = base_model.layers[-2].output
X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
base_model = models.Model(inputs=base_model.inputs, outputs=X)
for i in range(len(base_model.layers)-1):
    base_model.layers[i].trainable = False

base_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

out_start = time.time()
history = base_model.fit_generator(
    train_generator,
    validation_data=valid_generator, epochs=EPOCHS,
    steps_per_epoch=len(train_generator) / batch_size,
    validation_steps=len(valid_generator), callbacks=[reduce_LR]
)
out_end = time.time()

# log the results of training
best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
log_tuple = ('relu', 'he_normal', 0, 1, [], [], [], [], [], history.history['loss'][best_acc_index],  history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index], out_end - out_start)
log_df.loc[log_df.shape[0], :] = log_tuple
log_df.to_csv(RESULTS_PATH)

# freezing the layers of the model
base_model = DenseNet121(include_top=True, weights='imagenet', input_shape=(224, 224, 3)) # because of last GlobalAveragePooling2D layer
base_model = models.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False


## optimize dense layers
best_acc = 0
fc_layer_range = range(1, 3)
dense_opt_s = []
for num_dense in fc_layer_range:
    temp_model = models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
    time.sleep(3)

    # making bounds list
    bounds = []
    for j in range(num_dense):
        bounds.append({'name': 'units_' + str(j + 1), 'type': 'discrete', 'domain': [2 ** k for k in range(6, 11)]})
        bounds.append({'name': 'dropout_' + str(j + 1), 'type': 'discrete', 'domain': np.arange(start=0, stop=1, step=0.1)})

    history = None
    def model_fit_dense(x):
        """
        Callback function for GPyOpt optimizer
        """
        global history

        num_neurons = []
        dropouts = []

        dense_params = OrderedDict()

        j = 0
        while j < x.shape[1]:
            dense_params['units_' + str((j // 2) + 1)] = x[:, j]
            num_neurons.append(int(x[:, j]))
            j += 1
            dense_params['dropout_' + str((j // 2) + 1)] = x[:, j]
            dropouts.append(float(x[:, j]))
            j += 1

        to_train_model = get_model_dense(temp_model, dense_params)
        to_train_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

        # training the modified model
        in_start = time.time()
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=EPOCHS,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )
        in_end = time.time()

        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
        assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
        train_loss = history.history['loss'][best_acc_index]
        train_acc = history.history['acc'][best_acc_index]
        val_loss = history.history['val_loss'][best_acc_index]
        val_acc = history.history['val_acc'][best_acc_index]

        # log the training results
        log_tuple = ('relu', 'he_normal', 0, num_dense + 1, num_neurons, dropouts, [], [], [], train_loss, train_acc, val_loss, val_acc, in_end - in_start)
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        return min(history.history['val_loss'])

    # initialize GPyOpt optimizer
    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit_dense, domain=bounds)
    opt_.run_optimization(max_iter=20)
    dense_opt_s.append(opt_)

    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
    temp_acc = history.history['val_acc'][best_acc_index]

    best_acc = max(temp_acc, best_acc)

    print("Optimized Parameters:")
    for k in range(len(bounds)):
        print(f"\t{bounds[k]['name']}: {opt_.x_opt[k]}")
    print(f"Optimized Function value: {opt_.fx_opt}")

    time.sleep(3)

optim_neurons = []
optim_dropouts = []
req_opt_ = min(dense_opt_s, key=lambda x: x.fx_opt)
k = 0
while k < len(req_opt_.x_opt):
    optim_neurons.append(int(req_opt_.x_opt[k]))
    k += 1
    optim_dropouts.append(float(req_opt_.x_opt[k]))
    k += 1


## optimize conv layers
best_acc = 0
# list of layers not considered in optimization
meaningless = [
    layers.Activation,
    layers.GlobalAveragePooling2D,
    layers.ZeroPadding2D,
    layers.Add,
]

for i in range(1, len(base_model.layers) + 1):
    unfreeze = i
    if type(base_model.layers[-i]) in meaningless:
        continue
    temp_model = models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
    print(f"Tuning last {unfreeze} layers.")
    time.sleep(3)

    # saving the architecture
    temp_arc = []
    temp_acts = []
    for j in range(1, unfreeze + 1):
        if type(temp_model.layers[-j]) == layers.Conv2D:
            temp_arc.append('conv')
        elif type(temp_model.layers[-j]) == layers.MaxPooling2D:
            temp_arc.append('maxpool')
        elif type(temp_model.layers[-j]) == layers.GlobalAveragePooling2D:
            temp_arc.append('globalavgpool')
        elif type(temp_model.layers[-j]) == layers.Add:
            temp_arc.append('add')
        elif type(temp_model.layers[-j]) == layers.BatchNormalization:
            temp_arc.append('batch')
        elif type(temp_model.layers[-j]) == layers.ZeroPadding2D:
            temp_arc.append('zeropad')
        elif type(temp_model.layers[-j]) == layers.Activation:
            temp_arc.append('activation')
            temp_acts.append(temp_model.layers[-j].activation)


    # making bounds list
    bounds = []
    for iter_ in range(len(temp_arc)):
        if temp_arc[iter_] == 'conv':
            bounds.extend(
                [
                    {'name': 'conv_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [2, 3, 5]},
                    {'name': 'conv_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [64, 128, 256, 512]},
                    {'name': 'conv_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )
        elif temp_arc[iter_] == 'maxpool':
            bounds.extend(
                [
                    {'name': 'maxpool_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [2, 3]},
                    {'name': 'maxpool_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'maxpool_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )
        elif temp_arc[iter_] == 'zeropad':
            bounds.extend(
                [
                    {'name': 'zeropad_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1, 3]},
                    {'name': 'zeropad_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'zeropad_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )
        elif temp_arc[iter_] == 'globalavgpool':
            bounds.extend(
                [
                    {'name': 'avgpool_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'avgpool_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'avgpool_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )
        elif type(temp_arc[iter_]) == tuple:
            bounds.extend(
                [
                    {'name': 'activation_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'activation_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'activation_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )
        else:
            bounds.extend(
                [
                    {'name': temp_arc[iter_] + '_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': temp_arc[iter_] + '_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': temp_arc[iter_] + '_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]}
                ]
            )


    history = None
    def model_fit_conv(x):
        """
        Callback function for GPyOpt optimizer
        """
        global history

        filter_sizes = []
        num_filters = []
        stride_sizes = []

        conv_params = OrderedDict()

        j = 0
        while j < x.shape[1]:
            conv_params[temp_arc[j // 3] + '_filter_size_' + str((j // 3) + 1)] = x[:, j]
            if temp_arc[j // 3] not in meaningless:
                filter_sizes.append(int(x[:, j]))
            j += 1
            conv_params[temp_arc[j // 3] + '_num_filters_' + str((j // 3) + 1)] = x[:, j]
            if temp_arc[j // 3] == 'conv':
                num_filters.append(int(x[:, j]))
            j += 1
            conv_params[temp_arc[j // 3] + '_stride_size_' + str((j // 3) + 1)] = x[:, j]
            if temp_arc[j // 3] == 'conv':
                stride_sizes.append(int(x[:, j]))
            j += 1

        to_train_model = get_model_conv(temp_model, -len(conv_params) // 3, reverse_list(temp_arc), conv_params, optim_neurons, optim_dropouts, reverse_list(temp_acts))
        to_train_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

        # train the modified model
        in_start = time.time()
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=EPOCHS,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )
        in_end = time.time()

        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
        assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
        train_loss = history.history['loss'][best_acc_index]
        train_acc = history.history['acc'][best_acc_index]
        val_loss = history.history['val_loss'][best_acc_index]
        val_acc = history.history['val_acc'][best_acc_index]

        # log the results
        log_tuple = ('relu', 'he_normal', unfreeze, len(optim_neurons) + 1, optim_neurons, optim_dropouts, filter_sizes, num_filters, stride_sizes, train_loss, train_acc, val_loss, val_acc, in_end - in_start)
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        return min(history.history['val_loss'])


    # initialize the GPyOpt optimizer
    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit_conv, domain=bounds)
    opt_.run_optimization(max_iter=20)

    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
    temp_acc = history.history['val_acc'][best_acc_index]

    print("Optimized Parameters:")
    for k in range(len(bounds)):
        print(f"\t{bounds[k]['name']}: {opt_.x_opt[k]}")
    print(f"Optimized Function value: {opt_.fx_opt}")

    if temp_acc < best_acc:
        print("Validation Accuracy did not improve")
        print(f"Breaking out at {i} layers")
        break
    best_acc = max(temp_acc, best_acc)

    time.sleep(3)
