"""
tunes the hyperparameters of the conv and maxpool layers and varies the number of dense layers to be added to the model
with random search
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
from keras.applications import ResNet50
from keras.utils import plot_model

reverse_list = lambda l: list(reversed(l))

DATA_FOLDER = "CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
EPOCHS = 50
RESULTS_PATH = os.path.join("AutoConv_ResNet50_new", "AutoTune_AutoConv_ResNet50_log_" + DATA_FOLDER.split('/')[-1] + "_autoconv_bayes_opt_SKIP_v1.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.5e-10)

# Creating a CSV file if one does not exist
try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'num_layers_tuned', 'num_fc_layers', 'num_neurons', 'dropouts', 'filter_sizes', 'num_filters', 'stride_sizes', 'pool_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time_taken (s)'])
    log_df = log_df.set_index('index')


# function to modify architecture for current hyperparams
def get_model_dense(model, dense_params):
    X = model.layers[-1].output

    for j in range(len(dense_params) // 2):
        params_dicts = OrderedDict(filter(lambda x: x[0].split('_')[-1] == str(j + 1), dense_params.items()))
        print(params_dicts)
        units, dropout = params_dicts.values()
        X = layers.Dense(int(units), activation='relu', kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(float(dropout))(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=model.inputs, outputs=X)


# function to modify architecture for current hyperparams
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
        X = layers.Dropout(float(dropout))(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=new_model.inputs, outputs=X)


# training the original model
base_model = ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
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

# log the results
best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
log_tuple = ('relu', 'he_normal', 0, 1, [], [], [], [], [], history.history['loss'][best_acc_index],  history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index], out_end - out_start)
log_df.loc[log_df.shape[0], :] = log_tuple
log_df.to_csv(RESULTS_PATH)

# freezing the layers of the model
base_model = ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3)) # because of last GlobalAveragePooling2D layer
base_model = models.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False


## optimize dense layers
# search spaces for each kind of hyperparam
fc_layer_range = range(1, 3)
units_space = [2 ** j for j in range(6, 11)]
dropouts_space = [0.1 * j for j in range(10)]
best_acc = 0
best_dense_params = None

for num_dense in fc_layer_range:
    for _ in range(15):
        print(f"Currently trying FC architecture:")
        curr_units = random.sample(units_space, num_dense)
        curr_dropouts = random.sample(dropouts_space, num_dense)
        print(f"\tUnits: {curr_units}")
        print(f"\tDropouts: {curr_dropouts}")

        to_train_model = get_model_dense(base_model, [curr_units, curr_dropouts])
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
        temp_acc = history.history['val_acc'][best_acc_index]

        # log the results
        log_tuple = ('relu', 'he_normal', None, num_dense, curr_units, curr_dropouts, None, None, None, None, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index], in_end - in_start)
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        if temp_acc > best_acc:
            best_dense_params = [curr_units, curr_dropouts]
        best_acc = max(temp_acc, best_acc)


## optimize conv layers
optim_neurons, optim_dropouts = best_dense_params
# list of layers not considered in optimization
meaningless = [
    layers.Activation,
    layers.GlobalAveragePooling2D,
    layers.ZeroPadding2D,
    layers.Add,
]
# search spaces for each kind of hyperparam
filter_size_space = [2, 3, 5]
num_filter_space = [64, 128, 256, 512]
pool_size_space = [2, 3]
pad_size_space = list(range(1, 5))
for unfreeze in range(1, len(base_model.layers) + 1):
    print(f"Tuning last {unfreeze} layers.")
    if type(base_model.layers[-unfreeze]) in meaningless:
        continue

    for _ in range(15):
        temp_model = models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
        time.sleep(3)

        curr_filter_size = []
        curr_num_filters = []
        curr_pool_size = []
        curr_acts = []
        curr_pad = []

        # saving the architecture
        temp_arc = []
        for j in range(1, unfreeze + 1):
            if type(temp_model.layers[-j]) == layers.Conv2D:
                temp_arc.append('conv')
                curr_filter_size.append(random.sample(filter_size_space, 1)[0])
                curr_num_filters.append(random.sample(num_filter_space, 1)[0])
            elif type(temp_model.layers[-j]) == layers.MaxPooling2D:
                temp_arc.append('maxpool')
                curr_pool_size.append(random.sample(pool_size_space, 1)[0])
            elif type(temp_model.layers[-j]) == layers.GlobalAveragePooling2D:
                temp_arc.append('globalavgpool')
            elif type(temp_model.layers[-j]) == layers.Activation:
                temp_arc.append('activation')
                curr_acts.append(temp_model.layers[-j].activation)
            elif type(temp_model.layers[-j]) == layers.Add:
                temp_arc.append('add')
            elif type(temp_model.layers[-j]) == layers.BatchNormalization:
                temp_arc.append('batch')
            elif type(temp_model.layers[-j]) == layers.ZeroPadding2D:
                temp_arc.append('zeropad')
                curr_pad.append(random.sample(pad_size_space, 1)[0])

        to_train_model = get_model_conv(temp_model, -unfreeze, reverse_list(temp_arc), reverse_list(curr_num_filters), reverse_list(curr_filter_size), reverse_list(curr_pool_size), reverse_list(curr_acts), reverse_list(curr_pad), optim_neurons, optim_dropouts)
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
        temp_acc = history.history['val_acc'][best_acc_index]

        # log the results
        log_tuple = ('relu', 'he_normal', unfreeze, len(optim_neurons), optim_neurons, optim_dropouts, curr_filter_size, curr_num_filters, [1] * len(curr_num_filters), curr_pool_size, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index], in_end - in_start)
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        best_acc = max(best_acc, temp_acc)
