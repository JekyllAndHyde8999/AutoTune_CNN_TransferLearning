import time
import os
import math
import numpy as np
import pandas as pd
import GPyOpt
import keras
import random
import math
from itertools import product, combinations
from collections import OrderedDict
from keras.preprocessing import image
from keras import layers, models, optimizers, callbacks, initializers
# from keras.applications import AlexNet
# from alex import AlexNet

reverse_list = lambda l: list(reversed(l))
load = lambda : models.load_model("alexnet_imagenet.h5")

DATA_FOLDER = "/media/kishank/Disk 3/shabbeer/CalTech101"
# DATA_FOLDER = "CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
EPOCHS = 50
RESULTS_PATH = os.path.join("AutoConv_AlexNet_new", "AutoFCL_AutoConv_AlexNet_randomsearch_log_" + DATA_FOLDER.split('/')[-1] + "_autoconv_v10.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    data_format="channels_first") # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.5e-10)

# adagrad optimizer
ADAM = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)


try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'num_layers_tuned', 'num_fc_layers', 'num_neurons', 'dropouts', 'filter_sizes', 'num_filters', 'stride_sizes', 'pool_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')


def upsample(shape, target_size=5):
    upsampling_factor = math.ceil(target_size / shape[1])
    return layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))


def get_model_conv(model, index, architecture, num_filters, filter_sizes, pool_sizes, acts, zero_pads, optim_neurons, optim_dropouts):
    # assert optim_neurons and optim_dropouts, "No optimum architecture for dense layers is provided."
    X = model.layers[index - 1].output
    print(type(model.layers[index - 1]))

    # for i in range(len(conv_params) // 3):
    for i in range(len(architecture)):
        global_index = index + i
        if architecture[i] == 'add':
            continue
        print(f"global_index: {global_index}")
        print(f"Layer: {architecture[i]}")

        if architecture[i] == 'conv':
            assert type(model.layers[global_index]) == layers.Conv2D
            num_filter = num_filters.pop(0)
            filter_size = filter_sizes.pop(0)
            try:
                X = layers.Conv2D(filters=int(num_filter), kernel_size=(int(filter_size), int(filter_size)), kernel_initializer='he_normal', activation='relu')(X)
            except:
                X = upsample(X.shape)(X)
                X = layers.Conv2D(filters=int(num_filter), kernel_size=(int(filter_size), int(filter_size)), kernel_initializer='he_normal', activation='relu')(X)
        elif architecture[i] == 'maxpool':
            assert type(model.layers[global_index]) == layers.MaxPooling2D
            pool_size = pool_sizes.pop(0)
            X = layers.MaxPooling2D(pool_size=int(pool_size))(X)
        elif architecture[i] == 'globalavgpool':
            assert type(model.layers[global_index]) == layers.GlobalAveragePooling2D
            # pool_size = pool_sizes.pop(0)
            X = layers.GlobalAveragePooling2D()(X)
        elif architecture[i] == 'batch':
            assert type(model.layers[global_index]) == layers.BatchNormalization
            X = layers.BatchNormalization()(X)
        elif architecture[i] == 'activation':
            assert type(model.layers[global_index]) == layers.Activation
            X = layers.Activation(acts.pop(0))(X)

    X = layers.Flatten()(X)

    for units, dropout in zip(optim_neurons, optim_dropouts):
        X = layers.Dense(units, kernel_initializer='he_normal', activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(float(dropout))(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=model.inputs, outputs=X)


# base_model = AlexNet(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
base_model = load()
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False

## training original model
X = base_model.layers[-2].output
X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
to_train_model = models.Model(inputs=base_model.inputs, outputs=X)
to_train_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
# to_train_model.summary()
history = to_train_model.fit_generator(
    train_generator,
    validation_data=valid_generator, epochs=EPOCHS,
    steps_per_epoch=len(train_generator) / batch_size,
    validation_steps=len(valid_generator), callbacks=[reduce_LR]
)

# base_model = AlexNet(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
base_model = load()
base_model = models.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False

best_acc = 0
# optim_neurons, optim_dropouts = [], []
meaningless = [
    layers.Activation,
    layers.GlobalAveragePooling2D,
    # layers.MaxPooling2D,
    layers.ZeroPadding2D,
    layers.Add,
]
## optimize conv layers
filter_size_space = [1, 3]
num_filter_space = [32, 64, 128, 256]
pool_size_space = [2, 3]
units_space = [2 ** j for j in range(6, 11)]
dropouts_space = np.arange(0, 1, step=0.1)
pad_size_space = list(range(1, 5))
for unfreeze in range(1, len(base_model.layers) + 1):
    if type(base_model.layers[-unfreeze]) in meaningless:
        continue

    for _ in range(20):
        temp_model = models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
        print(f"Tuning last {unfreeze} layers.")
        time.sleep(3)

        curr_filter_size = []
        curr_num_filters = []
        curr_pool_size = []
        curr_acts = []
        curr_pad = []
        curr_units  []
        curr_dropouts = []
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
                # curr_pool_size.append(random.sample(pool_size_space, 1)[0])
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
            elif type(temp_model.layers[-j]) == layers.Dense:
                temp_arc.append('dense')
                curr_units.append(random.sample(units_space, 1)[0])
                curr_dropouts.append(random.sample(dropouts_space, 1)[0])

        print(f"temp_arc: {temp_arc}")

        to_train_model = get_model_conv(temp_model, -unfreeze, reverse_list(temp_arc), reverse_list(curr_num_filters), reverse_list(curr_filter_size), reverse_list(curr_pool_size), reverse_list(curr_acts), reverse_list(curr_pad), curr_units, curr_dropouts)
        to_train_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        # to_train_model.summary()
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=EPOCHS,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )

        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
        temp_acc = history.history['val_acc'][best_acc_index]

        log_tuple = ('relu', 'he_normal', unfreeze, len(curr_units), curr_units, curr_dropouts, curr_filter_size, curr_num_filters, [1] * len(curr_num_filters), curr_pool_size, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        best_acc = max(best_acc, temp_acc)
