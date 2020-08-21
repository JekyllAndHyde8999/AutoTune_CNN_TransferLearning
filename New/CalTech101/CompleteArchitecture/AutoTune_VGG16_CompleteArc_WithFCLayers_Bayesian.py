import time
import os
import math
import numpy as np
import pandas as pd
import GPyOpt
import keras
import math
from itertools import product
from collections import OrderedDict
from keras.preprocessing import image
from keras import layers, models, optimizers, callbacks, initializers, activations
from keras.applications import VGG16

reverse_list = lambda l: list(reversed(l))

DATA_FOLDER = "CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
EPOCHS = 50
RESULTS_PATH = os.path.join("AutoConv_VGG16_new1", "Upsampling_AutoConv_VGG16_log_" + DATA_FOLDER.split('/')[-1] + "_autoconv_bayes_opt_v1.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.5e-10)

NUM_HYPERPARAMS = 3

# activations
act_map = [
    activations.relu,
    activations.sigmoid,
    activations.tanh,
    activations.elu,
    activations.selu
]


# Creating a CSV file if one does not exist
try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'num_layers_tuned', 'num_fc_layers', 'num_neurons', 'dropouts', 'filter_sizes', 'num_filters', 'stride_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')


# utility function
def upsample(shape, target_size=5):
    upsampling_factor = math.ceil(target_size / shape[1])
    return layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))


# function to modify architecture for current hyperparams
def get_model_conv(model, index, architecture, conv_params, optim_neurons, optim_dropouts):
    X = model.layers[index - 1].output

    dense_params = OrderedDict(filter(lambda x: x[0].startswith('dense'), conv_params.items()))
    conv_params = OrderedDict(filter(lambda x: not x[0].startswith('dense'), conv_params.items()))

    for i in range(len(conv_params) // NUM_HYPERPARAMS):
        global_index = index + i
        if architecture[i] == 'add':
            continue

        params_dicts = OrderedDict(filter(lambda x: x[0].startswith(architecture[i]) and x[0].split('_')[-1] == str(-global_index), conv_params.items()))
        filter_size, num_filters, stride_size = [x for x in params_dicts.values()]

        if architecture[i] == 'conv':
            assert type(model.layers[global_index]) == layers.Conv2D
            try:
                X = layers.Conv2D(filters=int(num_filters), kernel_size=(int(filter_size), int(filter_size)), kernel_initializer='he_normal', activation=act_map[int(stride_size)])(X)
            except:
                X = upsample(X.shape)(X)
                X = layers.Conv2D(filters=int(num_filters), kernel_size=(int(filter_size), int(filter_size)), kernel_initializer='he_normal', activation=act_map[int(stride_size)])(X)
        elif architecture[i] == 'maxpool':
            assert type(model.layers[global_index]) == layers.MaxPooling2D
            X = layers.MaxPooling2D(pool_size=int(filter_size))(X)
        elif architecture[i] == 'globalavgpool':
            assert type(model.layers[global_index]) == layers.GlobalAveragePooling2D
            X = layers.GlobalAveragePooling2D()(X)
        elif architecture[i] == 'batch':
            assert type(model.layers[global_index]) == layers.BatchNormalization
            X = layers.BatchNormalization()(X)
        elif architecture[i] == 'activation':
            assert type(model.layers[global_index]) == layers.Activation
            X = layers.Activation(act_map[int(stride_size)])(X)
        elif architecture[i] == 'flatten':
            X = layers.Flatten()(X)

    for i in range(len(dense_params) // NUM_HYPERPARAMS):
        units = int(dense_params['dense_filter_size_' + str(i + 1)])
        dropout = float(dense_params['dense_num_filters_' + str(i + 1)])
        act = int(dense_params['dense_stride_size_' + str(i + 1)])

        X = layers.Dense(units, activation=act_map[act], kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(dropout)(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
    return models.Model(inputs=model.inputs, outputs=X)


# training the original model
base_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
X = base_model.layers[-2].output
X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)
base_model = models.Model(inputs=base_model.inputs, outputs=X)

# freezing the layers of the model
for i in range(len(base_model.layers)-1):
    base_model.layers[i].trainable = False

base_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

history = base_model.fit_generator(
    train_generator,
    validation_data=valid_generator, epochs=EPOCHS,
    steps_per_epoch=len(train_generator) / batch_size,
    validation_steps=len(valid_generator), callbacks=[reduce_LR]
)

# log the results
best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
log_tuple = ('relu', 'he_normal', 0, 1, [], [], [], [], [], history.history['loss'][best_acc_index],  history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
log_df.loc[log_df.shape[0], :] = log_tuple
log_df.to_csv(RESULTS_PATH)

# tuning the model
base_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
base_model = models.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False


# optimize layers
optim_neurons = []
optim_dropouts = []
best_acc = 0
# layers not considered in optimization
meaningless = [
    layers.Activation,
    layers.GlobalAveragePooling2D,
    layers.BatchNormalization,
    layers.ZeroPadding2D,
    layers.Add,
    layers.Flatten
]

for i in range(1, len(base_model.layers) + 1):
    unfreeze = i
    if type(base_model.layers[-i]) in meaningless:
        continue
    temp_model = models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
    print(f"Tuning last {unfreeze} layers.")
    time.sleep(3)

    temp_arc = []
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
        elif type(temp_model.layers[-j]) == layers.Dense:
            temp_arc.append('dense')
        elif type(temp_model.layers[-j]) == layers.Activation:
            temp_arc.append('activation')
        elif type(temp_model.layers[-j]) == layers.Flatten:
            temp_arc.append('flatten')

    # making bounds list
    bounds = []
    for iter_ in range(len(temp_arc)):
        if temp_arc[iter_] == 'conv':
            bounds.extend(
                [
                    {'name': 'conv_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [2, 3, 5]},
                    {'name': 'conv_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [64, 128, 256, 512]},
                    {'name': 'conv_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': list(range(len(act_map)))},
                ]
            )
        elif temp_arc[iter_] == 'dense':
            bounds.extend(
                [
                    {'name': 'dense_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [2 ** j for j in range(6, 11)]},
                    {'name': 'dense_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': np.arange(0, 1, step=0.1)},
                    {'name': 'dense_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': list(range(len(act_map)))},
                ]
            )
        elif temp_arc[iter_] == 'flatten':
            bounds.extend(
                [
                    {'name': 'flatten_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'flatten_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'flatten_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                ]
            )
        elif temp_arc[iter_] == 'maxpool':
            bounds.extend(
                [
                    {'name': 'maxpool_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [2, 3]},
                    {'name': 'maxpool_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'maxpool_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                ]
            )
        elif temp_arc[iter_] == 'globalavgpool':
            bounds.extend(
                [
                    {'name': 'avgpool_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'avgpool_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'avgpool_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                ]
            )
        elif temp_arc[iter_] == 'activation':
            bounds.extend(
                [
                    {'name': 'activation_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': list(range(len(act_map)))},
                    {'name': 'activation_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': 'activation_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                ]
            )
        else:
            bounds.extend(
                [
                    {'name': temp_arc[iter_] + '_filter_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': temp_arc[iter_] + '_num_filters_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
                    {'name': temp_arc[iter_] + '_stride_size_' + str(iter_ + 1), 'type': 'discrete', 'domain': [1]},
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
        acts = []

        conv_params = OrderedDict()

        j = 0
        while j < x.shape[1]:
            conv_params[temp_arc[j // NUM_HYPERPARAMS] + '_filter_size_' + str((j // NUM_HYPERPARAMS) + 1)] = x[:, j]
            if temp_arc[j // NUM_HYPERPARAMS] not in meaningless:
                filter_sizes.append(int(x[:, j]))
            j += 1
            conv_params[temp_arc[j // NUM_HYPERPARAMS] + '_num_filters_' + str((j // NUM_HYPERPARAMS) + 1)] = x[:, j]
            if temp_arc[j // NUM_HYPERPARAMS] == 'conv':
                num_filters.append(int(x[:, j]))
            j += 1
            conv_params[temp_arc[j // NUM_HYPERPARAMS] + '_stride_size_' + str((j // NUM_HYPERPARAMS) + 1)] = x[:, j]
            if temp_arc[j // NUM_HYPERPARAMS] == 'conv' or temp_arc[j // NUM_HYPERPARAMS] == 'dense':
                acts.append(act_map[int(x[:, j])])
            j += 1

        to_train_model = get_model_conv(temp_model, -len(conv_params) // NUM_HYPERPARAMS, reverse_list(temp_arc), conv_params, optim_neurons, optim_dropouts)
        to_train_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

        # train the modified model
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=EPOCHS,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )

        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
        assert history.history['val_acc'][best_acc_index] == max(history.history['val_acc'])
        train_loss = history.history['loss'][best_acc_index]
        train_acc = history.history['acc'][best_acc_index]
        val_loss = history.history['val_loss'][best_acc_index]
        val_acc = history.history['val_acc'][best_acc_index]

        # log the results
        log_tuple = (acts, 'he_normal', unfreeze, len(optim_neurons) + 1, optim_neurons, optim_dropouts, filter_sizes, num_filters, stride_sizes, train_loss, train_acc, val_loss, val_acc)
        log_df.loc[log_df.shape[0], :] = log_tuple
        log_df.to_csv(RESULTS_PATH)

        return min(history.history['val_loss'])


    # initialize the optimizer
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
