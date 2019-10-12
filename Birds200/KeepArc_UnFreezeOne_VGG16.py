import os
import math
import numpy as np
import pandas as pd
import GPyOpt
import keras
from collections import OrderedDict
from keras.preprocessing import image
from keras import layers, models, optimizers, callbacks
from keras.applications import VGG16

DATA_FOLDER = "/home/shabbeer/Sravan/CalTech101"
# DATA_FOLDER = "CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
RESULTS_PATH = os.path.join("AutoConv_VGG16", "AutoConv_VGG16_log_" + DATA_FOLDER.split('/')[-1] + "autoconv_bayes_opt_v5.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.5e-10)

# adagrad optimizer
ADAM = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)

try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'num_layers_tuned', 'activation', 'weight_initializer', 'filter_sizes', 'num_filters', 'stride_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')


# create an instance of the VGG16
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
arc = []
for layer in base_model.layers:
    if type(layer) == layers.Conv2D:
        arc.append('conv')
    if type(layer) == layers.MaxPooling2D:
        arc.append('maxpool')

print(arc)
base_model.summary()

print("Freezing the layers")
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False


def get_model_fake(model, index, **kwargs):
    X = model.layers[index].output
    for i in range(len(kwargs) // 3):
        # check dimension and upsample if necessary
        if X.shape[1] < 5:
            print(f"upsampling: {X.shape}")
            upsampling_factor = math.ceil(5 / X.shape[1])
            X = layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))(X)

        conv_params = filter(lambda x: x[0].endswith(str(i + 1)), kwargs.items())
        filter_size, num_filters, stride_size = map(lambda x: x[1], conv_params)
        print('filter_size', filter_size)
        print('num_filters', num_filters)
        print('stride_size', stride_size)
        X = layers.Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), strides=(stride_size, stride_size), activation='relu', kernel_initializer='he_normal')(X)

    X = layers.Flatten()(X)
    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)

    return models.Model(inputs=model.inputs, outputs=X)


def get_model_2(model, index, architecture, **kwargs):
    X = model.layers[index].output
    for i in range(len(kwargs) // 3):
        params_dicts = filter(lambda x: x[0].startswith(architecture[index + i + 1]) and x[0].endswith(str((i + 1))), kwargs.items())
        filter_size, num_filters, stride_size = map(lambda x: x[1], params_dicts)
        # print(f'{architecture[i]} layer: {filter_size}, {num_filters}, {stride_size}')

        if architecture[index + i + 1] == 'conv':
            X = layers.Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), strides=(stride_size, stride_size))(X)
        if architecture[index + i + 1] == 'maxpool':
            X = layers.MaxPooling2D(pool_size=filter_size)(X)

    X = layers.Flatten()(X)
    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)

    return models.Model(inputs=model.inputs, outputs=X)


def get_model(model, index, architecture, **kwargs):
    #X = model.layers[index].output
    for i in range(len(kwargs) // 3):
        params_dicts = filter(lambda x: x[0].startswith(architecture[index + i + 1]) and x[0].endswith(str((i + 1))), kwargs.items())
        filter_size, num_filters, stride_size = map(lambda x: x[1], params_dicts)
        # print(f'{architecture[i]} layer: {filter_size}, {num_filters}, {stride_size}')

        if architecture[index + i + 1] == 'conv':
            #X = layers.Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), strides=(stride_size, stride_size))(X)
            model.layers[index + i + 1].filters = num_filters
            model.layers[index + i + 1].kernel_size = (filter_size, filter_size)
            model.layers[index + i + 1].strides = (stride_size, stride_size)
        if architecture[index + i + 1] == 'maxpool':
            #X = layers.MaxPooling2D(pool_size=filter_size)(X)
            model.layers[index + i + 1].pool_size = filter_size

    new_model = models.model_from_json(model.to_json())
    X = new_model.layers[-1].output
    X = layers.Flatten()(X)
    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax', kernel_initializer='he_normal')(X)

    return models.Model(inputs=new_model.inputs, outputs=X)


best_acc = 0 # initialize best accuracy

# do for original model
to_train_model = get_model(base_model, -1, arc)
to_train_model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
to_train_model.summary()
history = to_train_model.fit_generator(
    train_generator,
    validation_data=valid_generator, epochs=40,
    steps_per_epoch=len(train_generator) / batch_size,
    validation_steps=len(valid_generator), callbacks=[reduce_LR]
)

best_acc_index = history.history['val_acc'].index(max(history.history['val_acc'])) # get the epoch number for the best validation accuracy
log_tuple = (0, 'relu', 'he_normal', [], [], [], history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
log_df.loc[log_df.shape[0]] = log_tuple # add the record to the dataframe
log_df.to_csv(RESULTS_PATH) # save the dataframe in a CSV file
best_acc = max(best_acc, history.history['val_acc'][-1])

bounds = []
for iter_ in range(1, len(base_model.layers) + 1):
    if type(base_model.layers[-iter_]) == layers.Conv2D:
        bounds.extend(
            [
                {'name': 'conv_filter_size_' + str(iter_), 'type': 'discrete', 'domain': [1, 3]},
                {'name': 'conv_num_filters_' + str(iter_), 'type': 'discrete', 'domain': [32, 64, 128, 256]},
                {'name': 'conv_stride_size_' + str(iter_), 'type': 'discrete', 'domain': [1]}
            ]
        )

    if type(base_model.layers[-iter_]) == layers.MaxPooling2D:
        bounds.extend(
            [
                {'name': 'maxpool_filter_size_' + str(iter_), 'type': 'discrete', 'domain': [1, 3]},
                {'name': 'maxpool_num_filters_' + str(iter_), 'type': 'discrete', 'domain': [1]},
                {'name': 'maxpool_stride_size_' + str(iter_), 'type': 'discrete', 'domain': [1]}
            ]
        )

    filter_sizes = []
    num_filters = []
    stride_sizes = []
    history = None


    def model_fit(x):
        global filter_sizes
        global num_filters
        global stride_sizes
        global history

        # get params
        hyper_params = OrderedDict()
        i = 0
        while i < len(bounds):
            hyper_params[arc[i // 3] + '_filter_size_' + str((i // 3) + 1)] = int(x[:, (i // 3) + (i % 3)])
            filter_sizes.append(int(x[:, (i // 3) + (i % 3)]))
            i += 1
            hyper_params[arc[i // 3] + '_num_filters_' + str((i // 3) + 1)] = int(x[:, (i // 3) + (i % 3)])
            num_filters.append(int(x[:, (i // 3) + (i % 3)]))
            i += 1
            hyper_params[arc[i // 3] + '_stride_size_' + str((i // 3) + 1)] = int(x[:, (i // 3) + (i % 3)])
            stride_sizes.append(int(x[:, (i // 3) + (i % 3)]))
            i += 1

        to_train_model = get_model(base_model, -iter_, arc, **hyper_params)
        to_train_model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
        to_train_model.summary()
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=40,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )

        return min(history.history['val_loss']) # return value of the function

    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit, domain=bounds)
    opt_.run_optimization(max_iter=1) # 20

    if iter_ % 2 == 0:
        import sys
        sys.exit(-1)

    new_acc = history.history['val_acc'][-1]
    if new_acc < best_acc:
        print(f"Validation Accuracy({new_acc}) didn\'t improve.")
        print(f"Breaking out at {iter_} layers.")
        break
    else:
        best_acc = max(best_acc, new_acc)

    print("Optimized Parameters:")
    for i in range(len(bounds)):
        print(f"\t{bounds[i]['name']}: {opt_.x_opt[i]}")
    print(f"Optimized Function value: {opt_.fx_opt}")

    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc'])) # get the epoch number for the best validation accuracy
    log_tuple = (iter_, 'relu', 'he_normal', filter_sizes, num_filters, stride_sizes, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
    log_df.loc[log_df.shape[0]] = log_tuple # add the record to the dataframe
    log_df.to_csv(RESULTS_PATH) # save the dataframe in a CSV file
