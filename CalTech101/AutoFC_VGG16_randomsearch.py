import os
import numpy
import random
import keras
import time

import pandas as pd

from itertools import product
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image
from keras.applications import *
from keras import models, layers

# Declaring global variables
DATA_FOLDER = "Caltech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
RESULTS_PATH = os.path.join("AutoFC_VGG16", "AutoFC_VGG16_log_" + DATA_FOLDER + "_random_search_v1.csv") # The path to the results file

# Creating generators from training and validation data
batch_size = 8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data


# Creating callbacks to use in the model
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=numpy.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-10)


# Creating a pandas dataframe to log the results if the results file does not exist already.
try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["index", "num_layers", "activation", "neurons", "dropout", "weight_initializer", "time", "train_loss", "train_acc", "val_loss", "val_acc"])
    log_df = log_df.set_index('index')


# Creating a hyperparameter grid space to search over
param_grid = {
    'activation': ['relu', 'tanh', 'sigmoid'], # the possible activation functions that can be used in each layer
    'neurons': (2  ** j for j in range(6, 11)), # the possible number of nuerons that can be used in each layer
    'dropout': numpy.arange(0, 0.6, 0.1), # the possible dropout values that can be used in each layer
    'weight_initializer': ['he_normal'], # the possible weight initializer that can be used in each layer
    'num_layers': range(0, 3) # the possible number of layers that can be used in the model
}

# removing the num_layers key from the param_grid dictionary
num_layers = param_grid['num_layers']
inner_grid = {key: param_grid[key] for key in param_grid.keys() if key != 'num_layers'}

inner_hyper = list(product(*inner_grid.values())) # get combinations of the remaining hyperparameters

# iterate over the number of layers
for i in num_layers:
    temp_store = [] # store the combinations to use in this iteration
    NUMBER_OF_SAMPLES = 1 if i == 0 else 33 # the number of configurations to be chosen randomly based on the number of layers

    for z in range(NUMBER_OF_SAMPLES): # randomly sample combinations
        use_now = random.sample(inner_hyper, i) # sample 'i' (the number of layers) configurations [one for each layer]
        while use_now in temp_store: # ensure the sample configurations are not repeated
            use_now = random.sample(inner_hyper, i)

        temp_store.append(use_now) # store to be used in the experiment

    for j in temp_store: # iterate over the list of sampled configurations
        # j: list of 'i' configurations [one for each layer]
        """a list for each type of hyperparameter to log results later on"""
        act_list = [] # for activations
        neu_list = [] # for number of neurons
        drop_list = [] # for the dropout
        weight_list = [] # for weight initializer

        base_model = VGG16(weights="imagenet") # make an instance of an existing architecture to start the experiment
        for layer in base_model.layers: # freeze the existing layers to make them non-trainable
            layer.trainable = False

        X = base_model.layers[-4].output # removing the last fully connected layers

        for k in j: # iterate over each layer's configuration
            activation = k[0]
            neurons = k[1]
            dropout = k[2]
            weight_init = k[3]

            # append the hyperparameter values to the corresponding lists for logging results later on
            act_list.append(activation)
            neu_list.append(neurons)
            drop_list.append(dropout)
            weight_list.append(weight_init)

            # adding the fully connected layer with the current configuration
            X = layers.Dense(neurons, activation=activation, kernel_initializer=weight_init)(X)
            X = layers.Dropout(dropout)(X)
            X = layers.BatchNormalization()(X)
        X = layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(X) # adding the final layer for classification

        new_model = models.Model(inputs=base_model.input, outputs=X) # making a keras Model instance for training
        new_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=["accuracy"]) # compiling the model with the specified loss function and optimizer
        start = time.time()
        history = new_model.fit_generator(train_generator,
            validation_data=valid_generator, epochs=20, callbacks=[lr_reducer],
            steps_per_epoch=len(train_generator)/batch_size,
            validation_steps =len(valid_generator)
        )
        time_taken = time.time() - start # calculating the time taken to train the model with the current configuration

        # log the results in the log dataframe
        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc'])) # get the epoch number for the best validation accuracy

        # create the log record for the current experiment
        log_tuple = (i, act_list, neu_list, drop_list, weight_list, time_taken, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
        log_df.loc[log_df.shape[0]] = log_tuple # add the record to the dataframe
        log_df.to_csv(RESULTS_PATH) # save the dataframe in a CSV file
