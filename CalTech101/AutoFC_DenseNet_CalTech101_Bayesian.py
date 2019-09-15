import os
import numpy
import keras
import GPyOpt, GPy

import pandas as pd

from keras.preprocessing import image
from keras.applications import *
from keras import models, layers
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime
from itertools import product

# Declaring global variables
DATA_FOLDER = "Caltech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
RESULTS_PATH = os.path.join("AutoFC_DenseNet", "AutoFC_DenseNet_log_CalTech_101_bayes_opt_v5.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data


# function to build a model and return it
def get_model(num_layers, num_neurons, dropout, activation, weight_initializer):
    base_model = DenseNet121(weights="imagenet") # make an instance of an existing architecture
    for layer in base_model.layers: # freeze the existing layers to make them non-trainable
        layer.trainable = False

    X = base_model.layers[-2].output # removing the last fully connected layers

    # add the required fully connected layers one-by-one
    for i in range(num_layers):
        X = layers.Dense(num_neurons[i], activation=activation, kernel_initializer=weight_initializer)(X)
        X = layers.Dropout(dropout[i])(X)
        X = layers.BatchNormalization()(X)

    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(X) # the final layer for classification
    model = models.Model(inputs=base_model.inputs, outputs=X) # making a keras Model instance for training
    return model # returing the model


# Creating callbacks to use in the model
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-10)

# Creating a pandas dataframe to log the results if the results file does not exist already.
try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'dropout', 'num_neurons', 'num_layers', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')


# the grid space of hyperparameters to search over
p_space = {
    #'activation': ['relu', 'tanh', 'sigmoid'], # the possible activation functions that can be used in each layer
    'weight_initializer': ['he_normal'], # the possible weight initializer that can be used in each layer
    'num_layers': list(range(0,3)) # the possible number of layers that can be used in the model
}

activation_mapping = {
    0: 'relu',
    1: 'tanh',
    2: 'sigmoid'
}

p_space = list(product(*p_space.values())) # get combinations of the hyperparameters in the search space
start = datetime.time(datetime.now())

# iterate over each combination
for combo in p_space:
    weight_initializer, num_layers = combo
    bounds = [] # initialize a list bounds for search space for Bayesian Optimization
    for i in range(num_layers): # add dropout ranges based on number of layers
        bounds.append({'name': 'dropout' + str(i + 1), 'type': 'discrete', 'domain': numpy.arange(0, 0.6, 0.1)})
    for i in range(num_layers): # add number of neurons ranges based on number of layers
        bounds.append({'name': 'num_neurons' + str(i + 1), 'type': 'discrete', 'domain': [2 ** j for j in range(6, 11)]})

    bounds.append({'name': 'activation', 'type': 'discrete', 'domain': [0, 1, 2]})

    history = None
    neurons = None
    dropouts = None

    # model_fit function: funtion to be optimized using bayesian optimization
    def model_fit(x):
        global neurons # make neurons accessible outside the function
        global dropouts # make dropouts accessible outside the function
        dropouts = [float(x[:, i]) for i in range(0, num_layers)] # get the values of dropouts currently selected
        neurons = [int(x[:, i]) for i in range(num_layers, len(bounds))] # get the values of num_neurons currently selected
        activation_fn = activation_mapping[int(x[:, -1])] # get the activation function

        model = get_model(
            dropout=dropouts,
            num_layers=num_layers,
            num_neurons= neurons,
            activation=activation_fn,
            weight_initializer=weight_initializer
        ) # get the model with the current set of hyperparameters

        model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy']) # compiling the model with the specified loss function and optimizer

        global history # make history accessible outside the function

        history = model.fit_generator(train_generator,
            validation_data=valid_generator, epochs=20, callbacks=[lr_reducer],
            steps_per_epoch=len(train_generator)/batch_size,
            validation_steps =len(valid_generator)
        )

        best_acc_index = history.history['val_acc'].index(max(history.history['val_acc'])) # get the epoch number for the best validation accuracy

        # create the log record for the current experiment
        log_tuple = (activation, weight_initializer, dropouts, neurons, num_layers, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
        log_df.loc[log_df.shape[0]] = log_tuple # add the record to the dataframe
        return min(history.history['val_loss']) # return value of the function

    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit, domain=bounds) # make an instance of the BayesianOptimization class with the function to optimize
    opt_.run_optimization(max_iter=5) # run the optimization
    log_df.to_csv(RESULTS_PATH) # save the dataframe in a CSV file

end = datetime.time(datetime.now())
