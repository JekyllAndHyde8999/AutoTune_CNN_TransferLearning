import os
import numpy as np
import GPyOpt
import keras
from keras.preprocessing import image
from keras import layers, models
from keras.applications import ResNet50
DATA_FOLDER = "/home/deb/Sravan/CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
RESULTS_PATH = os.path.join("AutoFC_ResNet", "AutoFC_ResNet_log_" + DATA_FOLDER + "_bayes_opt_v5.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

NUMBER_OF_FROZEN_LAYERS = 151 # can change later
#MODEL_FILE_NAME = None
base_model = ResNet50(weights="imagenet")
#base_model = models.load_model(MODEL_FILE_NAME)

for i in range(NUMBER_OF_FROZEN_LAYERS):
    base_model.layers[i].trainable = False

print(f'Froze {NUMBER_OF_FROZEN_LAYERS} layers in the model.')

bounds = [
    {'name': 'conv1_filter_size', 'type': 'discrete', 'domain': [1, 3, 5]},
    {'name': 'conv1_num_filters', 'type': 'discrete', 'domain': [3, 5]},
    {'name': 'conv1_stride_size', 'type': 'discrete', 'domain': [1, 2, 3]},
    {'name': 'conv2_filter_size', 'type': 'discrete', 'domain': [1, 3, 5]},
    {'name': 'conv2_stride_size', 'type': 'discrete', 'domain': [1, 2, 3]},
    {'name': 'conv2_num_filters', 'type': 'discrete', 'domain': [3, 5]},
    {'name': 'max_pooling_filter_size', 'type': 'discrete', 'domain': [2, 3]},
    {'name': 'number_occurrences', 'type': 'discrete', 'domain': range(3, 6)}
]


def get_model(conv1_filter_size, conv1_num_filters, conv1_stride_size, conv2_filter_size, conv2_num_filters, conv2_stride_size, max_pooling_filter_size, number_occurrences):
    X = base_model.layers[NUMBER_OF_FROZEN_LAYERS - 1].output
    X = layers.ZeroPadding2D(padding=(conv1_filter_size, conv1_filter_size))(X)

    for _ in range(number_occurrences):
        X = layers.Conv2D(filters=conv1_num_filters, kernel_size=(conv1_filter_size, conv1_filter_size), strides=(conv1_stride_size, conv1_stride_size), activation='relu')(X)
        #X = layers.SeparableConv2D(filters=conv2_num_filters, filter_size=(conv2_filter_size, conv2_filter_size), strides=(conv2_stride_size, conv2_stride_size), activation='relu')(X)
        X = layers.MaxPooling2D(pool_size=max_pooling_filter_size)(X)

    X = layers.Flatten()(X)
    X = layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(X)

    model = models.Model(inputs=base_model.inputs, outputs=X)
    return model


def model_fit(x):
    conv1_filter_size = int(x[:, 0])
    conv1_num_filters = int(x[:, 1])
    conv1_stride_size = int(x[:, 2])
    conv2_filter_size = int(x[:, 3])
    conv2_num_filters = int(x[:, 4])
    conv2_stride_size = int(x[:, 5])
    max_pooling_filter_size = int(x[:, 6])
    activation = 'relu'
    number_occurrences = int(x[:, 7])

    temp_model = get_model(
        conv1_filter_size=conv1_filter_size,
        conv1_num_filters=conv1_num_filters,
        conv1_stride_size=conv1_stride_size,
        conv2_filter_size=conv2_filter_size,
        conv2_num_filters=conv2_num_filters,
        conv2_stride_size=conv2_stride_size,
        max_pooling_filter_size=max_pooling_filter_size,
        number_occurrences=number_occurrences
    )

    temp_model.compile(optimizer='adam', loss='categorical_crossentropy')
    history = temp_model.fit_generator(
        train_generator,
        valid_generator=valid_generator, epochs=20,
        steps_per_epoch=len(train_generator) / batch_size,
        validation_steps=len(valid_generator)
    )

    return min(history.history['val_loss']) # return value of the function


opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit, domain=bounds) # make an instance of the BayesianOptimization class with the function to optimize
opt_.run_optimization(max_iter=10) # run the optimization

print("Optimized Parameters:")
for i in range(len(bounds)):
    print(f"\t{bounds[i]['name']}: {opt_.x_opt[i]}")
print(f"Optimized Function value: {opt_.fx_opt}")
