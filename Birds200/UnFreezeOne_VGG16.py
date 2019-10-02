import os
import math
import numpy as np
import GPyOpt
import keras
from keras.preprocessing import image
from keras import layers, models, optimizers, callbacks
from keras.applications import VGG16

DATA_FOLDER = "/home/deb/Sravan/CalTech101"
TRAIN_PATH = os.path.join(DATA_FOLDER, "training") # Path for training data
VALID_PATH = os.path.join(DATA_FOLDER, "validation") # Path for validation data
NUMBER_OF_CLASSES = len(os.listdir(TRAIN_PATH)) # Number of classes of the dataset
RESULTS_PATH = os.path.join("AutoFC_ResNet", "AutoFC_ResNet_log_" + DATA_FOLDER + "_bayes_opt_v5.csv") # The path to the results file

# Creating generators from training and validation data
batch_size=8 # the mini-batch size to use for the dataset
datagen = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) # creating an instance of the data generator
train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for training data
valid_generator = datagen.flow_from_directory(VALID_PATH, target_size=(224, 224), batch_size=batch_size) # creating the generator for validation data

# creating callbacks for the model
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-10)

try:
    log_df = pd.read_csv(RESULTS_PATH, header=0, index_col=['index'])
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['index', 'activation', 'weight_initializer', 'filter_sizes', 'num_filters', 'stride_sizes', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    log_df = log_df.set_index('index')

# create an instance of the VGG16
base_model = VGG16(include_top=False)
# freeze all layers and count conv_layers
i = 0
for i in range(len(base_model.layers)):
    if type(base_model.layers[i]) == layers.Conv2D:
        i += 1
    base_model.layers[i].trainable = False


def get_model(model, **kwargs):
    X = model
    for i in range(len(kwargs) // 3):
        # check dimension and upsample if necessary
        if X.shape[1] < 5:
            upsampling_factor = math.ceil(5 / X.shape[1])
            X = layers.UpSampling2D(size=(upsampling_factor, upsampling_factor))(X)

        conv_params = filter(lambda x: x[0].endswith(str(i + 1)), kwargs.items())
        filter_size, num_filters, stride_size = map(lambda x: x[1], conv_params)
        X = layers.Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), strides=(stride_size, stride_size), activation='tanh')(X)

    return models.Model(inputs=model, outputs=X)

# initialise bounds list
bounds = []

best_acc = 0 # initialise best accuracy
# Loop over number of layers
for iter_ in range(1, len(base_model.layers) + 1):
    if type(base_model.layers[-iter_]) == layers.Conv2D:
        temp_model = base_model.layers[-iter_].output
    else:
        continue

    bounds.extend(
        [
            {'name': 'conv_filter_size_' + str(iter_), 'type': 'discrete', 'domain': [1, 3]},
            {'name': 'conv_num_filters_' + str(iter_), 'type': 'discrete', 'domain': [32, 64, 128,512]},
            {'name': 'conv_stride_size_' + str(iter_), 'type': 'discrete', 'domain': [1, 2]}
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
        hyper_params = {}
        i = 0
        while i < len(bounds):
            hyper_params['conv_filter_size_' + str((i // 3) + 1)] = x[:, (i // 3) + (i % 3)]
            filter_sizes.append(x[:, (i // 3) + (i % 3)])
            i += 1
            hyper_params['conv_num_filters_' + str((i // 3) + 1)] = x[:, (i // 3) + (i % 3)]
            num_filters.append(x[:, (i // 3) + (i % 3)])
            i += 1
            hyper_params['conv_stride_size_' + str((i // 3) + 1)] = x[:, (i // 3) + (i % 3)]
            i += 1
            stride_sizes.append(x[:, (i // 3) + (i % 3)])

        to_train_model = get_model(temp_model, **hyper_params)
        history = to_train_model.fit_generator(
            train_generator,
            validation_data=valid_generator, epochs=20,
            steps_per_epoch=len(train_generator) / batch_size,
            validation_steps=len(valid_generator), callbacks=[reduce_LR]
        )

        return min(history.history['val_loss']) # return value of the function

    opt_ = GPyOpt.methods.BayesianOptimization(f=model_fit, domain=bounds)
    opt_.run_optimization(max_iter=10)

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
    log_tuple = ('tanh', 'glorot_uniform', filter_sizes, num_filters, stride_sizes, history.history['loss'][best_acc_index], history.history['acc'][best_acc_index], history.history['val_loss'][best_acc_index], history.history['val_acc'][best_acc_index])
    log_df.loc[log_df.shape[0]] = log_tuple # add the record to the dataframe
    log_df.to_csv(RESULTS_PATH) # save the dataframe in a CSV file
