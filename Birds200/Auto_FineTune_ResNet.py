import keras
import numpy as np
from keras import applications
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import os
from keras.utils import multi_gpu_model
import numpy as np
from keras.optimizers import SGD 
import matplotlib.pyplot as plt
import h5py
import csv
img_width, img_height = 224, 224
TRAIN_PATH = os.path.join("/home/deb/Sravan/corel-1k/", "training")
VALID_PATH = os.path.join("/home/deb/Sravan/corel-1k/", "validation")
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(img_width, img_height), batch_size=1)
valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(VALID_PATH, target_size=(img_width, img_height), batch_size=1)
#model = applications.VGG16(weights = "imagenet", include_top=True, input_shape = (img_width, img_height, 3))
model = ResNet50(weights = "imagenet", include_top=True, input_shape = (img_width, img_height, 3))
#model = applications.
#print(model.summary())
x = model.layers[-2].output
#x = Flatten()(x)
predictions = Dense(10, activation="softmax")(x)
# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_final = multi_gpu_model(model_final,gpus=2)
model_final.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
model_final.summary()
#for layer in model.layers:
#    print(layer.trainable)

#for i in range(1,len(model.layers)):
#    for layer in model.layers[:i]:
#        print(layer.trainable)
#    print(i,'\n**********')
f = open("brute_force_log.csv",mode = 'w')  
layer_acc_lst = []
max_acc = 0
prev_max = 0
flag = 1
num_layers = []
for i in reversed(range(1,len(model.layers))):
    num_layers.append(i)
    f.write('layers frozen,epoch,train_acc,val_acc\n')
    model = applications.ResNet50(weights = "imagenet", include_top=True, input_shape = (img_width, img_height, 3))
    for layer in model.layers[:i]:#freeze initial i layers
        layer.trainable = False
    x = model.layers[-2].output
    predictions = Dense(10, activation="softmax")(x)
    model_final = Model(inputs = model.input, outputs = predictions)
    model_final.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(lr = 1e-3), metrics=["accuracy"])
    if flag == 1:
        print(model_final.summary())
        flag = 0
    print(str(i),' layers freezed')
    train_acc_lst = []
    val_acc_lst = []
    history = model_final.fit_generator(generator=train_generator,epochs=2,validation_data = valid_generator,steps_per_epoch = 200, validation_steps = 50,verbose=2)
    best_acc_index = history.history['val_acc'].index(max(history.history['val_acc']))
    print('best epoch',best_acc_index+1)
    train_acc = history.history['acc'][best_acc_index]
    train_acc_lst.append(train_acc)
    val_acc = history.history['val_acc'][best_acc_index]
    val_acc_lst.append(val_acc)
    best_acc_index = best_acc_index + 1
    f.write(str(i)+','+ str(best_acc_index) +','+ str(train_acc) + ',' + str(val_acc) + '\n')
    if val_acc > max_acc:
        max_acc = val_acc
       	fname = 'Auto_fine_tune_sequential.h5'
       	model_final.save(fname)
       	layer_acc_lst.append([train_acc_lst,val_acc_lst])
f.close()
max_train_lst = []
max_val_lst = []
for i in layer_acc_lst:
    max_val = max(i[1])
    max_val_ind = i[1].index(max_val)
    max_train = i[0][max_val_ind]
    max_train_lst.append(max_train)
    max_val_lst.append(max_val)
print(max_train_lst)
print(max_val_lst)
plt.plot(num_layers,max_train_lst,'r',label = 'train_accuracy')
plt.plot(num_layers,max_val_lst,'b',label = 'validation accuracy')
plt.xlabel('number of layers frozen')
plt.ylabel('accuracy')
plt.legend(loc = 'best')
plt.show()
