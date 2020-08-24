from keras import models, layers

def AlexNet(input_shape=(224, 224, 3)):
    X_in = layers.Input(shape=input_shape)

    # 1st conv layer
    X = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(X_in)
    X = layers.MaxPooling2D()(X)

    # 2nd conv layer
    X = layers.Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), activation='relu')(X)
    X = layers.MaxPooling2D()(X)

    # 3rd conv layer
    X = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)

    # 4th conv layer
    X = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)

    # 5th conv layer
    X = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)
    X = layers.MaxPooling2D()(X)

    # Flatten layer
    X = layers.Flatten()(X)

    # FC1
    X = layers.Dense(4096, activation='relu')(X)

    # FC2
    X = layers.Dense(4096, activation='relu')(X)

    # Output layer
    X = layers.Dense(1000, activation='softmax')(X)

    # build the model
    return models.Model(inputs=X_in, outputs=X)


if __name__ == '__main__':
    AlexNet()
