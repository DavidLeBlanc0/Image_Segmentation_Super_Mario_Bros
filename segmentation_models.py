import numpy as np

from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def create_large_model(inputDims, outputDims):
    learningRate = 1e-4
    denseUnits = 128
    dropoutVal = 0.0
    usingPooling = False
    model = Sequential()
    model.add(Input(shape = inputDims))

    #TODO: rework conv layers
    model.add(Conv2D(
        filters = 32, strides = 4, kernel_size = 4,
        input_shape = (*(inputDims),), activation = 'relu'
        #, data_format = 'channels_first'
    ))
    if usingPooling:
        model.add(MaxPooling2D())

    model.add(Conv2D(
        filters = 64, strides = 2, kernel_size = 2,
        activation = 'relu'
    ))
    if usingPooling:
        model.add(MaxPooling2D())

    model.add(Conv2D(
        filters = 64, strides = 1, kernel_size = 1,
        activation = 'relu'
    ))
    if usingPooling:
        model.add(MaxPooling2D())


    # Flatten layer output
    model.add(Flatten())
    model.add(Dense(units = denseUnits, activation = 'relu'))
    model.add(Dropout(dropoutVal))

    model.add(Dense(units = np.prod(outputDims)))
    model.add(Reshape(outputDims))
    
    opti = Adam(learning_rate = learningRate)
    model.compile(loss = "mse", optimizer=opti, metrics=["mse"])

    return model

def create_small_model(inputDims, outputDims):
        learningRate = 1e-4
        denseUnits = 512
        dropoutVal = 0.1
        usingPooling = True
        model = Sequential()
        model.add(Input(shape = inputDims))

        #TODO: rework conv layers
        model.add(Conv2D(
            filters = 64, strides = 4, kernel_size = 16,
            input_shape = (*(inputDims),), activation = 'relu'
            #, data_format = 'channels_first'
        ))
        if usingPooling:
            model.add(MaxPooling2D())

        model.add(Conv2D(
            filters = 64, strides = 2, kernel_size = 8,
            activation = 'relu'
        ))
        if usingPooling:
            model.add(MaxPooling2D())


        # Flatten layer output
        model.add(Flatten())
        model.add(Dense(units = denseUnits, activation = 'relu'))
        model.add(Dropout(dropoutVal))

        model.add(Dense(units = np.prod(outputDims)))
        model.add(Reshape(outputDims))\
        
        opti = Adam(learning_rate = learningRate)
        model.compile(loss = "mse", optimizer=opti, metrics=["mse"])

        return model
