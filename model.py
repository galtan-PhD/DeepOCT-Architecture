import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import constants as CONST

def DeepOCT():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size = (4, 4), activation='relu', input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, kernel_size=(12,12), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(96, kernel_size=(13,13), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(140, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(470, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))

    opt = tf.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
    print('DeepOCT model prepared...')
    return model