from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.activations import relu, tanh
import numpy as np

batch_size = 1
in_sequence_len = 3
out_sequence_len = 3
init_res_gen = 512

def generator():
    network = Sequential()
    network.add(Dense(1024, input_dim=200, activation='relu'))
    network.add(Dense(in_sequence_len*4*4*init_res_gen, activation='relu'))
    network.add(Reshape((in_sequence_len,4,4,init_res_gen)))
    network.add(UpSampling3D((1,2,2)))
    network.add(Conv3D(init_res_gen/2, kernel_size=3, activation='relu', padding='same'))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    network.add(UpSampling3D((1, 2, 2)))
    network.add(Conv3D(init_res_gen/4, kernel_size=3, activation='relu', padding='same'))
    network.add(BatchNormalization())
    network.add(Activation('relu'))
    network.add(UpSampling3D((1, 2, 2)))
    # network.add(Conv3D(init_res_gen / 8, kernel_size=3, activation='relu', padding='same'))
    # network.add(BatchNormalization())
    # network.add(Activation('relu'))
    # network.add(UpSampling3D((in_sequence_len, 64, 64)))
    # network.add(Conv3D(1, kernel_size=3, activation='relu', padding='same'))
    # network.add(BatchNormalization())
    # network.add(Activation('tanh'))

    input = np.random.rand(batch_size, 200)
    output = network.predict(input, batch_size=batch_size, verbose=0)
    print np.shape(output)

generator()