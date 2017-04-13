import numpy as np
from keras.layers import Dense, Reshape
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

batch_size = 2
in_sequence_len = 3
out_sequence_len = 3
init_res_gen = 512

def generator_32x32():
    network = Sequential()
    network.add(Conv3D(16, kernel_size=3, activation='relu', padding='same', input_shape=(3, 64, 64, 3)))
    network.add(Conv3D(32, kernel_size=3, activation='relu', padding='same'))
    network.add(Conv3D(64, kernel_size=3, activation='relu', padding='same'))
    network.add(Conv3D(32, kernel_size=3, activation='relu', padding='same'))
    network.add(Conv3D(16, kernel_size=3, activation='relu', padding='same'))
    network.add(Conv3D(3, kernel_size=3, activation='tanh', padding='same'))

    input = np.random.rand(batch_size, 3, 64, 64, 3)
    output = network.predict(input, batch_size=batch_size, verbose=0)
    print np.shape(output)
#
generator_32x32()
