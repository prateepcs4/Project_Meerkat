from keras.layers.convolutional import Conv3D
from keras.models import Sequential
from keras.layers import Dense


batch_size = 10
in_sequence_len = 3
out_sequence_len = 3

def generator():
    network = Sequential()
    