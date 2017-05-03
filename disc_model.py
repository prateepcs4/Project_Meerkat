import tensorflow as tf
from tfutils import w, b

class Discriminator_model:

    def __init__(self, seqlen, feature_maps, kernel_sizes, fc_layer_sizes):

        self.seqlen = seqlen
        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes
        self.fc_layer_sizes = fc_layer_sizes

    def define_graph(self):
        # Sets up the model graph

        # The variables to train
        self.train_vars = []

        # Sets up the layer
        with tf.name_scope('net'):
            with tf.name_scope('setup'):
                # Convolution
                with tf.name_scope('convolutions'):
                    self.conv_ws = []
                    self.conv_bs = []

                    for i in xrange(len(self.kernel_sizes)):
                        self.conv_ws.append(w([self.seqlen, self.kernel_sizes[i], self.kernel_sizes[i],
                                               self.feature_maps[i],
                                               self.feature_maps[i+1]]))
                        self.conv_bs.append(b([self.feature_maps[i+1]]))
