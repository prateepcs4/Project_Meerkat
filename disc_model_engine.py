import tensorflow as tf
from tfutils import w, b, conv_out_size

class Discriminator_Model_Engine:

    def __init__(self, height, width, seqlen, feature_maps, kernel_sizes, fc_layer_sizes):

        self.height = height
        self.width = width
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
                    last_out_height = self.height
                    last_out_width = self.width
                    # last_out_seqlen = self.seqlen

                    for i in xrange(len(self.kernel_sizes)):
                        self.conv_ws.append(w([self.seqlen, self.kernel_sizes[i], self.kernel_sizes[i],
                                               self.feature_maps[i],
                                               self.feature_maps[i+1]]))
                        self.conv_bs.append(b([self.feature_maps[i+1]]))
                        last_out_height = conv_out_size(last_out_height, 'SAME', self.kernel_sizes[i], 1)
                        last_out_width = conv_out_size(last_out_width, 'SAME', self.kernel_sizes[i], 1)

                with tf.name_scope('fully-connected'):
                    # Add in an initial layer to go from the last conv to the first fully-connected.
                    # Use /2 for the height and width because there is a 2x2 pooling layer
                    self.fc_layer_sizes.insert(0, self.seqlen * (last_out_height / 2) * (last_out_width / 2) *
                                               self.feature_maps[-1])

                    self.fc_ws = []
                    self.fc_bs = []
                    for i in xrange(len(self.fc_layer_sizes) - 1):
                        self.fc_ws.append(w([self.fc_layer_sizes[i], self.fc_layer_sizes[i+1]]))
                        self.fc_bs.append(b([self.fc_layer_sizes[i+1]]))

                self.train_vars += self.conv_ws
                self.train_vars += self.conv_bs
                self.train_vars += self.fc_ws
                self.train_vars += self.fc_bs

