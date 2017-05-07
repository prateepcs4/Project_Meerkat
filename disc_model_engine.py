import tensorflow as tf
from tfutils import w, b, conv_out_size, leaky_relu
import constants as c

class Discriminator_Model_Engine:

    def __init__(self, height, width, feature_maps, kernel_sizes, fc_layer_sizes):

        self.height = height
        self.width = width
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
                        self.conv_ws.append(w([self.kernel_sizes[i], self.kernel_sizes[i],
                                               self.feature_maps[i],
                                               self.feature_maps[i+1]]))
                        self.conv_bs.append(b([self.feature_maps[i+1]]))
                        last_out_height = conv_out_size(last_out_height, 'SAME', self.kernel_sizes[i], 1)
                        last_out_width = conv_out_size(last_out_width, 'SAME', self.kernel_sizes[i], 1)

                with tf.name_scope('fully-connected'):
                    # Add in an initial layer to go from the last conv to the first fully-connected.
                    # Use /2 for the height and width because there is a 2x2 pooling layer
                    self.fc_layer_sizes.insert(0, (last_out_height / 2) * (last_out_width / 2) *
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

    def generate_predictions(self, input_frames):
        # Runs input_frames through the network to generate a prediction from 0
        # (generated img) to 1 (real img).
        input_shape = tf.shape(input_frames)
        batch_size = input_shape[0]

        preds = tf.zeros([batch_size, c.OUT_LEN])
        last_input = input_frames

        with tf.name_scope('convolutions'):
            for i in xrange(len(self.conv_ws)):
                # Convolve layer and activate with ReLU
                preds = tf.nn.conv2d(last_input, self.conv_ws[i], [1, 1, 1, 1], padding=c.PADDING_D)
                preds = leaky_relu(preds + self.conv_bs[i])
                last_input = preds

        # pooling layer
        with tf.name_scope('pooling'):
            preds = tf.nn.max_pool(preds, [1, 2, 2, 1], [1, 2, 2, 1], padding=c.PADDING_D)

        # flatten preds for dense layers
        shape = preds.get_shape().as_list()
        preds = tf.reshape(preds, [-1, shape[1] * shape[2] * shape[3]])

        # fully-connected layers
        with tf.name_scope('fully-connected'):
            for i in xrange(len(self.fc_ws)):
                preds = tf.matmul(preds, self.fc_ws[i]) + self.fc_bs[i]

                # Activate with ReLU (or Sigmoid for last layer)
                if i == len(self.fc_ws) - 1:
                    preds = tf.sigmoid(preds)
                else:
                    # preds = tf.nn.relu(preds)
                    preds = leaky_relu(preds)

        # clip preds between [.1, 0.9] for stability
        with tf.name_scope('clip'):
            preds = tf.clip_by_value(preds, 0.1, 0.9)

        return preds

