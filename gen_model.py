from Loss import combined_loss
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from skimage.transform import resize
from tfutils import w, b

class Generator_Model:
    def __init__(self, session, seqlen, frame_height, frame_width, channel, feature_maps, kernel_sizes):
        self.sess = session
        self.seqlen = seqlen
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_chanels = channel
        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes

    def define_graph(self):
        # Sets up the model graph
        with tf.name_scope('generator'):
            # Data
            with tf.name_scope('data'):
                # Prepare the placeholder for input frames
                self.input_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])
                # Prepare the placeholder for ground-truth frames
                self.gt_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])

                # Variable batchsize
                self.batch_size_train = tf.shape(self.input_frames_train)[0]

                # Sets up the generator network
                with tf.name_scope('scale_1'):
                    with tf.name_scope('setup'):

                        # Set up the weights and biases
                        ws = []
                        bs = []

                        for i in xrange(len(self.kernel_sizes)):
                            ws.append(w([self.seqlen, self.kernel_sizes[i], self.kernel_sizes[i], self.feature_maps[i],
                                         self.feature_maps[i + 1]]))
                            bs.append(b([self.feature_maps[i + 1]]))

                    with tf.name_scope('calculation'):
                        def calculate(height, width, inputs, gts):

                            preds = inputs
                            # Perform convolutions
                            with tf.name_scope('convolutions'):
                                for i in xrange(len(self.kernel_sizes)):
                                    preds = tf.nn.conv3d(preds, ws[i], [1, 1, 1, 1, 1], padding='same')


