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
                # Prepare the placeholder for train input frames
                self.input_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])
                # Prepare the placeholder for train ground-truth frames
                self.gt_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])

                # Prepare the placeholder for test input frames
                self.input_frames_test = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])
                # Prepare the placeholder for test ground-truth frames
                self.gt_frames_test = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                         self.frame_width, self.num_chanels])

                # Variable batchsize
                self.batch_size_train = tf.shape(self.input_frames_train)[0]
                self.batch_size_test = tf.shape(self.input_frames_test)[0]

                self.summaries_train = []
                self.preds_train = []  # the generated images
                self.gts_train = []  # the ground truth images
                self.d_preds = []  # the predictions from the discriminator model

                self.summaries_test = []
                self.preds_test = []  # the generated images
                self.gts_test = []  # the ground truth images

                # Sets up the generator network
                with tf.name_scope('scale_1'):
                    with tf.name_scope('setup'):

                        # Sets up the weights and biases
                        ws = []
                        bs = []

                        for i in xrange(len(self.kernel_sizes)):
                            ws.append(w([self.seqlen, self.kernel_sizes[i], self.kernel_sizes[i], self.feature_maps[i],
                                         self.feature_maps[i + 1]]))
                            bs.append(b([self.feature_maps[i + 1]]))

                    with tf.name_scope('calculation'):
                        def calculate(inputs, gts):
                            preds = inputs
                            # Perform convolutions
                            with tf.name_scope('convolutions'):
                                for i in xrange(len(self.kernel_sizes)):
                                    # 3D Convolve
                                    preds = tf.nn.conv3d(preds, ws[i], [1, 1, 1, 1, 1], padding='same')

                                    # Add activation units (tanh for last layer and relu for the rest)
                                    if i == len(self.kernel_sizes) - 1:
                                        preds = tf.nn.tanh(preds + bs[i])
                                    else:
                                        preds = tf.nn.relu(preds + bs[i])
                            return preds, gts

                        # Perform train calculation
                        train_preds, train_gts = calculate(self.input_frames_train, self.gt_frames_train)

                        self.preds_train.append(train_preds)
                        self.gts_train.append(train_gts)

                        # Run the network first to get generated frames, run the
                        # discriminator on those frames to get d_preds, then run this
                        # again for the loss optimization.
                        self.d_preds.append(tf.placeholder(tf.float32, [None, 1]))

                        test_preds, test_gts = calculate(self.input_frames_test, self.gt_frames_test)

                        self.preds_test.append(test_preds)
                        self.gts_test.append(test_gts)


