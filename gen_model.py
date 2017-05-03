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

    # noinspection PyAttributeOutsideInit
    def define_graph(self, discriminator):
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

                self.train_vars = [] # the variables to train
                self.summaries_train = []
                self.preds_train = []  # the generated images
                self.gts_train = []  # the ground truth images
                self.d_preds = []  # the predictions from the discriminator model

                self.summaries_test = []
                self.preds_test = []  # the generated images
                self.gts_test = []  # the ground truth images
                self.ws = []
                self.bs = []

                # Sets up the generator network
                with tf.name_scope('scale_1'):
                    with tf.name_scope('setup'):

                        # Sets up the weights and biases
                        scale_ws = []
                        scale_bs = []
                        for i in xrange(len(self.kernel_sizes)):
                            scale_ws.append(w([self.seqlen, self.kernel_sizes[i], self.kernel_sizes[i],
                                           self.feature_maps[i],
                                         self.feature_maps[i + 1]]))
                            scale_bs.append(b([self.feature_maps[i + 1]]))

                        # Add to trainable parameters
                        self.train_vars += scale_ws
                        self.train_vars += scale_bs

                        self.ws.append(scale_ws)
                        self.bs.append(scale_bs)

                    with tf.name_scope('calculation'):
                        # Perform train calculation
                        train_preds, train_gts = self.generate_predictions(self.input_frames_train, self.gt_frames_train)

                        self.preds_train.append(train_preds)
                        self.gts_train.append(train_gts)

                        test_preds, test_gts = self.generate_predictions(self.input_frames_test, self.gt_frames_test)

                        self.preds_test.append(test_preds)
                        self.gts_test.append(test_gts)

                with tf.name_scope('d_preds'):
                    self.d_preds = []
                    with tf.name_scope('scale_1'):
                        with tf.name_scope('calculation'):

                            # IMPLEMENT THIS FIRST (INCOMPLETE)

                            self.d_preds.append(discriminator.nets.generate_predictions())
                with tf.name_scope('train'):
                    self.global_loss = combined_loss(self.d_preds, self.input_frames_train, self.preds_train, self.gts_train)
                    self.global_step = tf.Variable(0, trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, name='optim')
                    self.train_op = self.optimizer.minimize(self.global_loss, global_step=self.global_step,
                                                            var_list=self.train_vars, name='train_op')
                    loss_summary = tf.summary.scalar('train_loss_G', self.global_loss)
                    self.summaries_train.append(loss_summary)

    def generate_predictions(self, inputs, gts):
        preds = inputs
        # Perform convolutions
        with tf.name_scope('convolutions'):

            for i in xrange(len(self.kernel_sizes)):
                # 3D Convolve
                preds = tf.nn.conv3d(preds, self.ws[i], [1, 1, 1, 1, 1], padding='same')

                # Add activation units (tanh for last layer and relu for the rest)
                if i == len(self.kernel_sizes) - 1:
                    preds = tf.nn.tanh(preds + self.bs[i])
                else:
                    preds = tf.nn.relu(preds + self.bs[i])
        return preds, gts


    def train_step(self, batch, discriminator=None):

        input_frames = batch[:, :, :, :, :-3]
        gt_frames = batch[:, :, :, :, -3:]

        feed_dict = {self.input_frames_train:input_frames, self.gt_frames_train:gt_frames}

        # Run the generator first to get generated frames
        preds = self.sess.run(self.preds_train, feed_dict=feed_dict)

        _, global_loss, global_step, summaries = self.sess.run([self.train_op, self.global_loss, self.global_step,
                                                self.summaries_train],
                                                feed_dict=feed_dict)

        # IMPLEMENT USER OUTPUTS HERE (INCOMPLETE)

        return global_step



