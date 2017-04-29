from Loss import combined_loss
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from skimage.transform import resize

class Generator_Model:
    def __init__(self, session, seqlen, frame_height, frame_width, channel):
        self.sess = session
        self.seqlen = seqlen
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_chanels = channel

    def define_graph(self):
        # Sets up the model graph
        with tf.name_scope('generator'):
            # Data
            with tf.name_scope('data'):
                # Prepare the placeholder for input frames
                self.input_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])
                # Prepare the placeholder for ground truth frames
                self.gt_frames_train = tf.placeholder(tf.float32, shape=[None, self.seqlen, self.frame_height,
                                                                            self.frame_width, self.num_chanels])

                # Variable batchsize
                self.batch_size_train = tf.shape(self.input_frames_train)[0]
