import tensorflow as tf
from tfutils import w, b, conv_out_size
from disc_model_engine import Discriminator_Model_Engine
import constants as c
from Loss import bce_loss

class Discriminator_Model:
    def __init__(self, session, height, width, feature_maps, kernel_sizes, fc_layer_sizes):

        # Initializes a discriminator model
        self.sess = session
        self.height = height
        self.width = width
        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes
        self.fc_layer_sizes = fc_layer_sizes

        self.train_vars = []

    def setup_nets(self):
        # Sets up networks. Each makes prediction for images

        self.nets = []
        with tf.name_scope('scale_1'):
            model = Discriminator_Model_Engine(self.height, self.width, self.feature_maps,
                                               self.kernel_sizes, self.fc_layer_sizes)
            self.nets.append(model)
            self.train_vars += model.train_vars

    def define_graph(self, generator):
        # Sets up the model graph in Tensorflow
        with tf.name_scope('discriminator'):
            self.input_clips = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3*(c.HIST_LEN + c.OUT_LEN)])

            self.g_input_frames = self.input_clips[:, :, :, :3*c.HIST_LEN]
            self.gt_frames = self.input_clips[:, :, :, 3*c.HIST_LEN:]
            input_shape = tf.shape(self.g_input_frames)
            batch_size = input_shape[0]

            # Get generator frames
            with tf.name_scope('gen_frames'):
                self.g_preds = [] #Generated frames
                self.gts = [] #Ground truths

                train_preds, train_gts = generator.generate_predictions(self.g_input_frames, self.gt_frames)

                self.g_preds.append(train_preds)
                self.gts.append(train_gts)

            # Concatenate the generated images and ground truths
            self.inputs = []
            self.inputs.append(tf.concat([self.g_preds[0], self.gts[0]], 0))

            # Create the labels
            self.labels = tf.concat([tf.zeros([batch_size, c.OUT_LEN]), tf.ones([batch_size, c.OUT_LEN])], 0)

            # Prediction tensors
            self.preds = []

            with tf.name_scope('scale_1'):
                with tf.name_scope('calculation'):
                    self.preds.append(self.nets[0].generate_predictions(self.inputs[0]))

            with tf.name_scope('training'):
                self.global_loss = bce_loss(self.preds, self.labels)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.optimizer = tf.train.AdamOptimizer(c.LRATE_D, name='optimizer')
                self.train_op = self.optimizer.minimize(self.global_loss,
                                                        global_step=self.global_step,
                                                        var_list=self.train_vars,
                                                        name='train_op')

                all_preds = tf.stack(self.preds)
                # Calculate the accuracy
                for i in xrange(c.OUT_LEN):
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(all_preds[i]), self.labels[i]), tf.int8))


    def train_step(self, batch):
        # Runs a training step using the global loss on the discriminator network.
        feed_dict = {self.input_clips : batch}
        _, global_loss, global_step = self.sess.run([self.train_op, self.global_loss, self.global_step],
                                                               feed_dict=feed_dict)

        if global_step % c.STATS_FREQ == 0:
            print 'DiscriminatorModel: step %d | global loss: %f' % (global_step, global_loss)

        return global_step


