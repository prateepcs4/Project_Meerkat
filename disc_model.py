import tensorflow as tf
from tfutils import w, b, conv_out_size
from disc_model_engine import Discriminator_Model_Engine

class Discriminator_Model:
    def __init__(self, session, seqlen, height, width, feature_maps, kernel_sizes, fc_layer_sizes):

        # Initializes a discriminator model
        self.sess = session
        self.seqlen = seqlen
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
            model = Discriminator_Model_Engine(self.height, self.width, self.seqlen, self.feature_maps,
                                               self.kernel_sizes, self.fc_layer_sizes)
            self.nets.append(model)
            self.train_vars += model.train_vars

    def define_graph(self, generator):
        # Sets up the model graph in Tensorflow
        with tf.name_scope('discriminator'):
            self.input_clips = tf.placeholder(tf.float32, shape=[None, self.seqlen + 1, self.height, self.width, 3])

            self.g_input_frames = self.input_clips[:, :-1, :, :, :]
            self.gt_frames = self.input_clips[:, -1:, :, :, :]
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
            self.inputs.append(tf.concat(0, [self.g_preds[0], self.gts[0]]))





