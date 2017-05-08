import tensorflow as tf
import getopt
import sys
import os
import constants as c
from gen_model import Generator_Model
from disc_model import Discriminator_Model
from utils import get_train_batch, get_test_batch

class Master_Runner:

    def __init__(self, num_steps, model_load_path, num_test_rec = 1):

        self.global_step = 0
        self.num_steps = num_steps
        self.num_test_rec = num_test_rec

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        print 'Initializing discriminator...'
        self.d_model = Discriminator_Model(self.sess, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, c.FMAPS_D, c.KERNEL_SIZES_D, c.FC_LAYER_SIZES_D)

        print 'Initializing generator...'
        self.g_model = Generator_Model(self.sess, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, c.FMAPS_G, c.KERNEL_SIZES_G)

        print 'Initializing variables...'
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.sess.run(tf.global_variables_initializer())

        # if load path specified, load a saved model
        if model_load_path is not None:
            self.saver.restore(self.sess, model_load_path)
            print 'Model restored from ' + model_load_path

    def train(self):

        for i in xrange(self.num_steps):

            # update discriminator
            batch = get_train_batch()
            print 'Training discriminator...'
            self.d_model.train_step(batch)

            # update generator
            batch = get_train_batch()
            print 'Training generator...'
            self.global_step = self.g_model.train_step(batch)

            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print '-' * 30
                print 'Saving models...'
                self.saver.save(self.sess, os.path.join(c.MODEL_SAVE_DIR, 'model.ckpt'), global_step=self.global_step)
                print 'Saved models'
                print '-' * 30


