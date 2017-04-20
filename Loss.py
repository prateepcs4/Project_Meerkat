import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()


# Returns the binary cross-entropy loss calculated from the predicted outputs (0/1) and the targets (0/1)
def bce_loss(preds, targets):
    return tf.squeeze(-1 * (tf.matmul(targets, np.log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, np.log10(1 - preds), transpose_a=True)))


# Returns the lp loss
def l1_loss(gen_frames, gt_frames, l_p):
    lp = 0
    for i in xrange(len(gen_frames)):
        lp = lp + tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i]) ** l_p)
    return lp


# Utility function for fast computation of patches from a 2D numpy array
def patchify(input, ksize, stride):
    patches = tf.extract_image_patches(input, ksizes=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                                               rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.reshape(patches, [-1, ksize, ksize, 1])
    return patches

patch_size_cur = 2
patch_size_prev = patch_size_cur + 2


# Returns the cross correlation loss
def cross_corr_loss(in_frames, gen_frames, gt_frames, alpha):
    n_batches, n_frames, n_rows, n_cols, n_channels = in_frames.get_shape().as_list()
    pad = tf.constant([[2, 2], [2, 2], [0, 0]])
    # Iterate over batches
    for batch_index in xrange(n_batches):
        # Extract the last input frame
        last_in_frame = in_frames[batch_index][-1]
        # print last_in_frame.get_shape()
        for frame_index in xrange(n_frames):
            prev_frame = last_in_frame if frame_index == 0 else gen_frames[batch_index][frame_index-1]
            cur_frame = gen_frames[batch_index][frame_index]
            # print prev_frame.get_shape()
            padded_prev_frame = tf.pad(prev_frame, pad, "CONSTANT")
            # Extract patches from the cur_frame
            cur_frame = cur_frame[:,:,0]
            cur_frame = tf.expand_dims(tf.expand_dims(cur_frame, 0), -1)
            patch_cur_frames = patchify(cur_frame, patch_size_cur, 2)



# Tester for loss functions

BATCH_SIZE = 20
NUM_SCALES = 5
MAX_P = 5
MAX_ALPHA = 1

in_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gen_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gt_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
res_tru = 0

sess.run(cross_corr_loss(in_frames, gen_frames, gt_frames, 1))
# assert res == res_tru, 'Failed'
