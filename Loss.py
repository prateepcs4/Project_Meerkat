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
def patchify(img, patch_shape):
    X, Y, a = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y, a)
    X_str, Y_str, a_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, a_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def sliding_patches(a, block_size):
    hBSZ = (block_size - 1) // 2
    a_ext = np.dstack(np.pad(a[..., i], hBSZ, 'edge') for i in range(a.shape[2]))
    return patchify(a_ext, (block_size, block_size))


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
            cur_frame = gen_frames[frame_index]
            # print prev_frame.get_shape()
            padded_prev_frame = tf.pad(prev_frame, pad, "CONSTANT")
#             Iterate through the cur_frame in non-overlapping sequence


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
