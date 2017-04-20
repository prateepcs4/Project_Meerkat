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


# Utility function for fast computation of patches from a 4D tensor (batchsize x H x W x channels) (supports batch mode)
def patchify(input, ksize, stride, channels):
    patches = tf.extract_image_patches(input, ksizes=[1, ksize, ksize, 1], strides=stride,
                                               rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.reshape(patches, [-1, ksize, ksize, channels])
    return patches

# Utility funtion for calculating the correlation score between a base matrix and an input template
def xcorr(base, patch):
    base_r_channel = tf.expand_dims(base[:, :, :, 0], -1)
    patch_r_channel = tf.expand_dims(tf.expand_dims(patch[0, :, :, 0], -1), -1)
    corr = tf.reduce_mean(tf.nn.conv2d(base_r_channel, patch_r_channel, [1, 1, 1, 1], padding='VALID'))
    print sess.run(corr)

stride = 2
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
            padded_prev_frame = tf.expand_dims(tf.pad(prev_frame, pad, "CONSTANT"), 0)
            # Extract patches from the cur_frame
            cur_frame = tf.expand_dims(cur_frame, 0)
            patches_cur_frame = patchify(cur_frame, patch_size_cur, [1, stride, stride, 1], 3)
            n_patches, patch_height, _, _ = patches_cur_frame.get_shape().as_list()
            # Extract patches from the previous frame for mapping
            patches_prev_frame = patchify(padded_prev_frame, patch_size_prev, [1, stride-1, stride-1, 1], 3)
            # Iterate over the patches
            dim_patch_grid = n_rows/patch_height
            for cur_patch_index in xrange(n_patches):
                left_x = 2 * (cur_patch_index / dim_patch_grid)
                left_y = 2 * (cur_patch_index % dim_patch_grid)
                cur_prev_patch = tf.slice(padded_prev_frame, [0, left_x, left_y, 0], [1, patch_size_prev,
                                                                                      patch_size_prev, 3])
                score = xcorr(cur_prev_patch, tf.expand_dims(patches_cur_frame[cur_patch_index], 0))

# Tester for patchify
def patch_test(input):
    input = tf.pad(input, [[0,0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    patch = patchify(input, 2, 2, 3)
    print patch.get_shape()
    print sess.run(patch[1,:,:,1])

# patch_test(tf.ones([1, 10, 10, 3]))

# Tester for loss functions

BATCH_SIZE = 20
NUM_SCALES = 5
MAX_P = 5
MAX_ALPHA = 1

in_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gen_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gt_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
res_tru = 0

cross_corr_loss(in_frames, gen_frames, gt_frames, 1)
# assert res == res_tru, 'Failed'
