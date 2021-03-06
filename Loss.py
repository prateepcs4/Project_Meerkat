import numpy as np
import tensorflow as tf
import os
from tfutils import log10

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()


# Returns the binary cross-entropy loss calculated from the predicted outputs (0/1) and the targets (0/1)
def bce_loss(preds, targets):
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))

# Returns the lp loss
def lp_loss(gen_frames, gt_frames, l_p):
    n_batches, n_frames, n_rows, n_cols, n_channels = gen_frames.get_shape().as_list()
    lp = 0
    for i in xrange(n_batches):
        lp = lp + tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i]) ** l_p)
    return lp


# Utility function for fast computation of patches from a 4D tensor (batchsize x H x W x channels) (supports batch mode)
def patchify(input, ksize, stride, channels):
    patches = tf.extract_image_patches(input, ksizes=[1, ksize, ksize, 1], strides=stride,
                                       rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.reshape(patches, [-1, ksize, ksize, channels])
    return patches


# Utility funtion for calculating the cross correlation score between a base matrix and an input template
def xcorr(base, patch):
    patch = tf.expand_dims(patch[0, :, :, :], -1)
    corr = tf.reduce_mean(tf.nn.conv2d(base, patch, [1, 1, 1, 1], padding='VALID'))
    return corr


stride = 2
patch_size_cur = 2
patch_size_prev = patch_size_cur + 2


# Returns the cross correlation loss
def cross_corr_loss(in_frames, gen_frames):
    n_batches, n_frames, n_rows, n_cols, n_channels = in_frames.get_shape().as_list()
    pad = tf.constant([[2, 2], [2, 2], [0, 0]])
    scores = np.ndarray(shape=(n_batches, 1))
    # Iterate over batches
    for batch_index in xrange(n_batches):
        # Extract the last input frame
        last_in_frame = in_frames[batch_index][-1]
        # print last_in_frame.get_shape()
        total_xcorr_score = 0
        for frame_index in xrange(n_frames):
            prev_frame = last_in_frame if frame_index == 0 else gen_frames[batch_index][frame_index - 1]
            cur_frame = gen_frames[batch_index][frame_index]
            padded_prev_frame = tf.expand_dims(tf.pad(prev_frame, pad, "CONSTANT"), 0)
            # Extract patches from the cur_frame
            cur_frame = tf.expand_dims(cur_frame, 0)
            patches_cur_frame = patchify(cur_frame, patch_size_cur, [1, stride, stride, 1], 3)
            n_patches, patch_height, _, _ = patches_cur_frame.get_shape().as_list()
            # Iterate over the patches
            dim_patch_grid = n_rows / patch_height
            for cur_patch_index in xrange(n_patches):
                # Patch matching logic between prev_frame and cur_patch
                left_x = 2 * (cur_patch_index / dim_patch_grid)
                left_y = 2 * (cur_patch_index % dim_patch_grid)
                cur_prev_patch = tf.slice(padded_prev_frame, [0, left_x, left_y, 0], [1, patch_size_prev,
                                                                                      patch_size_prev, 3])
                score = xcorr(cur_prev_patch, tf.expand_dims(patches_cur_frame[cur_patch_index], 0))
                total_xcorr_score += score
        scores[batch_index, 0] = sess.run(1 - total_xcorr_score)
    return tf.reduce_mean(tf.convert_to_tensor(scores, dtype=tf.float32))


# Utility function for calculating the Euclidean distance between two matrices
def get_distance(frame_1, frame_2):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(frame_1, frame_2))))


# Returns the contrastive divergence loss
def contrastive_loss(preds, in_frames, gen_frames):
    delta = tf.constant(1, dtype=tf.float32)
    n_batches, n_frames, n_rows, n_cols, n_channels = gen_frames.get_shape().as_list()
    scores = np.ndarray(shape=(n_batches, 1))
    for batch_index in xrange(n_batches):
        # Extract the last input frame
        last_in_frame = in_frames[batch_index][-1]
        # Iterate over the frames
        total_score = tf.constant(0, dtype=tf.float32)
        for frame_index in xrange(n_frames):
            prev_frame = last_in_frame if frame_index == 0 else gen_frames[batch_index][frame_index - 1]
            cur_frame = gen_frames[batch_index][frame_index]
            distance = get_distance(prev_frame, cur_frame)
            score = preds[batch_index][frame_index] * distance + (1 - preds[batch_index][frame_index]) * \
                                                                      tf.nn.relu(tf.subtract(delta, distance))
            total_score = tf.add(total_score, score)

        scores[batch_index, 0] = sess.run(total_score)
    return tf.reduce_mean(tf.convert_to_tensor(scores, dtype=tf.float32))


# Returns the combined loss value
def combined_loss(preds, in_frames, gen_frames, gt_frames):
    n_batches, n_frames, _, _, _ = in_frames.get_shape().as_list()
    adv_score = bce_loss(preds, tf.ones([n_batches, n_frames]))
    lp_score = lp_loss(gen_frames, gt_frames, 2)
    cross_corr_score = cross_corr_loss(in_frames, gen_frames)
    contrastive_score = contrastive_loss(preds, in_frames, gen_frames)

    final_score = adv_score + lp_score + cross_corr_score + contrastive_score
    return final_score


# Tester for patchify
def patch_test(input):
    input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    patch = patchify(input, 2, 2, 3)
    print patch.get_shape()
    print sess.run(patch[1, :, :, 1])


# patch_test(tf.ones([1, 10, 10, 3]))

# Tester for loss functions

BATCH_SIZE = 2
NUM_SCALES = 5
MAX_P = 5
MAX_ALPHA = 1

in_frames = tf.ones([BATCH_SIZE, 32, 32, 3])
gen_frames = tf.zeros([BATCH_SIZE, 32, 32, 3])
gt_frames = tf.ones([BATCH_SIZE, 32, 32, 3])
preds = tf.ones([BATCH_SIZE, 3])
res_tru = 0

print sess.run(bce_loss(preds, tf.zeros([BATCH_SIZE, 3])))
# assert res == res_tru, 'Failed'
