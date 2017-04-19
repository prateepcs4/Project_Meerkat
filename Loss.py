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
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y, _ = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

# Returns the 3D grad loss
def cross_corr_loss(in_frames, gen_frames, gt_frames, alpha):
    frames_numpy = sess.run([in_frames, gen_frames, gt_frames])
    in_frames_numpy = frames_numpy[0]
    gen_frames_numpy = frames_numpy[1]
    gt_frames_numpy = frames_numpy[2]

    patches_frame_prev = patchify(in_frames_numpy[0][:][:][0], [3, 3])
    patches_frame_cur = patchify(in_frames_numpy[1][:][:][0], [3, 3])
    print np.shape(patches_frame_prev)


# Tester for loss functions

BATCH_SIZE = 20
NUM_SCALES = 5
MAX_P      = 5
MAX_ALPHA  = 1

in_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gen_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
gt_frames = tf.ones([BATCH_SIZE, 3, 32, 32, 3])
res_tru = 0

sess.run(cross_corr_loss(in_frames, gen_frames, gt_frames, 1))
# assert res == res_tru, 'Failed'

