import numpy as np
import tensorflow as tf

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

def _3d_grad_loss(gen_frames, gt_frames, power):
    _3d_gdl = 0
