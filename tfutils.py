import tensorflow as tf
import numpy as np


def w(shape, stddev=0.01):
    """
    @return A weight layer with the given shape and standard deviation. Initialized with a
            truncated normal distribution.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def b(shape, const=0.1):
    """
    @return A bias layer with the given shape.
    """
    return tf.Variable(tf.constant(const, shape=shape))

def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def conv_out_size(i, p, k, s):
    """
    Gets the output size for a 3D convolution. (Assumes square input and kernel).
    @param i: The side length of the input.
    @param p: The padding type (either 'SAME' or 'VALID').
    @param k: The side length of the kernel.
    @param s: The stride.
    @type i: int
    @type p: string
    @type k: int
    @type s: int
    @return The side length of the output.
    """
    # convert p to a number
    if p == 'SAME':
        p = k // 2
    elif p == 'VALID':
        p = 0
    else:
        raise ValueError('p must be "SAME" or "VALID".')

    return int(((i + (2 * p) - k) / s) + 1)
