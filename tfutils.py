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