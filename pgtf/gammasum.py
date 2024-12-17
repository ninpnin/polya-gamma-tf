import tensorflow as tf
import numpy as np
from trainerlog import get_logger
import tensorflow_probability as tfp

PI = tf.constant(np.pi)
PI2 = PI ** 2

k = tf.cast(tf.range(200, delta=1) + 1, tf.float32)
@tf.function
def pg_truncated(alphas, c):
    L = alphas.shape[0]
    g = tf.random.gamma(shape=[200], alpha=tf.cast(alphas, tf.float32))
    c2 = 0.25 * (tf.cast(c, tf.float32) ** 2) / PI2

    k_offset = (k - 0.5) ** 2
    k_plus_c = tf.transpose(tf.tile([k], [L, 1])) + tf.tile([c2], [200, 1])

    x = 0.5 * (g / k_plus_c) / PI2 
    y = tf.reduce_sum(x, axis=0)

    return y