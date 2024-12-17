"""
Algorithms 2 and 3 from Polson et al. (2013)
"""
import tensorflow as tf
import numpy as np
from trainerlog import get_logger
import tensorflow_probability as tfp
import sys

"""
 * When 1/z < 0.64, We use a known sampling algorithm from Devroye
 * (1986), page 149. We sample until the generated variate is less than 0.64.
 *
 * When mu > 0.64, we use a Inverse-Chi-square distribution as a proposal,
 * as explained in [1], page 134. To generate a sample from this proposal, we
 * sample from the tail of a standard normal distribution such that the value
 * is greater than 1/sqrt(0.64). Once we obtain the sample, we square and invert
 * it to obtain a sample from a Inverse-Chi-Square(df=1) that is less than t.
 * An efficient algorithm to sample from the tail of a normal distribution
 * using a pair of exponential variates is shown in Devroye (1986) [page 382]
 * & Devroye (2009) [page 7]. This sample becomes our proposal. We accept the
 * sample only if we sample a standard uniform value less than the acceptance
 * probability. The probability is exp(-0.5 * z2 * x) (Refer to Appendix 1 of
 * [1] for its derivation).
"""

@tf.function
def rand_unif():
    return tf.random.uniform(shape=(), dtype=tf.float32)

@tf.function
def rand_exp():
    return - tf.math.log(1.0 - rand_unif())

# Algorithm 2: mu > t
@tf.function
def truncated_ig_mularge(mu, t):
    z = 1.0 / mu
    alpha = tf.constant(0.0)
    U = tf.constant(1.0)

    # While loops are terribly slow in TF;
    # We gotta make everything batched
    while U > alpha:
        E = rand_exp()
        E_prime = rand_exp()
        while E * E > 2 * E_prime / t:
            E = rand_exp()
            E_prime = rand_exp()

        X = t / ((1 + t * E) ** 2)
        alpha = tf.exp(- 0.5 * (z ** 2) * X)
        U = rand_unif()
    return X

def truncated_ig_mularge_batch(mus, t):
    N = len(mus)
    z = 1.0 / mus

    E = - tf.math.log(1.0 - tf.random.uniform(shape=[N], dtype=tf.float32))
    E_prime = - tf.math.log(1.0 - tf.random.uniform(shape=[N], dtype=tf.float32))

    X = t / ((1 + t * E) ** 2)
    alphas = tf.exp(- 0.5 * (z ** 2) * X)
    U = tf.random.uniform(shape=(), dtype=tf.float32)
    return X, E * E > 2 * E_prime / t, U <= alphas
    
# Algorithm 3: mu <= t
def truncated_ig_musmall(mu, t):
    sqrt_Y = tf.random.normal(0, 1)
    Y = sqrt_Y * sqrt_Y
    X 

    U = tf.random.uniform(shape=[1])
    pass


def truncated_inverse_gaussian(mu, t):
    if mu > t:
        return truncated_ig_mularge(mu, t)
    else:
        return truncated_ig_musmall(mu, t)
