"""
Algorithms 2 and 3 from Polson et al. (2013)
"""
import tensorflow as tf
import numpy as np
from trainerlog import get_logger
import tensorflow_probability as tfp

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

# mu <= t
def truncated_ig_musmall(mu, t):
    sqrt_Y = tf.random.normal(0, 1)
    Y = sqrt_Y * sqrt_Y
    X 

    U = tf.random.uniform(shape=[1])
    pass

# mu > t
def truncated_ig_mularge(mu, t):
    z = 1.0 / mu
    
    E = tf.random.exponential(shape=[1])
    E_prime = tf.random.exponential(shape=[1])
    while E * E > 2 * E_prime / t:

    U = tf.random.uniform(shape=[1])
    if U <= alpha:
        return X
    else:
        return truncated_ig_mularge(mu, t)


def truncated_inverse_gaussian(mu, t):
    if mu > t:
        return truncated_ig_mularge(mu, t)
    else:
        return truncated_ig_musmall(mu, t):
