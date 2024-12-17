import tensorflow as tf
import numpy as np
from trainerlog import get_logger
import tensorflow_probability as tfp

PI = tf.constant(np.pi)
PI2 = PI ** 2

"""
function rand_Jstar(z::Real, rng::AbstractRNG)
    t = 0.64 # paper recommends this constant
    K = 0.125*π^2 + 0.5*z^2
    p = 0.5*π*inv(K) * exp(-t*K)
    q = 2*exp(-z + logcdf(InverseGaussian(inv(z), 1.0), t))
    while true
        # Generate X
        if rand(rng, Uniform(0.0, 1.0)) < p/(p+q) #U ~ Uniform(0,1)
            # Draw exponential(K) truncated above t
            X = t + inv(K)*rand(rng, Exponential())
        else
            # Draw inverseGaussian(1/z, 1) truncated below t
            # WARNING: This implementation is unstable. For z ≈ 0, an infinite loop occurs
            # due to the InverseGaussian parameter being infinite. z + 0.0001 is a hack to 
            # address the case where z = 0. In the future, Algorithms 2 and 3 of 
            # Polson et al. 2013 should be implemented to fix this.
            X = rand(rng, truncated(InverseGaussian(inv(z+0.0001), 1.0); upper = t))
        end
        # Accumulate a(X) to S
        S = a_coefs(0, X, t)
        Y = S * rand(rng, Uniform(0.0, 1.0)) #V ~ Uniform(0,1)
        n = Int(0)
        while true
            n += 1
            if isodd(n)
                S -= a_coefs(n, X, t)
                if Y <= S
                    return X # accept X and exit
                end
            else
                S += a_coefs(n, X, t)
                if Y > S
                    break # reject X and start over
                end
            end
        end
    end
end
"""

#@tf.function
def random_exponential(rate=1.0, count=1):
    return tfp.distributions.Exponential(rate=rate).sample(count)

@tf.function
def truncated_inverse_gaussian(a,b, t):
    x = tfp.distributions.InverseGaussian(a,b).sample()
    if x > t:
        return truncated_inverse_gaussian(a,b, t)
    else:
        return x

@tf.function
def log_cdf_invgaussian(x):
    return tfp.distributions.InverseGaussian(1.0 / z, 1.0).log_cdf(x)

@tf.function
def rand_jstar(z):
    t = 0.64
    K = (PI) ** 2 * 0.125 + 0.5 * (z) ** 2
    p = 0.5 * PI/ K * tf.exp(-t*K)
    q = 2 * tf.exp(-z +log_cdf_invgaussian(t))

    if tf.random.uniform(shape=[1]) < p / (p + q): #U ~ Uniform(0,1)
        X = t + random_exponential() / K

    else:
        X = truncated_inverse_gaussian(1.0/(z+0.0001), 1.0, t)

    return jstar_loop(X)

@tf.function
def jstar_loop(X):
    S = a_coefs(0, X, t)
    Y = S * tf.random.rand(0.0, 1.0) #V ~ Uniform(0,1)
    n = 0

    while True:
        n += 1
        if n % 2 == 1:
            S += - a_coefs(n, x, t)
            if Y <= S:
                return X
        else:
            S += a_coefs(n, x, t)
            if Y > S:
                return jstar_loop(X)


def a_coefs(n, x, t):
    common_terms = PI * (n + 0.5) * tf.exp(-(n+0.5) ** 2)
    if x <= t:
        return common_terms * tf.exp(-2 / x)
    else:
        return common_terms * tf.exp(-x * 0.5 * PI2)

elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
print(alternate)

#rates = tf.constant([3.0, 5.0, 2.0])
#counts = tf.constant([1, 1, 1])
#vals = tf.map_fn(fn=lambda x: x[0] * x[1], elems=(rates, counts))
#print(vals)

z = 0.74
x_PG = rand_jstar(z)

print(x_PG)