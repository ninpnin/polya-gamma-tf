from pgtf.inverse_gaussian import *
import polars as pl
import tensorflow_probability as tfp
import progressbar
import seaborn as sns
from matplotlib import pyplot as plt
from trainerlog import get_logger

LOGGER = get_logger("invg", splitsec=True)
t = tf.constant(0.64)
R = 8000
mu = tf.constant(0.8)

d = {"x": [], "method": [] }


mus = tf.random.uniform(shape=[1000000], dtype=tf.float32) * 0.3 + t
LOGGER.info("Batched")
batched = truncated_ig_mularge_batch(mus, t)
LOGGER.info("end")

ref_dist = tfp.distributions.InverseGaussian(mu, 1.0)
for _ in progressbar.progressbar(range(R)):
    X = truncated_ig_mularge(mu, t)
    d["x"].append(X)
    d["method"].append("polson")

    X_prime = ref_dist.sample()
    if X_prime <= t:
        d["x"].append(X_prime)
        d["method"].append("ref")

df = pl.DataFrame(d)
print(df)

sns.kdeplot(df, x="x", hue="method", common_norm=False)
plt.show()