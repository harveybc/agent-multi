"""Bayesian utility functions shared by ioin plugins.

Centralizes posterior/prior factories, KL annealing callback builder and
Monte-Carlo Welford uncertainty estimation to reduce duplication across
ANN/CNN/LSTM/Transformer style plugins.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import Callback

# ---------------------------------------------------------------------------
# Posterior / Prior (mean-field) factories
# ---------------------------------------------------------------------------

def posterior_mean_field(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0  # ignore bias for simplification
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.0))
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=123), dtype=dtype, trainable=trainable,
                      name=(f"{name}_loc" if name else "posterior_loc"))
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=124), dtype=dtype, trainable=trainable,
                        name=(f"{name}_scale" if name else "posterior_scale"))
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_r, scale=scale_r),
        reinterpreted_batch_ndims=len(kernel_shape),
    )

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_r, scale=scale_r),
        reinterpreted_batch_ndims=len(kernel_shape),
    )

# ---------------------------------------------------------------------------
# KL Annealing Callback
# ---------------------------------------------------------------------------
class _KLAnnealCallback(Callback):
    def __init__(self, plugin, target_kl: float, anneal_epochs: int):
        super().__init__()
        self.plugin = plugin
        self.target_kl = target_kl
        self.anneal_epochs = max(1, anneal_epochs)
    def on_epoch_begin(self, epoch, logs=None):
        frac = min(1.0, (epoch + 1) / self.anneal_epochs)
        self.plugin.kl_weight_var.assign(self.target_kl * frac)

def build_kl_anneal_callback(plugin, target_kl: float, anneal_epochs: int):
    return _KLAnnealCallback(plugin, target_kl, anneal_epochs)

# ---------------------------------------------------------------------------
# Monte Carlo predictive mean & std (Welford incremental)
# ---------------------------------------------------------------------------

def predict_mc_welford(model, x_test, mc_samples: int = 50, batch_size: int | None = None, training: bool = False):
    """Monte-Carlo predictive mean/std with strict batching.

    Why this exists:
      Calling `model(x_test)` on the full validation/train tensor can create an
      enormous implicit batch (e.g., 25k samples) which spikes memory.
      During GA optimization this is repeated thousands of times and can lead
      to host OOM (and/or GPU OOM) even with small models.
    """
    if model is None:
        raise ValueError("Model not built.")

    x_test = np.asarray(x_test)
    n = int(x_test.shape[0])
    if n == 0:
        return [], []

    # Default to a conservative inference batch to avoid memory spikes.
    if batch_size is None or batch_size <= 0:
        batch_size = 512

    # Probe output structure/dims.
    sample = model(x_test[:1], training=training)
    sample = list(sample) if isinstance(sample, (list, tuple)) else [sample]
    heads = len(sample)
    ds = [int(s.shape[-1]) if getattr(s, "shape", None) is not None else 1 for s in sample]

    means = [np.zeros((n, d), dtype=np.float32) for d in ds]
    m2 = [np.zeros((n, d), dtype=np.float32) for d in ds]

    mc_samples = int(mc_samples)
    if mc_samples < 1:
        return means, [np.zeros_like(m) for m in means]

    # Welford update must use the SAME sample count for every slice.
    # A previous implementation incremented a global count per-batch, which biased
    # later slices toward zero and produced wildly wrong post-fit metrics.
    for s in range(mc_samples):
        k = float(s + 1)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            xb = x_test[start:end]
            preds = model(xb, training=training)
            preds = list(preds) if isinstance(preds, (list, tuple)) else [preds]
            for h in range(heads):
                arr = preds[h].numpy()
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, -1)
                delta = arr - means[h][start:end]
                means[h][start:end] += delta / k
                delta2 = arr - means[h][start:end]
                m2[h][start:end] += delta * delta2

    stds = []
    if mc_samples < 2:
        stds = [np.zeros_like(m) for m in means]
    else:
        for h in range(heads):
            var = m2[h] / float(mc_samples - 1)
            stds.append(np.sqrt(np.maximum(var, 0.0)))
    return means, stds

__all__ = [
    'posterior_mean_field', 'prior_fn', 'build_kl_anneal_callback', 'predict_mc_welford'
]
