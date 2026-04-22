"""Common loss and metric functions shared across ioin plugins.

All functions here are pure / stateless and TensorFlow-friendly so they can be
safely imported inside model building contexts. Keep signatures stable to avoid
serialization issues.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.losses import Huber

# Reuse one loss instance to avoid per-step object churn during long trainings.
_HUBER = Huber()
_EPS = tf.constant(1e-8, dtype=tf.float32)

# --- Metrics ---

def mae_magnitude(y_true, y_pred):
    """Mean Absolute Error on first column (magnitude).
    Expands y_true to two columns if it is shape (N,) or (N,1) to preserve
    backward compatibility with existing plugin logic.
    """
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """R^2 on first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    ss_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    ss_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# --- Auxiliary kernels ---

def _gaussian_kernel(x, y, sigma):
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    dist = tf.reduce_sum(tf.square(x - y), axis=-1)
    return tf.exp(-dist / (2.0 * sigma ** 2))

# --- MMD ---

def compute_mmd(x, y, sigma=1.0, sample_size=256):
    """Compute Maximum Mean Discrepancy with optional subsampling."""
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
    x_sample = tf.gather(x, idx)
    y_sample = tf.gather(y, idx)
    k_xx = _gaussian_kernel(x_sample, x_sample, sigma)
    k_yy = _gaussian_kernel(y_sample, y_sample, sigma)
    k_xy = _gaussian_kernel(x_sample, y_sample, sigma)
    return tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)


def _to_1d(t):
    t = tf.cast(tf.reshape(t, [-1]), tf.float32)
    return t


def _masked_mean(x, mask):
    x = tf.cast(x, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(x * mask) / (tf.reduce_sum(mask) + _EPS)


def _resolve_external_mask(is_gap_mask, n):
    if is_gap_mask is None:
        return tf.ones([n], dtype=tf.float32)
    m = tf.cast(is_gap_mask, tf.float32)
    if m.shape.rank == 0:
        return tf.fill([n], m)
    m = tf.reshape(m, [-1])
    size = tf.shape(m)[0]
    # XLA-friendly shape normalization with no tf.cond branching.
    pad_len = tf.maximum(n - size, 0)
    m = tf.pad(m, [[0, pad_len]], constant_values=1.0)
    return m[:n]


def _extract_mag_and_mask(y_true, y_pred, is_gap_mask=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_true.shape.rank == 1:
        mag_true = tf.reshape(y_true, [-1])
        mask = tf.ones_like(mag_true)
    elif y_true.shape.rank == 2:
        mag_true = tf.reshape(y_true[:, 0], [-1])
        if y_true.shape[1] is not None and y_true.shape[1] >= 2:
            mask = tf.cast(tf.reshape(y_true[:, 1], [-1]), tf.float32)
        else:
            mask = tf.ones_like(mag_true)
    else:
        mag_true = _to_1d(y_true)
        mask = tf.ones_like(mag_true)

    if y_pred.shape.rank == 1:
        mag_pred = tf.reshape(y_pred, [-1])
    elif y_pred.shape.rank == 2:
        mag_pred = tf.reshape(y_pred[:, 0], [-1])
    else:
        mag_pred = _to_1d(y_pred)

    n = tf.minimum(tf.shape(mag_true)[0], tf.shape(mag_pred)[0])
    mag_true = mag_true[:n]
    mag_pred = mag_pred[:n]
    mask = tf.cast(mask[:n], tf.float32)

    ext_mask = _resolve_external_mask(is_gap_mask, n)
    mask = tf.clip_by_value(mask * ext_mask, 0.0, 1.0)
    return mag_true, mag_pred, mask


def _huber_elementwise(y_true, y_pred, delta=1.0):
    err = tf.abs(y_true - y_pred)
    d = tf.cast(delta, tf.float32)
    quadratic = tf.minimum(err, d)
    linear = err - quadratic
    return 0.5 * tf.square(quadratic) + d * linear


def _weighted_std(x, mask):
    mu = _masked_mean(x, mask)
    var = _masked_mean(tf.square(x - mu), mask)
    return tf.sqrt(var + _EPS)


def _trend_sigma_loss(y_true, y_pred, trend_sigma_lambda=0.1, is_gap_mask=None):
    mag_true, mag_pred, mask = _extract_mag_and_mask(y_true, y_pred, is_gap_mask=is_gap_mask)

    huber_vals = _huber_elementwise(mag_true, mag_pred, delta=1.0)
    base_loss = _masked_mean(huber_vals, mask)

    dy_true = mag_true[-1] - mag_true[0]
    dy_pred = mag_pred[-1] - mag_pred[0]
    same_dir = tf.equal(tf.sign(dy_true), tf.sign(dy_pred))
    w_dir = tf.where(same_dir, 1.0, 2.5)

    sigma_true = _weighted_std(mag_true, mask)
    sigma_pred = _weighted_std(mag_pred, mask)
    sigma_penalty = tf.maximum(0.0, sigma_true - sigma_pred)

    batch_mask = tf.reduce_mean(mask)
    return ((base_loss * w_dir) + tf.cast(trend_sigma_lambda, tf.float32) * sigma_penalty) * batch_mask


def _pearson_structural_loss(y_true, y_pred, pearson_alpha=0.5, is_gap_mask=None):
    mag_true, mag_pred, mask = _extract_mag_and_mask(y_true, y_pred, is_gap_mask=is_gap_mask)

    mae = _masked_mean(tf.abs(mag_true - mag_pred), mask)

    mu_true = _masked_mean(mag_true, mask)
    mu_pred = _masked_mean(mag_pred, mask)
    ct = mag_true - mu_true
    cp = mag_pred - mu_pred
    cov = _masked_mean(ct * cp, mask)
    std_true = _weighted_std(mag_true, mask)
    std_pred = _weighted_std(mag_pred, mask)
    corr = cov / (std_true * std_pred + _EPS)
    corr = tf.clip_by_value(corr, -1.0, 1.0)

    batch_mask = tf.reduce_mean(mask)
    return (mae + tf.cast(pearson_alpha, tf.float32) * (1.0 - corr)) * batch_mask


def _combined_diff_loss(y_true, y_pred, diff_weight=1.0, is_gap_mask=None):
    """Huber on levels + weighted Huber on first temporal differences + variance penalty.

    With shuffle=False, consecutive batch elements are temporally adjacent, so
    first-differences Δy = y[t+1] - y[t] capture temporal dynamics.  A trivial
    (constant) ioin has Δŷ ≡ 0, which is heavily penalised when Δy ≠ 0.

    Parameters
    ----------
    diff_weight : float
        Multiplier on the first-difference Huber component (default 1.0).
    """
    mag_true, mag_pred, mask = _extract_mag_and_mask(y_true, y_pred, is_gap_mask=is_gap_mask)

    # --- Level component (standard Huber) ---
    level_loss = _masked_mean(_huber_elementwise(mag_true, mag_pred, delta=1.0), mask)

    # --- First-difference component ---
    dt = mag_true[1:] - mag_true[:-1]
    dp = mag_pred[1:] - mag_pred[:-1]
    mask_diff = mask[1:] * mask[:-1]  # valid only when both neighbours are valid
    diff_loss = _masked_mean(_huber_elementwise(dt, dp, delta=1.0), mask_diff)

    # --- Variance-ratio penalty: penalise under-variation ---
    sigma_true = _weighted_std(mag_true, mask)
    sigma_pred = _weighted_std(mag_pred, mask)
    var_ratio = sigma_pred / (sigma_true + _EPS)
    var_penalty = tf.maximum(0.0, 1.0 - var_ratio)  # 0 when pred var >= true var

    batch_mask = tf.reduce_mean(mask)
    return (level_loss
            + tf.cast(diff_weight, tf.float32) * diff_loss
            + 0.1 * var_penalty) * batch_mask


def _softmin3(a, b, c, gamma):
    vals = tf.stack([a, b, c], axis=0)
    vmin = tf.reduce_min(vals)
    return vmin - gamma * tf.math.log(tf.reduce_sum(tf.exp(-(vals - vmin) / gamma)) + _EPS)


def _soft_dtw_loss(y_true, y_pred, soft_dtw_gamma=0.1, is_gap_mask=None):
    mag_true, mag_pred, mask = _extract_mag_and_mask(y_true, y_pred, is_gap_mask=is_gap_mask)

    # XLA-safe formulation: keep full sequence length and mask costs instead of
    # branching on empty tensors with tf.cond/tf.boolean_mask.
    x = mag_true
    y = mag_pred
    n = tf.shape(x)[0]

    gamma = tf.maximum(tf.cast(soft_dtw_gamma, tf.float32), _EPS)
    x_col = tf.expand_dims(x, axis=1)
    y_row = tf.expand_dims(y, axis=0)
    dmat = tf.square(x_col - y_row)

    # Keep alignment focused on valid (non-gap) positions.
    m_col = tf.expand_dims(tf.cast(mask, tf.float32), axis=1)
    m_row = tf.expand_dims(tf.cast(mask, tf.float32), axis=0)
    m2 = m_col * m_row
    large = tf.constant(1e6, dtype=tf.float32)
    dmat = dmat * m2 + (1.0 - m2) * large

    inf = tf.constant(1e12, dtype=tf.float32)
    r = tf.fill([n + 1, n + 1], inf)
    r = tf.tensor_scatter_nd_update(r, [[0, 0]], [0.0])

    def outer_cond(i, rmat):
        return tf.less_equal(i, n)

    def outer_body(i, rmat):
        def inner_cond(j, rinner):
            return tf.less_equal(j, n)

        def inner_body(j, rinner):
            a = rinner[i - 1, j]
            b = rinner[i, j - 1]
            c = rinner[i - 1, j - 1]
            sm = _softmin3(a, b, c, gamma)
            val = dmat[i - 1, j - 1] + sm
            rnext = tf.tensor_scatter_nd_update(rinner, [[i, j]], [val])
            return j + 1, rnext

        _, rnew = tf.while_loop(
            inner_cond,
            inner_body,
            loop_vars=[tf.constant(1, dtype=tf.int32), rmat],
            parallel_iterations=1,
            maximum_iterations=n,
        )
        return i + 1, rnew

    _, r = tf.while_loop(
        outer_cond,
        outer_body,
        loop_vars=[tf.constant(1, dtype=tf.int32), r],
        parallel_iterations=1,
        maximum_iterations=n,
    )

    # If a batch is all masked-out, zero out the contribution.
    batch_mask = tf.cast(tf.reduce_any(tf.greater(mask, 0.0)), tf.float32)
    return r[n, n] * batch_mask


def _morphological_loss_dispatch(
    y_true,
    y_pred,
    *,
    loss_type="mae",
    trend_sigma_lambda=0.1,
    pearson_alpha=0.5,
    soft_dtw_gamma=0.1,
    diff_weight=1.0,
    is_gap_mask=None,
):
    lt = str(loss_type).strip().lower()
    if lt == "trend_sigma":
        return _trend_sigma_loss(
            y_true,
            y_pred,
            trend_sigma_lambda=trend_sigma_lambda,
            is_gap_mask=is_gap_mask,
        )
    if lt == "pearson_structural":
        return _pearson_structural_loss(
            y_true,
            y_pred,
            pearson_alpha=pearson_alpha,
            is_gap_mask=is_gap_mask,
        )
    if lt == "soft_dtw":
        return _soft_dtw_loss(
            y_true,
            y_pred,
            soft_dtw_gamma=soft_dtw_gamma,
            is_gap_mask=is_gap_mask,
        )
    if lt == "combined_diff":
        return _combined_diff_loss(
            y_true,
            y_pred,
            diff_weight=diff_weight,
            is_gap_mask=is_gap_mask,
        )
    # Backward-compatible fallback: preserve original MAE behavior used by TCN.
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))


def configurable_time_series_loss(
    y_true,
    y_pred,
    *,
    loss_type="mae",
    trend_sigma_lambda=0.1,
    pearson_alpha=0.5,
    soft_dtw_gamma=0.1,
    diff_weight=1.0,
    morphology_batch_size=32,
    is_gap_mask=None,
):
    """Public loss entrypoint with backward-compatible default.

    Supported values for ``loss_type``:
      - ``mae`` (default, preserves previous TCN behavior)
      - ``trend_sigma``
      - ``pearson_structural``
      - ``soft_dtw``
      - ``combined_diff``
    """
    return _morphological_loss_dispatch(
        y_true,
        y_pred,
        loss_type=loss_type,
        trend_sigma_lambda=trend_sigma_lambda,
        pearson_alpha=pearson_alpha,
        soft_dtw_gamma=soft_dtw_gamma,
        diff_weight=diff_weight,
        is_gap_mask=is_gap_mask,
    )

# --- Composite Loss Variants ---

def composite_loss_basic(y_true, y_pred, mmd_lambda=0.0, sigma=1.0):
        """Composite loss = (Huber / incentive) + mmd_lambda * MMD.

        Incentive logic (applied ONLY on the Huber term):
            predicted_error = mean(|y_true - y_pred|)
            naive_error     = mean(|y_true|)           (error of predicting 0 returns)

        If predicted_error > naive_error: incentive = 1 (no change).
        If predicted_error <= naive_error: incentive follows a linear scale
            predicted_error = naive_error  -> incentive = 1
            predicted_error = 0            -> incentive = 10
            Linear interpolation in between.

        This rewards models outperforming the naive zero-return ioin by
        shrinking the effective Huber loss (division by incentive in [1,10]).
        Edge case: if naive_error == 0 (all-zero targets), incentive = 10.
        """
        if y_true.shape.ndims == 1 or (y_true.shape.ndims == 2 and y_true.shape[1] == 1):
                y_true = tf.reshape(y_true, [-1, 1])

        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]

        # Base Huber loss on magnitude.
        huber_loss_val = _HUBER(mag_true, mag_pred)

        # Predicted vs naive errors (scalar tensors)
        eps = tf.keras.backend.epsilon()
        predicted_error = tf.reduce_mean(tf.abs(mag_true - mag_pred))
        naive_error = tf.reduce_mean(tf.abs(mag_true))

        # Condition where incentive applies (better or equal to naive baseline)
        cond = tf.less_equal(predicted_error, naive_error)

        # Linear incentive: 1 at predicted_error==naive_error, 10 at 0 error.
        # incentive = 10 - 9 * (pred_err / naive_err)
        # Safe handling when naive_error ~ 0: force incentive=10.
        ratio = predicted_error / (naive_error + eps)
        ratio = tf.clip_by_value(ratio, 0.0, 1.0)
        linear_incentive = 1000.0 - 999.0 * ratio
        incentive = tf.where(cond, linear_incentive, 1.0)

        # If predicted_error is (near) zero, override to max incentive (all targets zero case)
        incentive = tf.where(tf.less_equal(predicted_error, eps), 1000.0, incentive)

        # Apply incentive only to huber component.
        adjusted_huber = huber_loss_val / incentive

        if mmd_lambda != 0.0:
                mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
        else:
                mmd_loss_val = 0.0

        return adjusted_huber + mmd_lambda * mmd_loss_val

# Legacy signature adapter for multi-head plugins

def composite_loss_multihead(y_true, y_pred, head_index, mmd_lambda, sigma,
                             p, i, d,
                             list_last_signed_error,
                             list_last_stddev,
                             list_last_mmd,
                             list_local_feedback,
                             loss_type="trend_sigma",
                             trend_sigma_lambda=0.1,
                             pearson_alpha=0.5,
                             soft_dtw_gamma=0.1,
                             diff_weight=1.0,
                             morphology_batch_size=32,
                             is_gap_mask=None):
    """Adapter wrapping composite_loss_basic keeping legacy callable shape.
    Currently ignores control feedback lists (placeholders) but keeps them
    for interface compatibility.
    """
    return _morphological_loss_dispatch(
        y_true,
        y_pred,
        loss_type=loss_type,
        trend_sigma_lambda=trend_sigma_lambda,
        pearson_alpha=pearson_alpha,
        soft_dtw_gamma=soft_dtw_gamma,
        diff_weight=diff_weight,
        is_gap_mask=is_gap_mask,
    )

def composite_loss_noreturns(y_true, y_pred, head_index, mmd_lambda, sigma,
                             p, i, d,
                             list_last_signed_error,
                             list_last_stddev,
                             list_last_mmd,
                             list_local_feedback,
                             loss_type="trend_sigma",
                             trend_sigma_lambda=0.1,
                             pearson_alpha=0.5,
                             soft_dtw_gamma=0.1,
                             diff_weight=1.0,
                             morphology_batch_size=32,
                             is_gap_mask=None):
    """Adapter wrapping composite_loss_basic keeping legacy callable shape.
    Currently ignores control feedback lists (placeholders) but keeps them
    for interface compatibility.
    """
    return _morphological_loss_dispatch(
        y_true,
        y_pred,
        loss_type=loss_type,
        trend_sigma_lambda=trend_sigma_lambda,
        pearson_alpha=pearson_alpha,
        soft_dtw_gamma=soft_dtw_gamma,
        diff_weight=diff_weight,
        is_gap_mask=is_gap_mask,
    )


def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

__all__ = [
    'mae_magnitude', 'r2_metric', 'compute_mmd',
    'composite_loss_basic', 'composite_loss_multihead',
    'composite_loss_noreturns',
    '_trend_sigma_loss', '_pearson_structural_loss', '_soft_dtw_loss',
    '_combined_diff_loss',
    'configurable_time_series_loss',
    'random_normal_initializer_44'
]
