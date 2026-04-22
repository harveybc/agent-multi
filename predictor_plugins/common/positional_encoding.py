"""Positional encoding utility shared across ioin plugins.

Implements a deterministic sinusoidal positional encoding identical in shape to
the model inputs (window, channels) and broadcastable over the batch axis.
Returned tensor shape: (1, window, channels) so it can be added to an input
tensor of shape (batch, window, channels).

Reference: Vaswani et al. (2017) Attention Is All You Need.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


def positional_encoding(window: int, channels: int) -> tf.Tensor:
    """Create sinusoidal positional encoding.

    Args:
        window: Number of time steps.
        channels: Feature dimension (d_model).
    Returns:
        Tensor of shape (1, window, channels) dtype float32.
    """
    position = np.arange(window)[:, np.newaxis]  # (window, 1)
    dims = np.arange(channels)[np.newaxis, :]    # (1, channels)
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(channels))
    angle_rads = position * angle_rates  # (window, channels)
    # apply sin to even indices, cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pe = angle_rads[np.newaxis, ...]  # (1, window, channels)
    return tf.cast(pe, tf.float32)

__all__ = ["positional_encoding"]