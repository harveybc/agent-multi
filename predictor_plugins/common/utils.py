"""General utility helpers for ioin plugins."""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import l2


def build_branch(branch_input, branch_name, num_branch_layers=2, branch_units=32,
                 activation='relu', l2_reg=1e-5):
    """Shared dense branch builder used by ANN/Transformer/CNN/LSTM plugins."""
    x = Flatten(name=f"{branch_name}_flatten")(branch_input)
    for i in range(num_branch_layers):
        x = Dense(
            branch_units,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            name=f"{branch_name}_dense_{i+1}"
        )(x)
    return x

__all__ = ['build_branch']
