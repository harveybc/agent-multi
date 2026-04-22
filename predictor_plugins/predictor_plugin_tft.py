#!/usr/bin/env python
"""Temporal Fusion Transformer (TFT) Ioin Plugin.

Implements a TFT-inspired architecture adapted for the current pipeline.
Key components:
- Gated Residual Networks (GRN)
- LSTM Encoder
- Multi-Head Attention
- Skip connections and Gating
- Positional Encoding (optional)
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Dropout, LSTM,
    LayerNormalization, MultiHeadAttention, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude
from .common.positional_encoding import positional_encoding
from .common.base import BaseDeterministicKerasPredictor


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        return default
    return bool(v)


class Plugin(BaseDeterministicKerasPredictor):
    plugin_params = {
        "tft_hidden_units": 64,
        "tft_num_heads": 4,
        "tft_dropout": 0.1,
        "tft_lstm_layers": 2,
        "learning_rate": 1e-3,
        "early_patience": 20,
        "batch_size": 64,
        "l2_reg": 1e-6,
        "activation": "elu",
        "positional_encoding": False,
        "predicted_horizons": [1],
    }

    plugin_debug_vars = [
        "tft_hidden_units", "tft_num_heads", "tft_dropout",
        "tft_lstm_layers", "learning_rate", "l2_reg",
        "positional_encoding", "predicted_horizons",
    ]

    def _glu(self, x, units, l2_val):
        """Gated Linear Unit: GLU(x) = sigma(W1 x) * (W2 x)."""
        val = Dense(units, activation=None, kernel_regularizer=l2(l2_val))(x)
        gate = Dense(units, activation='sigmoid', kernel_regularizer=l2(l2_val))(x)
        return Multiply()([val, gate])

    def _grn(self, x, units, dropout_rate, l2_val):
        """Gated Residual Network with L2 regularization."""
        skip = x
        if x.shape[-1] != units:
            skip = Dense(units, kernel_regularizer=l2(l2_val))(x)

        h = Dense(units, activation='elu', kernel_regularizer=l2(l2_val))(x)
        h = Dense(units, kernel_regularizer=l2(l2_val))(h)
        h = Dropout(dropout_rate)(h)
        h = self._glu(h, units, l2_val)

        h = Add()([skip, h])
        h = LayerNormalization()(h)
        return h

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        ph = self.params["predicted_horizons"]

        units = int(self.params["tft_hidden_units"])
        num_heads = int(self.params["tft_num_heads"])
        dropout = float(self.params["tft_dropout"])
        lstm_layers = int(self.params["tft_lstm_layers"])
        l2_val = float(self.params.get("l2_reg", 1e-6))

        inputs = Input(shape=(time_steps, channels), name="input_layer")

        # Positional encoding (optional, like MIMO)
        if _as_bool(self.params.get("positional_encoding", False), default=False):
            pe = positional_encoding(time_steps, channels)
            x = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x = inputs

        # 1. Variable Selection / Embedding — project input to hidden units
        x = self._grn(x, units, dropout, l2_val)

        # 2. LSTM Encoder — captures local temporal patterns
        for i in range(lstm_layers):
            x = LSTM(units, return_sequences=True, dropout=dropout,
                     kernel_regularizer=l2(l2_val), name=f"lstm_enc_{i+1}")(x)
            x = self._grn(x, units, dropout, l2_val)

        # 3. Temporal Fusion Decoder — self-attention for long-range dependencies
        attn_out = MultiHeadAttention(
            num_heads=num_heads, key_dim=units, name="self_mha"
        )(x, x)

        # Post-attention gating + skip connection
        h = self._grn(attn_out, units, dropout, l2_val)
        h = Add()([x, h])
        h = LayerNormalization()(h)

        # 4. Output Generation — last timestep as context for each horizon head
        context = Lambda(lambda t: t[:, -1, :])(h)

        outputs = []
        self.output_names = []

        for horizon in ph:
            head = self._grn(context, units, dropout, l2_val)
            out = Dense(1, activation="linear", name=f"output_horizon_{horizon}")(head)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"TFT_{len(ph)}H")

        optimizer = AdamW(learning_rate=float(self.params.get("learning_rate", 1e-3)))

        # MAE loss per horizon (consistent with MIMO and optimizer fitness evaluation)
        loss_dict = {nm: tf.keras.losses.MeanAbsoluteError() for nm in self.output_names}
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}

        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)

        if not self.params.get('quiet', False):
            self.model.summary(line_length=140)
