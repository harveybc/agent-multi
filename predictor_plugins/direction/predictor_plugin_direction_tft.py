#!/usr/bin/env python
"""Direction TFT ioin plugin (deterministic).

Thin subclass of the regression TFT plugin for direction classification.
Replicates the GRN-based encoder (LSTM + MultiHeadAttention) and adds
a single Dense(1, sigmoid) direction output head.

Inherits _glu and _grn methods from the parent TFT plugin.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, LSTM,
    LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

from ..predictor_plugin_tft import Plugin as TFTPlugin
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES, _as_bool


class Plugin(DirectionMixin, TFTPlugin):
    plugin_params = {
        **TFTPlugin.plugin_params,
        "signal_type": "direction_long",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "tft_hidden_units", "tft_num_heads", "tft_dropout",
        "tft_lstm_layers", "learning_rate", "l2_reg",
        "positional_encoding", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        units = int(self.params["tft_hidden_units"])
        num_heads = int(self.params["tft_num_heads"])
        dropout = float(self.params["tft_dropout"])
        lstm_layers = int(self.params["tft_lstm_layers"])
        l2_val = float(self.params.get("l2_reg", 1e-6))

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(time_steps, channels), name="input_layer")
        if _as_bool(self.params.get("positional_encoding", False)):
            pe = positional_encoding(time_steps, channels)
            x = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            x = inputs

        x = self._grn(x, units, dropout, l2_val)
        for i in range(lstm_layers):
            x = LSTM(
                units, return_sequences=True, dropout=dropout,
                kernel_regularizer=l2(l2_val), name=f"lstm_enc_{i+1}",
            )(x)
            x = self._grn(x, units, dropout, l2_val)

        attn_out = MultiHeadAttention(
            num_heads=num_heads, key_dim=units, name="self_mha",
        )(x, x)
        h = self._grn(attn_out, units, dropout, l2_val)
        h = Add()([x, h])
        h = LayerNormalization()(h)

        context = Lambda(lambda t: t[:, -1, :])(h)

        head = self._grn(context, units, dropout, l2_val)
        output = Dense(1, activation="sigmoid", name=out_name)(head)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionTFT_{signal_type}",
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=float(self.params.get("learning_rate", 1e-3))
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        if not self.params.get("quiet", False):
            self.model.summary(line_length=140)
