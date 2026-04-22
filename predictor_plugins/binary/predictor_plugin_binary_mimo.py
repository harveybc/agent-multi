#!/usr/bin/env python
"""Binary MIMO ioin plugin.

Thin subclass of the regression MIMO plugin.  Reuses the MIMO encoder
(Conv1D stack + BiLSTM + GlobalAveragePooling) as the shared trunk and
replaces the multi-horizon cross-attention decoder with a single
Dense(1, sigmoid) binary output head.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Bidirectional, LSTM,
    GlobalAveragePooling1D, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

from ..predictor_plugin_mimo import Plugin as MIMOPlugin
from ..common.positional_encoding import positional_encoding
from .binary_base import BinaryMixin, BinaryF1Score, VALID_SIGNAL_TYPES, _as_bool


class Plugin(BinaryMixin, MIMOPlugin):
    plugin_params = {
        **MIMOPlugin.plugin_params,
        "signal_type": "buy_entry",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "batch_size", "learning_rate",
        "encoder_conv_layers", "encoder_base_filters",
        "encoder_lstm_units", "decoder_dropout",
        "activation", "l2_reg", "positional_encoding", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        window_size, num_features = input_shape
        act = self.params.get("activation", "relu")
        l2_reg_v = float(self.params.get("l2_reg", 1e-7))
        dropout_rate = float(self.params.get("decoder_dropout", 0.1))

        signal_type = self.params.get("signal_type", "buy_entry")
        if signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        # -- Encoder trunk (identical to regression MIMO encoder) --
        inputs = Input(
            shape=(window_size, num_features), name="input_window"
        )
        if _as_bool(self.params.get("positional_encoding", False)):
            pe = positional_encoding(window_size, num_features)
            x = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding",
            )(inputs)
        else:
            x = inputs

        num_conv_layers = int(self.params.get("encoder_conv_layers", 2))
        base_filters = int(self.params.get("encoder_base_filters", 128))
        for i in range(num_conv_layers):
            filters = max(16, base_filters // (2 ** i))
            x = tf.keras.layers.Conv1D(
                filters=filters, kernel_size=3, padding="causal",
                activation=act, kernel_regularizer=l2(l2_reg_v),
                name=f"enc_conv_{i+1}",
            )(x)

        lstm_units = int(self.params.get("encoder_lstm_units", 128))
        x = Bidirectional(
            LSTM(lstm_units, return_sequences=True), name="enc_bilstm",
        )(x)
        z_global = GlobalAveragePooling1D(name="enc_global_avg_pool")(x)

        # -- Single binary head --
        h = Dense(
            lstm_units, activation=act,
            kernel_regularizer=l2(l2_reg_v), name="head_dense",
        )(z_global)
        h = Dropout(dropout_rate, name="head_dropout")(h)

        output = Dense(1, activation="sigmoid", name=out_name)(h)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"BinaryMIMO_{signal_type}",
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
