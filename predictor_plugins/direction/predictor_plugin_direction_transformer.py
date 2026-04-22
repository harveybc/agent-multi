#!/usr/bin/env python
"""Direction Transformer ioin plugin.

Thin subclass of the regression Transformer plugin for direction
classification.  Replicates the shared trunk (MultiHeadAttention + Conv1D)
and adds a single Dense(1, sigmoid) direction output head.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Bidirectional, LSTM, Conv1D,
    MultiHeadAttention, LayerNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from ..predictor_plugin_transformer import Plugin as TransformerPlugin
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES


class Plugin(DirectionMixin, TransformerPlugin):
    plugin_params = {
        **TransformerPlugin.plugin_params,
        "signal_type": "direction_long",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "batch_size", "merged_units", "branch_units", "learning_rate",
        "l2_reg", "num_attention_heads", "signal_type",
        "positional_encoding",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        w, c = input_shape
        act = self.params.get("activation", "relu")
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        merged_units = self.params.get("merged_units", 128)
        branch_units = self.params.get("branch_units", 64)
        lstm_units = max(8, branch_units // 2)

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(w, c), name="input_layer")
        if self.params.get("positional_encoding", True):
            pe = positional_encoding(w, c)
            x = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            x = inputs

        heads = self.params.get("num_attention_heads", 2)
        key_dim = max(1, c // heads)
        attn = MultiHeadAttention(
            num_heads=heads, key_dim=key_dim,
            kernel_regularizer=l2(l2_reg_v), name="mh_attention",
        )(x, x)
        x = Add(name="res_attn")([x, attn])
        x = LayerNormalization(name="attn_ln")(x)
        x = Conv1D(
            filters=merged_units, kernel_size=3, strides=2, padding="same",
            activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_1",
        )(x)
        x = Conv1D(
            filters=branch_units, kernel_size=3, strides=2, padding="same",
            activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_2",
        )(x)

        h = Conv1D(
            filters=branch_units, kernel_size=3, strides=2, padding="same",
            activation=act, kernel_regularizer=l2(l2_reg_v),
            name="head_conv1",
        )(x)
        h = Conv1D(
            filters=lstm_units, kernel_size=3, strides=2, padding="same",
            activation=act, kernel_regularizer=l2(l2_reg_v),
            name="head_conv2",
        )(h)
        h = Bidirectional(
            LSTM(lstm_units, return_sequences=False), name="bilstm",
        )(h)

        output = Dense(1, activation="sigmoid", name=out_name)(h)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionTransformer_{signal_type}",
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=self.params.get("learning_rate", 1e-3)
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
