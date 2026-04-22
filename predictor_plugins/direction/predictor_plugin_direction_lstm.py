#!/usr/bin/env python
"""Direction LSTM ioin plugin.

Inherits from DirectionMixin + BaseBayesianKerasPredictor and replicates
the LSTM trunk architecture for direction classification:

    Positional Encoding → MultiHeadAttention + Residual + LayerNorm
    → AveragePooling → BiLSTM × 2 → AveragePooling → trunk

Then adds a single Dense(1, sigmoid) direction output head.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Bidirectional, LSTM, Conv1D,
    MultiHeadAttention, LayerNormalization, AveragePooling1D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from ..common.base import BaseBayesianKerasPredictor
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES


class Plugin(DirectionMixin, BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "activation": "relu",
        "l2_reg": 1e-5,
        "early_patience": 10,
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "mc_samples": 50,
        "signal_type": "direction_long",
    }

    plugin_debug_vars = [
        "batch_size", "learning_rate", "l2_reg",
        "early_patience", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        w, c = input_shape
        act = self.params.get("activation", "relu")
        l2_reg_v = float(self.params.get("l2_reg", 1e-5))
        merged_units = int(self.params.get("initial_layer_size", 128))
        divisor = int(self.params.get("layer_size_divisor", 2))
        branch_units = max(8, merged_units // divisor)
        lstm_units = max(8, branch_units // divisor)

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(w, c), name="input_layer")
        pe = positional_encoding(w, c)
        x = Lambda(
            lambda t, pe=pe: t + pe, name="add_positional_encoding"
        )(inputs)

        attn_key_dim = max(1, c // 2)
        attn_out = MultiHeadAttention(
            num_heads=2, key_dim=attn_key_dim,
            kernel_regularizer=l2(l2_reg_v), name="mha_1",
        )(x, x)
        x = Add(name="res_attn")([x, attn_out])
        x = LayerNormalization(name="attn_ln")(x)
        x = AveragePooling1D(
            pool_size=3, strides=2, padding="same", name="avg_pool_1"
        )(x)

        x = Bidirectional(
            LSTM(lstm_units, return_sequences=True,
                 kernel_regularizer=l2(l2_reg_v)),
            name="bilstm_1",
        )(x)
        x = Bidirectional(
            LSTM(lstm_units, return_sequences=True,
                 kernel_regularizer=l2(l2_reg_v)),
            name="bilstm_2",
        )(x)
        x = AveragePooling1D(
            pool_size=3, strides=2, padding="same", name="avg_pool_2"
        )(x)

        h = Conv1D(
            filters=branch_units, kernel_size=3, strides=2, padding="same",
            kernel_regularizer=l2(l2_reg_v), name="head_conv1",
        )(x)
        h = Conv1D(
            filters=lstm_units, kernel_size=3, strides=2, padding="same",
            kernel_regularizer=l2(l2_reg_v), name="head_conv2",
        )(h)
        h = Bidirectional(
            LSTM(lstm_units, return_sequences=False), name="head_bilstm",
        )(h)

        output = Dense(1, activation="sigmoid", name=out_name)(h)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionLSTM_{signal_type}",
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=self.params.get("learning_rate", 1e-3)
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
