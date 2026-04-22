#!/usr/bin/env python
"""Binary CNN ioin plugin.

Thin subclass of the regression CNN plugin.  Replicates the Conv1D trunk
and replaces the per-horizon output heads with a single sigmoid head.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Bidirectional, LSTM, Conv1D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from ..predictor_plugin_cnn import Plugin as CNNPlugin
from ..common.positional_encoding import positional_encoding
from .binary_base import BinaryMixin, BinaryF1Score, VALID_SIGNAL_TYPES


class Plugin(BinaryMixin, CNNPlugin):
    plugin_params = {
        **CNNPlugin.plugin_params,
        "signal_type": "buy_entry",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "batch_size", "branch_units", "merged_units", "learning_rate",
        "l2_reg", "signal_type", "positional_encoding",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        w, c = input_shape
        act = self.params.get("activation", "relu")
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        initial_layer_size = self.params.get("initial_layer_size", 128)
        layer_size_divisor = self.params.get("layer_size_divisor", 2)
        intermediate_layers = int(self.params.get("intermediate_layers", 1))
        head_layers_count = int(self.params.get("head_layers", 1))

        signal_type = self.params.get("signal_type", "buy_entry")
        if signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        # -- Shared Conv1D trunk (identical to regression CNN) --
        inputs = Input(shape=(w, c), name="input_layer")
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(w, c)
            x_in = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            x_in = inputs

        x = x_in
        num_layers = max(1, intermediate_layers)
        sizes = [initial_layer_size] + [
            max(8, initial_layer_size // (layer_size_divisor ** i))
            for i in range(1, num_layers)
        ]
        for i, filters_i in enumerate(sizes):
            x = Conv1D(
                filters=filters_i, kernel_size=3, strides=2,
                padding="causal", activation=act,
                kernel_regularizer=l2(l2_reg_v), name=f"conv_{i+1}",
            )(x)

        # -- Single binary head (Conv1D stack → BiLSTM → sigmoid) --
        last_root_filters = sizes[-1]
        head_num = max(1, head_layers_count)
        base_head_filters = max(8, last_root_filters // 2)
        head_sizes = [base_head_filters] + [
            max(8, base_head_filters // (layer_size_divisor ** i))
            for i in range(1, head_num)
        ]
        h = x
        for j, f_j in enumerate(head_sizes):
            h = Conv1D(
                filters=f_j, kernel_size=3, strides=2, padding="same",
                activation=act, kernel_regularizer=l2(l2_reg_v),
                name=f"head_conv{j+1}",
            )(h)

        last_head_filters = head_sizes[-1]
        lstm_total_units = max(8, last_head_filters // 2)
        h = Bidirectional(
            LSTM(max(1, lstm_total_units // 2), return_sequences=False),
            name="bilstm",
        )(h)

        output = Dense(1, activation="sigmoid", name=out_name)(h)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output], name=f"BinaryCNN_{signal_type}"
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=self.params.get("learning_rate", 1e-3)
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
