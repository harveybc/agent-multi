#!/usr/bin/env python
"""Direction TCN ioin plugin (deterministic).

Thin subclass of the regression TCN plugin for direction classification.
Replicates the dilated causal Conv1D backbone and adds a single
Dense(1, sigmoid) direction output head.

Inherits _temporal_block from the parent TCN plugin.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Conv1D, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

from ..predictor_plugin_tcn import Plugin as TCNPlugin
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES, _as_bool


class Plugin(DirectionMixin, TCNPlugin):
    plugin_params = {
        **TCNPlugin.plugin_params,
        "signal_type": "direction_long",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "tcn_filters", "tcn_kernel_size", "tcn_stack_layers",
        "tcn_dilations_per_stack", "tcn_dropout",
        "tcn_head_layers", "tcn_head_units",
        "learning_rate", "l2_reg", "positional_encoding", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        filters = int(self.params["tcn_filters"])
        kernel_size = int(self.params["tcn_kernel_size"])
        stack_layers = int(self.params["tcn_stack_layers"])
        dilations_per_stack = int(self.params["tcn_dilations_per_stack"])
        dropout = float(self.params["tcn_dropout"])
        use_batch_norm = _as_bool(
            self.params.get("tcn_use_batch_norm", False)
        )
        use_layer_norm = _as_bool(
            self.params.get("tcn_use_layer_norm", True), default=True
        )
        head_layers = int(self.params.get("tcn_head_layers", 1))
        head_units = int(self.params.get("tcn_head_units", 32))
        l2_val = float(self.params.get("l2_reg", 1e-6))
        activation = str(self.params.get("activation", "elu"))

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

        x = Conv1D(
            filters, 1, padding="same", activation=None,
            kernel_regularizer=l2(l2_val), name="input_proj",
        )(x)

        block_idx = 0
        for s in range(stack_layers):
            for d in range(dilations_per_stack):
                dilation_rate = 2 ** d
                x = self._temporal_block(
                    x, filters, kernel_size, dilation_rate,
                    dropout, l2_val, activation,
                    use_batch_norm, use_layer_norm,
                    block_name=f"tcn_s{s+1}_d{dilation_rate}_{block_idx}",
                )
                block_idx += 1

        context = Lambda(
            lambda t: t[:, -1, :], name="last_timestep"
        )(x)

        h = context
        for hl in range(head_layers):
            h = Dense(
                head_units, activation=activation,
                kernel_regularizer=l2(l2_val),
                name=f"head_dense_{hl+1}",
            )(h)
            h = Dropout(dropout, name=f"head_drop_{hl+1}")(h)

        output = Dense(1, activation="sigmoid", name=out_name)(h)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionTCN_{signal_type}",
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=float(self.params.get("learning_rate", 1e-3))
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
