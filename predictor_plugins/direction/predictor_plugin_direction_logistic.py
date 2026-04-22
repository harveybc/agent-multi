#!/usr/bin/env python
"""Direction Logistic Regression baseline ioin plugin.

A minimal single-layer Dense(1, sigmoid) model for direction classification.
Serves as a baseline to measure whether deeper/complex models add any
value on directional prediction.
"""
from __future__ import annotations

from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from ..common.base import BaseDeterministicKerasPredictor
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES


class Plugin(DirectionMixin, BaseDeterministicKerasPredictor):
    plugin_params = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "signal_type": "direction_long",
        "window_size": 74,
        "early_patience": 15,
        "start_from_epoch": 5,
        "positional_encoding": False,
        "l2_reg": 0.0,
    }

    plugin_debug_vars = [
        "batch_size", "learning_rate", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        window, channels = input_shape

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(window, channels), name="input_layer")

        if self.params.get("positional_encoding", False):
            pe = positional_encoding(window, channels)
            enc = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            enc = inputs

        x = Flatten(name="flatten_inputs")(enc)

        l2_val = self.params.get("l2_reg", 0.0)
        reg = None
        if l2_val > 0:
            from tensorflow.keras.regularizers import l2 as l2_reg
            reg = l2_reg(l2_val)

        output = Dense(
            1, activation="sigmoid", name=out_name,
            kernel_regularizer=reg,
        )(x)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionLogistic_{signal_type}",
        )
        self.model.compile(
            optimizer=Adam(
                learning_rate=self.params.get("learning_rate", 1e-3),
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=120)
