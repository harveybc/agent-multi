#!/usr/bin/env python
"""Temporal Convolutional Network (TCN) Ioin Plugin.

Implements a TCN architecture adapted for the current pipeline.
Key components:
- Dilated causal Conv1D layers with exponentially increasing dilation
- Residual connections per temporal block
- Optional Layer Normalization or Batch Normalization
- Spatial Dropout for regularization
- Per-horizon output heads with configurable depth
- Positional Encoding (optional)

Reference: Bai, Kolter & Koltun (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Conv1D, Dropout, SpatialDropout1D,
    LayerNormalization, BatchNormalization, GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude, configurable_time_series_loss
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
        "tcn_filters": 64,
        "tcn_kernel_size": 3,
        "tcn_stack_layers": 3,
        "tcn_dilations_per_stack": 4,
        "tcn_dropout": 0.1,
        "tcn_use_batch_norm": False,
        "tcn_use_layer_norm": True,
        "tcn_head_layers": 1,
        "tcn_head_units": 32,
        "learning_rate": 1e-3,
        "early_patience": 80,
        "batch_size": 32,
        "l2_reg": 1e-6,
        "activation": "elu",
        "positional_encoding": False,
        # Backward-compatible default: keep historical MAE unless overridden in config.
        "loss_type": "mae",
        "trend_sigma_lambda": 0.1,
        "pearson_alpha": 0.5,
        "soft_dtw_gamma": 0.1,
        "diff_weight": 1.0,
        "morphology_batch_size": 32,
        "predicted_horizons": [1],
    }

    plugin_debug_vars = [
        "tcn_filters", "tcn_kernel_size", "tcn_stack_layers",
        "tcn_dilations_per_stack", "tcn_dropout",
        "tcn_use_batch_norm", "tcn_use_layer_norm",
        "tcn_head_layers", "tcn_head_units",
        "learning_rate", "l2_reg",
        "positional_encoding", "predicted_horizons",
        "loss_type", "trend_sigma_lambda", "pearson_alpha", "soft_dtw_gamma", "morphology_batch_size",
    ]

    def _temporal_block(self, x, filters, kernel_size, dilation_rate,
                        dropout_rate, l2_val, activation, use_batch_norm,
                        use_layer_norm, block_name):
        """Single TCN residual block with dilated causal convolution.

        Architecture per block:
            x -> CausalConv1D -> Norm -> Activation -> SpatialDropout
              -> CausalConv1D -> Norm -> Activation -> SpatialDropout
              -> Add(residual) -> output
        """
        # Causal padding: pad (kernel_size - 1) * dilation_rate on the left
        pad_size = (kernel_size - 1) * dilation_rate

        # First dilated causal conv
        h = Lambda(lambda t, p=pad_size: tf.pad(t, [[0, 0], [p, 0], [0, 0]]),
                   name=f"{block_name}_pad1")(x)
        h = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation=None,
                   kernel_regularizer=l2(l2_val),
                   name=f"{block_name}_conv1")(h)
        if use_batch_norm:
            h = BatchNormalization(name=f"{block_name}_bn1")(h)
        if use_layer_norm:
            h = LayerNormalization(name=f"{block_name}_ln1")(h)
        h = tf.keras.layers.Activation(activation, name=f"{block_name}_act1")(h)
        h = SpatialDropout1D(dropout_rate, name=f"{block_name}_drop1")(h)

        # Second dilated causal conv
        h = Lambda(lambda t, p=pad_size: tf.pad(t, [[0, 0], [p, 0], [0, 0]]),
                   name=f"{block_name}_pad2")(h)
        h = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation=None,
                   kernel_regularizer=l2(l2_val),
                   name=f"{block_name}_conv2")(h)
        if use_batch_norm:
            h = BatchNormalization(name=f"{block_name}_bn2")(h)
        if use_layer_norm:
            h = LayerNormalization(name=f"{block_name}_ln2")(h)
        h = tf.keras.layers.Activation(activation, name=f"{block_name}_act2")(h)
        h = SpatialDropout1D(dropout_rate, name=f"{block_name}_drop2")(h)

        # Residual connection: match dimensions if needed
        if x.shape[-1] != filters:
            skip = Conv1D(filters, 1, padding='same',
                          kernel_regularizer=l2(l2_val),
                          name=f"{block_name}_skip_proj")(x)
        else:
            skip = x

        out = Add(name=f"{block_name}_residual")([skip, h])
        return out

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        ph = self.params["predicted_horizons"]

        filters = int(self.params["tcn_filters"])
        kernel_size = int(self.params["tcn_kernel_size"])
        stack_layers = int(self.params["tcn_stack_layers"])
        dilations_per_stack = int(self.params["tcn_dilations_per_stack"])
        dropout = float(self.params["tcn_dropout"])
        use_batch_norm = _as_bool(self.params.get("tcn_use_batch_norm", False), default=False)
        use_layer_norm = _as_bool(self.params.get("tcn_use_layer_norm", True), default=True)
        head_layers = int(self.params.get("tcn_head_layers", 1))
        head_units = int(self.params.get("tcn_head_units", 32))
        l2_val = float(self.params.get("l2_reg", 1e-6))
        activation = str(self.params.get("activation", "elu"))

        inputs = Input(shape=(time_steps, channels), name="input_layer")

        # Positional encoding (optional)
        if _as_bool(self.params.get("positional_encoding", False), default=False):
            pe = positional_encoding(time_steps, channels)
            x = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x = inputs

        # 1. Input projection to match filter dimension
        x = Conv1D(filters, 1, padding='same', activation=None,
                   kernel_regularizer=l2(l2_val), name="input_proj")(x)

        # 2. TCN backbone — stacked temporal blocks with exponential dilation
        # Each stack repeats dilations [1, 2, 4, ..., 2^(dilations_per_stack-1)]
        # Multiple stacks repeat this pattern for deeper receptive fields
        block_idx = 0
        for s in range(stack_layers):
            for d in range(dilations_per_stack):
                dilation_rate = 2 ** d
                x = self._temporal_block(
                    x, filters, kernel_size, dilation_rate,
                    dropout, l2_val, activation, use_batch_norm, use_layer_norm,
                    block_name=f"tcn_s{s+1}_d{dilation_rate}_{block_idx}",
                )
                block_idx += 1

        # 3. Global context — use last timestep (causal: contains all past info)
        context = Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

        # 4. Per-horizon output heads
        outputs = []
        self.output_names = []

        for horizon in ph:
            h = context
            for hl in range(head_layers):
                h = Dense(head_units, activation=activation,
                          kernel_regularizer=l2(l2_val),
                          name=f"head_h{horizon}_dense_{hl+1}")(h)
                h = Dropout(dropout, name=f"head_h{horizon}_drop_{hl+1}")(h)
            out = Dense(1, activation="linear",
                        name=f"output_horizon_{horizon}")(h)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"TCN_{len(ph)}H")

        optimizer = AdamW(learning_rate=float(self.params.get("learning_rate", 1e-3)))

        loss_type = str(self.params.get("loss_type", "mae")).strip().lower()
        trend_sigma_lambda = float(self.params.get("trend_sigma_lambda", 0.1))
        pearson_alpha = float(self.params.get("pearson_alpha", 0.5))
        soft_dtw_gamma = float(self.params.get("soft_dtw_gamma", 0.1))
        diff_weight = float(self.params.get("diff_weight", 1.0))
        morphology_batch_size = int(self.params.get("morphology_batch_size", self.params.get("batch_size", 32)))

        if loss_type in ("trend_sigma", "pearson_structural", "soft_dtw", "combined_diff"):
            def _loss_fn(y_true, y_pred):
                return configurable_time_series_loss(
                    y_true,
                    y_pred,
                    loss_type=loss_type,
                    trend_sigma_lambda=trend_sigma_lambda,
                    pearson_alpha=pearson_alpha,
                    soft_dtw_gamma=soft_dtw_gamma,
                    diff_weight=diff_weight,
                    morphology_batch_size=morphology_batch_size,
                )

            loss_dict = {nm: _loss_fn for nm in self.output_names}
        else:
            # MAE loss per horizon (consistent with prior TCN behavior)
            loss_dict = {nm: tf.keras.losses.MeanAbsoluteError() for nm in self.output_names}

        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}

        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)
