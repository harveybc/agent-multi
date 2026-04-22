#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_stl_mimo.py

STL-MIMO Ioin Plugin.
Supports:
1. Standard MIMO (used for sequential training of components).
2. Simultaneous MIMO (3 branches for Trend, Seasonal, Resid -> Sum -> Loss).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Bidirectional,
    LSTM,
    GlobalAveragePooling1D,
    LayerNormalization,
    Dropout,
    Add,
    Concatenate,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding
from .common.losses import mae_magnitude


class Plugin(BaseBayesianKerasPredictor):
    """
    Ioin MIMO capable of Simultaneous STL decomposition training.
    """

    plugin_params: Dict[str, Any] = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "encoder_conv_layers": 2,
        "encoder_base_filters": 128,
        "encoder_lstm_units": 128,
        "horizon_embedding_dim": 32,
        "horizon_attn_heads": 4,
        "horizon_attn_key_dim": 64,
        "decoder_dropout": 0.1,
        "activation": "relu",
        "l2_reg": 1e-7,
        "predicted_horizons": [1],
        "positional_encoding": False,
        
        # New param for simultaneous mode
        "use_simultaneous_stl": False, 
    }

    plugin_debug_vars: List[str] = [
        "batch_size",
        "learning_rate",
        "use_simultaneous_stl",
    ]

    def _build_mimo_block(self, inputs, horizons, prefix=""):
        """
        Builds a complete MIMO encoder-decoder block.
        Returns a list of output tensors (one per horizon).
        """
        activation_name = self.params.get("activation", "relu")
        l2_reg_value = float(self.params.get("l2_reg", 1e-7))
        
        # --- Encoder ---
        x = inputs
        num_conv_layers = int(self.params.get("encoder_conv_layers", 2))
        base_filters = int(self.params.get("encoder_base_filters", 128))

        for layer_idx in range(num_conv_layers):
            filters = max(16, base_filters // (2 ** layer_idx))
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding="causal",
                activation=activation_name,
                kernel_regularizer=l2(l2_reg_value),
                name=f"{prefix}enc_conv_{layer_idx+1}",
            )(x)

        lstm_units = int(self.params.get("encoder_lstm_units", 128))
        x_seq = Bidirectional(
            LSTM(lstm_units, return_sequences=True, name=f"{prefix}enc_lstm"),
            name=f"{prefix}enc_bilstm",
        )(x)

        z_global = GlobalAveragePooling1D(name=f"{prefix}enc_global_avg_pool")(x_seq)
        d_model = int(x_seq.shape[-1])

        # --- Decoder ---
        num_horizons = len(horizons)
        max_horizon = int(max(horizons))
        horizon_emb_dim = int(self.params.get("horizon_embedding_dim", 32))

        horizon_ids = tf.constant(horizons, dtype=tf.int32, name=f"{prefix}horizon_ids")
        
        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=max_horizon + 1,
            output_dim=horizon_emb_dim,
            name=f"{prefix}horizon_embedding",
        )
        horizon_embs = horizon_embedding_layer(horizon_ids)

        horizon_embs_expanded = Lambda(lambda e: tf.expand_dims(e, axis=0))(horizon_embs)
        
        # Tile embeddings
        horizon_embs_tiled = Lambda(
            lambda tensors: tf.tile(tensors[0], [tf.shape(tensors[1])[0], 1, 1])
        )([horizon_embs_expanded, z_global])

        # Tile global context
        z_expanded = Lambda(lambda z: tf.expand_dims(z, axis=1))(z_global)
        z_tiled = Lambda(lambda z: tf.tile(z, [1, num_horizons, 1]))(z_expanded)

        horizon_tokens = Concatenate(axis=-1)([z_tiled, horizon_embs_tiled])

        horizon_tokens_proj = Dense(
            units=d_model,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name=f"{prefix}horizon_proj_dense",
        )(horizon_tokens)

        # --- Cross Attention ---
        attn_heads = int(self.params.get("horizon_attn_heads", 4))
        attn_key_dim = int(self.params.get("horizon_attn_key_dim", 64))
        decoder_dropout = float(self.params.get("decoder_dropout", 0.1))

        cross_attn = MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name=f"{prefix}horizon_cross_mha",
        )(horizon_tokens_proj, x_seq)

        tokens_res = Add()([horizon_tokens_proj, cross_attn])
        tokens_norm = LayerNormalization()(tokens_res)

        ff = Dense(
            units=d_model,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name=f"{prefix}horizon_ffn_dense",
        )(tokens_norm)
        ff = Dropout(decoder_dropout)(ff)
        
        tokens_ffn_res = Add()([tokens_norm, ff])
        tokens_final = LayerNormalization()(tokens_ffn_res)

        head_repr = Dense(
            units=d_model,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name=f"{prefix}horizon_head_dense",
        )(tokens_final)

        horizon_outputs = Dense(
            units=1,
            activation=None,
            name=f"{prefix}horizon_output_dense",
        )(head_repr)

        # Split into list of tensors
        outputs = []
        for idx, h in enumerate(horizons):
            out_i = Lambda(lambda t, i=idx: t[:, i, :], name=f"{prefix}output_horizon_{h}")(horizon_outputs)
            outputs.append(out_i)
            
        return outputs

    def build_model(self, input_shape: Tuple[int, int], x_train: Any, config: Dict[str, Any]) -> None:
        if config:
            self.params.update(config)

        window_size, num_features = input_shape
        horizons = sorted(list(self.params.get("predicted_horizons", [])))
        if not horizons:
            raise ValueError("Predicted horizons must be defined.")

        inputs = Input(shape=(window_size, num_features), name="input_window")
        
        # Positional Encoding
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(window_size, num_features)
            x = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x = inputs

        use_simultaneous = self.params.get("use_simultaneous_stl", False)

        if use_simultaneous:
            # Build 3 parallel branches
            trend_outputs = self._build_mimo_block(x, horizons, prefix="trend_")
            seasonal_outputs = self._build_mimo_block(x, horizons, prefix="seasonal_")
            resid_outputs = self._build_mimo_block(x, horizons, prefix="resid_")
            
            # Sum outputs
            final_outputs = []
            self.output_names = []
            for i, h in enumerate(horizons):
                # Sum the 3 components for this horizon
                summed = Add(name=f"output_horizon_{h}")([
                    trend_outputs[i], 
                    seasonal_outputs[i], 
                    resid_outputs[i]
                ])
                final_outputs.append(summed)
                self.output_names.append(f"output_horizon_{h}")
                
        else:
            # Standard Single MIMO
            final_outputs = self._build_mimo_block(x, horizons, prefix="")
            self.output_names = [f"output_horizon_{h}" for h in horizons]

        # Compile
        self.model = Model(inputs=inputs, outputs=final_outputs, name="STL_MIMO_Predictor")
        
        optimizer = AdamW(learning_rate=float(self.params.get("learning_rate", 1e-3)))
        
        loss_dict = {name: tf.keras.losses.MeanAbsoluteError() for name in self.output_names}
        metrics_dict = {name: [mae_magnitude] for name in self.output_names}

        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)
