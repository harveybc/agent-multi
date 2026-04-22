#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_mimo.py

Plugin de ioin MIMO multi-horizonte para tu sistema.
Encoder:
  - Conv1D stack (causal) + BiLSTM (return_sequences=True) + z_global opcional.
Decoder:
  - Embeddings de horizonte + z_global → tokens de horizonte.
  - Cross-attention (queries = horizontes, keys/values = secuencia codificada).
  - FFN + residual + LayerNorm.
  - Una salida escalar (batch, 1) por horizonte con nombre "output_horizon_{h}".
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
    Ioin MIMO multi-horizonte con cross-attention sobre la secuencia codificada.
    """

    # Hiperparámetros por defecto (tuneables vía JSON/DEAP)
    plugin_params: Dict[str, Any] = {
        "batch_size": 32,              # Tamaño de batch
        "learning_rate": 1e-3,         # LR para AdamW

        # Encoder
        "encoder_conv_layers": 2,      # Nº de capas Conv1D en el encoder
        "encoder_base_filters": 128,   # Filtros de la primera Conv1D
        "encoder_lstm_units": 128,     # Unidades de la BiLSTM

        # Decoder de horizontes
        "horizon_embedding_dim": 32,   # Dimensión de embeddings de horizonte
        "horizon_attn_heads": 2,       # Nº de cabezas de atención
        "horizon_attn_key_dim": 64,    # Dim de clave en MultiHeadAttention
        "decoder_dropout": 0.1,        # Dropout en FFN del decoder

        # Regularización / activación
        "activation": "relu",          # Activación principal
        "l2_reg": 1e-7,                # L2 suave (similar a tu CNN)

        # Se sobreescribe desde el JSON
        "predicted_horizons": [1],     # Placeholder

        # Positional encoding opcional, como en tu CNN
        "positional_encoding": False,
    }

    plugin_debug_vars: List[str] = [
        "batch_size",
        "learning_rate",
        "encoder_conv_layers",
        "encoder_base_filters",
        "encoder_lstm_units",
        "horizon_embedding_dim",
        "horizon_attn_heads",
        "horizon_attn_key_dim",
        "decoder_dropout",
        "activation",
        "l2_reg",
        "predicted_horizons",
        "positional_encoding",
    ]

    def build_model(
        self,
        input_shape: Tuple[int, int],
        x_train: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Construye y compila el modelo Keras.

        Parameters
        ----------
        input_shape : (int, int)
            (window_size, num_features).
        x_train : Any
            Datos de entrenamiento (no usados aquí, pero se mantiene la firma).
        config : dict
            Config global; se mezcla con plugin_params.
        """
        # Mezclar config externa con parámetros del plugin (igual que en tu CNN)
        if config:
            self.params.update(config)

        def _as_bool(v: Any, default: bool = False) -> bool:
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

        window_size, num_features = input_shape

        # Lista de horizontes de salida
        horizons: List[int] = list(self.params.get("predicted_horizons", []))
        if not horizons:
            raise ValueError(
                "MIMO ioin: 'predicted_horizons' está vacío; "
                "debes definir al menos un horizonte en la configuración."
            )
        horizons = sorted(horizons)

        activation_name: str = self.params.get("activation", "relu")
        l2_reg_value: float = float(self.params.get("l2_reg", 1e-7))

        # ------------------------------------------------------------------ #
        # 1) Entrada + positional encoding opcional                         #
        # ------------------------------------------------------------------ #
        inputs = Input(
            shape=(window_size, num_features),
            name="input_window",
        )

        if _as_bool(self.params.get("positional_encoding", False), default=False):
            pe = positional_encoding(window_size, num_features)
            x = Lambda(
                lambda t, pe=pe: t + pe,
                name="add_positional_encoding",
            )(inputs)
        else:
            x = inputs

        # ------------------------------------------------------------------ #
        # 2) Encoder: Conv1D stack + BiLSTM (secuencia completa)            #
        # ------------------------------------------------------------------ #
        num_conv_layers: int = int(self.params.get("encoder_conv_layers", 2))
        base_filters: int = int(self.params.get("encoder_base_filters", 128))

        for layer_idx in range(num_conv_layers):
            filters = max(16, base_filters // (2 ** layer_idx))
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding="causal",
                activation=activation_name,
                kernel_regularizer=l2(l2_reg_value),
                name=f"enc_conv_{layer_idx+1}",
            )(x)

        lstm_units: int = int(self.params.get("encoder_lstm_units", 128))
        # Salida secuencial para cross-attention: (batch, T, d_model)
        x_seq = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                name="enc_lstm",
            ),
            name="enc_bilstm",
        )(x)

        # Contexto global adicional (se usa para inicializar tokens de horizonte)
        z_global = GlobalAveragePooling1D(
            name="enc_global_avg_pool",
        )(x_seq)

        # Dimensión de modelo (d_model = 2 * lstm_units)
        d_model = int(x_seq.shape[-1])

        # ------------------------------------------------------------------ #
        # 3) Tokens de horizonte: embeddings + z_global                      #
        # ------------------------------------------------------------------ #
        num_horizons: int = len(horizons)
        max_horizon: int = int(max(horizons))
        horizon_emb_dim: int = int(self.params.get("horizon_embedding_dim", 32))

        # IDs de horizonte (p.ej. [4, 8, 12, 16, 20, 24, ...])
        horizon_ids = tf.constant(
            horizons,
            dtype=tf.int32,
            name="horizon_ids",
        )

        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=max_horizon + 1,
            output_dim=horizon_emb_dim,
            name="horizon_embedding",
        )
        # (num_horizons, horizon_emb_dim)
        horizon_embs = horizon_embedding_layer(horizon_ids)

        # Expansión a (1, num_horizons, horizon_emb_dim)
        horizon_embs_expanded = Lambda(
            lambda e: tf.expand_dims(e, axis=0),
            name="expand_horizon_embs",
        )(horizon_embs)

        # Réplica a lo largo del batch usando z_global para el batch_size
        horizon_embs_tiled = Lambda(
            lambda tensors: tf.tile(
                tensors[0],
                [tf.shape(tensors[1])[0], 1, 1],
            ),
            name="tile_horizon_embs",
        )([horizon_embs_expanded, z_global])

        # Expandimos z_global a (batch, 1, d_model)
        z_expanded = Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            name="expand_global",
        )(z_global)

        # Tile de z_global a (batch, num_horizons, d_model)
        z_tiled = Lambda(
            lambda z: tf.tile(
                z,
                [1, num_horizons, 1],
            ),
            name="tile_global",
        )(z_expanded)

        # Concat global ⊕ embedding horizonte → tokens iniciales (batch, H, d_model + emb_dim)
        horizon_tokens = Concatenate(
            axis=-1,
            name="concat_global_horizon",
        )([z_tiled, horizon_embs_tiled])

        # Proyectar tokens de horizonte a d_model para usar misma dimensión que x_seq
        horizon_tokens_proj = Dense(
            units=d_model,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name="horizon_proj_dense",
        )(horizon_tokens)

        # ------------------------------------------------------------------ #
        # 4) Cross-attention: queries = horizontes, keys/values = x_seq     #
        # ------------------------------------------------------------------ #
        attn_heads: int = int(self.params.get("horizon_attn_heads", 2))
        attn_key_dim: int = int(self.params.get("horizon_attn_key_dim", 64))
        decoder_dropout: float = float(self.params.get("decoder_dropout", 0.1))

        cross_attn = MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name="horizon_cross_mha",
        )(horizon_tokens_proj, x_seq)

        # Residual + LayerNorm
        tokens_res = Add(
            name="horizon_cross_residual",
        )([horizon_tokens_proj, cross_attn])

        tokens_norm = LayerNormalization(
            name="horizon_cross_ln",
        )(tokens_res)

        # FFN sobre tokens de horizonte
        ff = Dense(
            units=d_model,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name="horizon_ffn_dense",
        )(tokens_norm)

        ff = Dropout(
            decoder_dropout,
            name="horizon_ffn_dropout",
        )(ff)

        tokens_ffn_res = Add(
            name="horizon_ffn_residual",
        )([tokens_norm, ff])

        tokens_final = LayerNormalization(
            name="horizon_ffn_ln",
        )(tokens_ffn_res)

        # Proyección final antes de la cabeza escalar
        head_dim = d_model  # podrías tunear esto si quieres
        head_repr = Dense(
            units=head_dim,
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name="horizon_head_dense",
        )(tokens_final)

        # ------------------------------------------------------------------ #
        # 5) Salidas MIMO: 1 escalar (batch, 1) por horizonte               #
        # ------------------------------------------------------------------ #
        horizon_outputs = Dense(
            units=1,
            activation=None,
            name="horizon_output_dense",
        )(head_repr)

        outputs: List[tf.Tensor] = []
        self.output_names: List[str] = []

        for idx, h in enumerate(horizons):
            out_i = Lambda(
                lambda t, i=idx: t[:, i, :],
                name=f"output_horizon_{h}",
            )(horizon_outputs)
            outputs.append(out_i)
            self.output_names.append(f"output_horizon_{h}")

        # ------------------------------------------------------------------ #
        # 6) Modelo, pérdidas y compilación                                 #
        # ------------------------------------------------------------------ #
        self.model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f"MIMOPredictor_{len(horizons)}H",
        )

        optimizer = AdamW(
            learning_rate=float(self.params.get("learning_rate", 1e-3)),
        )

        # MAE por horizonte como loss básica (estable y alineada con mae_magnitude)
        loss_dict: Dict[str, Any] = {
            name: tf.keras.losses.MeanAbsoluteError()
            for name in self.output_names
        }

        metrics_dict: Dict[str, List[Any]] = {
            name: [mae_magnitude]
            for name in self.output_names
        }

        self.model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            metrics=metrics_dict,
        )

        if not self.params.get('quiet', False):
            self.model.summary(line_length=140)
