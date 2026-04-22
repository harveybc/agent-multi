#!/usr/bin/env python
"""Transformer multi-horizon ioin using BaseBayesianKerasPredictor."""
from __future__ import annotations
import numpy as np, tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44, composite_loss_noreturns, r2_metric, composite_loss_basic
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding

def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def _positional_encoding(position, d_model):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

"""Posterior/prior moved to common.bayesian."""

class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "merged_units": 128,
        "branch_units": 64,
        "activation": "relu",
        "l2_reg": 1e-7,
        "learning_rate": 0.001,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "num_attention_heads": 2,
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "mc_samples": 50,
        "early_patience": 10,
    "positional_encoding": True,
    }
    plugin_debug_vars = [
        "batch_size","merged_units","branch_units","activation","l2_reg","learning_rate","mmd_lambda","sigma_mmd","predicted_horizons","num_attention_heads","mc_samples","positional_encoding"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        w, c = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "relu")
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        merged_units = self.params.get("merged_units", 128)
        branch_units = self.params.get("branch_units", 64)
        lstm_units = max(8, branch_units // 2)
        inputs = Input(shape=(w, c), name="input_layer")
        if self.params.get("positional_encoding", True):  # default True for transformer
            pe = positional_encoding(w, c)
            x = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x = inputs
        heads = self.params.get("num_attention_heads", 2)
        key_dim = max(1, c // heads)
        attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim, kernel_regularizer=l2(l2_reg_v), name="mh_attention")(x, x)
        x = Add(name="res_attn")([x, attn])
        x = LayerNormalization(name="attn_ln")(x)
        x = Conv1D(filters=merged_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_1")(x)
        x = Conv1D(filters=branch_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_2")(x)
        merged = x
        outputs = []
        self.output_names = []
        DenseFlipout = tfp.layers.DenseFlipout
        KLW = self.kl_weight_var
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"
            h_in = Conv1D(filters=branch_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name=f"head_conv1{suf}")(merged)
            h_in = Conv1D(filters=lstm_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name=f"head_conv2{suf}")(h_in)
            lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False), name=f"bilstm{suf}")(h_in)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KLW,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(lstm_out)
            final = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(final)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"TransformerPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
        use_returns = self.params.get("use_returns", False)
        # Keep losses pure-TensorFlow and free of per-batch Python object creation.
        if use_returns:
            def _loss_fn(y_true, y_pred):
                return composite_loss_basic(y_true, y_pred, mmd_lambda=mmd_lambda, sigma=sigma_mmd)
            for nm in self.output_names:
                loss_dict[nm] = _loss_fn
        else:
            huber = Huber()
            for nm in self.output_names:
                loss_dict[nm] = huber
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)


if __name__ == '__main__':
    plugin = Plugin()
    plugin.build_model((24,3), None, {"predicted_horizons":[1,3], "plotted_horizon":1})
    print(plugin.output_names)