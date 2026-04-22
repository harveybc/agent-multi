#!/usr/bin/env python
"""ANN multi-horizon ioin using BaseBayesianKerasPredictor with optional positional encoding."""
from __future__ import annotations
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from .common.losses import (
    mae_magnitude,
    composite_loss_multihead as composite_loss,
    composite_loss_noreturns,
    composite_loss_basic,
    random_normal_initializer_44,
)
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding


class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "hidden_units": 256,
        "num_hidden_layers": 2,
        "dropout_rate": 0.1,
        "activation": "relu",
        "learning_rate": 0.001,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "early_patience": 10,
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "mc_samples": 50,
        "positional_encoding": False,
    }
    plugin_debug_vars = [
        "batch_size","hidden_units","num_hidden_layers","dropout_rate","learning_rate","mmd_lambda","sigma_mmd","predicted_horizons","early_patience","kl_weight","mc_samples","positional_encoding"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        window, channels = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "relu")
        hidden = self.params.get("hidden_units", 256)
        n_layers = self.params.get("num_hidden_layers", 2)
        dr = self.params.get("dropout_rate", 0.0)
        inputs = Input(shape=(window, channels), name="input_layer")
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(window, channels)
            enc = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            enc = inputs
        # NOTE: Using Keras Flatten layer instead of tf.reshape to avoid the KerasTensor error.
        x = Flatten(name="flatten_inputs")(enc)
        for i in range(n_layers):
            x = Dense(hidden, activation=act, name=f"shared_dense_{i}")(x)
            if dr > 0:
                x = Dropout(dr, name=f"shared_dropout_{i}")(x)
        trunk = x
        DenseFlipout = tfp.layers.DenseFlipout
        KLW = self.kl_weight_var
        outputs = []
        self.output_names = []
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"
            head = Dense(hidden // 2, activation=act, name=f"head_dense1{suf}")(trunk)
            if dr > 0:
                head = Dropout(dr, name=f"head_dropout1{suf}")(head)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KLW,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(head)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(head)
            out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"ANNPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
        # Mirror CNN plugin logic: choose loss variant depending on use_returns flag.
        use_returns = self.params.get("use_returns", False)
        # CRITICAL: keep losses pure TF; do not create Python objects per batch.
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
    plug = Plugin({"predicted_horizons": [1,3], "plotted_horizon": 1, "positional_encoding": True})
    plug.build_model((24,3), None, {})
    print('Outputs:', plug.output_names)