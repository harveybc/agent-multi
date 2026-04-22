#!/usr/bin/env python
"""CNN multi-horizon ioin using shared BaseBayesianKerasPredictor.

Concrete plugin now only implements build_model & parameter lists; all training,
metrics, persistence, and MC uncertainty logic are inherited.
"""
from __future__ import annotations
import sys
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from .common.losses import mae_magnitude, composite_loss_basic, random_normal_initializer_44, r2_metric
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding


class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "branch_units": 32,
        "merged_units": 128,
        "learning_rate": 0.001,
        "activation": "relu",
        "l2_reg": 1e-7,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "early_patience": 10,
        "mc_samples": 50,
    "positional_encoding": False,
    }
    plugin_debug_vars = [
        "batch_size","branch_units","merged_units","learning_rate","l2_reg","mmd_lambda","sigma_mmd","predicted_horizons","kl_weight","early_patience","mc_samples","positional_encoding"
    ]

    def build_model(self, input_shape, x_train, config):
        # Ensure GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("\n" + "="*80)
            print("CRITICAL ERROR: No GPU detected by TensorFlow!")
            print("="*80)
            print("The application requires a GPU to run. Please check:")
            print("1. NVIDIA drivers are installed and up to date.")
            print("2. CUDA and cuDNN are installed and compatible with TensorFlow.")
            print("3. The 'tensorflow' or 'tensorflow-gpu' package is correctly installed.")
            print(f"Current TensorFlow version: {tf.__version__}")
            print("="*80 + "\n")
            sys.exit(1)
        else:
            print(f"GPU initialized: {len(gpus)} device(s) found.")
            # IMPORTANT: Do not call set_memory_growth() here.
            # GPU allocator/memory growth must be configured before TF initializes the device.
            # This is handled centrally in `app/main.py` to avoid late-init warnings.

        if config:
            self.params.update(config)
        w, c = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "relu")
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        initial_layer_size = self.params.get("initial_layer_size", 128)
        layer_size_divisor = self.params.get("layer_size_divisor", 2)
        intermediate_layers = int(self.params.get("intermediate_layers", 1))
        head_layers = int(self.params.get("head_layers", 1))
        use_returns = self.params.get("use_returns", False)

        inputs = Input(shape=(w, c), name="input_layer")
        # Optional positional encoding
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(w, c)
            x_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x_in = inputs

        # Dynamic Conv1D stack with first layer = initial_layer_size
        x = x_in
        num_layers = max(1, intermediate_layers)

        # Build channel sizes: first layer exactly initial_layer_size, subsequent layers downscaled
        sizes = [initial_layer_size] + [
            max(8, initial_layer_size // (layer_size_divisor ** i)) for i in range(1, num_layers)
        ]

        for i, filters_i in enumerate(sizes):
            x = Conv1D(
            filters=filters_i,
            kernel_size=3,
            strides=2,
            padding="causal",
            activation=act,
            kernel_regularizer=l2(l2_reg_v),
            name=f"conv_{i+1}",
            )(x)

        # Per-head dynamic Conv1D stack derived from the last root layer
        last_root_filters = sizes[-1]
        merged = x
        outputs = []
        self.output_names = []
        DenseFlipout = tfp.layers.DenseFlipout
        KL_WEIGHT = self.kl_weight_var
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"

            # Build head conv sizes:
            # - first head layer = half of last root layer (>= 8)
            # - subsequent layers use the same layer_size_divisor (>= 8)
            head_num_layers = max(1, head_layers)
            base_head_filters = max(8, last_root_filters // 2)
            head_sizes = [base_head_filters] + [
            max(8, base_head_filters // (layer_size_divisor ** i)) for i in range(1, head_num_layers)
            ]

            h_in = merged
            for j, f_j in enumerate(head_sizes):
                h_in = Conv1D(
                    filters=f_j,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation=act,
                    kernel_regularizer=l2(l2_reg_v),
                    name=f"head_conv{j+1}{suf}",
                )(h_in)

            # LSTM total units = half of the last head conv filters (>= 8)
            last_head_filters = head_sizes[-1]
            lstm_total_units = max(8, last_head_filters // 2)
            lstm_out = Bidirectional(
                LSTM(max(1, lstm_total_units // 2), return_sequences=False),
                name=f"bilstm{suf}",
                )(h_in)
            flip_name = f"flipout{suf}"
            flip_name_2 = f"flipout_2_{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="relu",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=flip_name,
            )
            #bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)

            bias = Dense(16, activation="relu", kernel_initializer=random_normal_initializer_44, name=f"bias_0_{suf}")(lstm_out)
            bias = Dense(1, activation="linear",kernel_initializer=random_normal_initializer_44, name=f"output_horizon_{horizon}")(bias)
            #out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            out = bias
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"CNNPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
        # CRITICAL: Keep losses pure-TensorFlow and free of per-batch Python object creation.
        # Creating new Python lists inside the loss call path can cause tf.function retracing
        # and unbounded host-RAM growth across long trainings.
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

if __name__ == "__main__":  # Minimal sanity check
    plugin = Plugin()
    plugin.build_model((24, 3), None, {"predicted_horizons": [1]})
    print(plugin.output_names)