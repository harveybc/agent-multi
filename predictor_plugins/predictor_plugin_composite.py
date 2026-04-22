#!/usr/bin/env python
"""CNN multi-horizon ioin using shared BaseBayesianKerasPredictor.

Concrete plugin now only implements build_model & parameter lists; all training,
metrics, persistence, and MC uncertainty logic are inherited.
"""
from __future__ import annotations
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, composite_loss_basic, random_normal_initializer_44, composite_loss_noreturns, r2_metric
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
        """
        Multi-branch Conv1D design that preserves the time axis for the heads.

        Modes (config flags preserved from your plugin):
        - close_window_only = True:
            * Branch A (CLOSE full window): causal Conv1D stack, no pooling -> (batch, w, 64).
            * Branch B15/B30 (HF): if process_HF True, take last 16 columns from the last row:
                - 15m: first 8 => shape (batch, 8, 1) -> small Conv1D -> linear resize to w -> (batch, w, 32).
                - 30m: next  8 => shape (batch, 8, 1) -> small Conv1D -> linear resize to w -> (batch, w, 32).
                If process_HF False, both are zero tensors shaped (batch, w, 32).
            * Branch C (point features at t): last-row vector (excluding CLOSE and HF) -> tile to w -> 1x1 Conv1D -> (batch, w, 32).
            * Merge: concat along channels -> (batch, w, 160) -> Conv1D fuse to sizes[-1] -> feed unchanged head stacks.
        - close_window_only = False:
            * Your original single-stream Conv1D backbone (unchanged).

        Notes:
        - CLOSE default index = 3 (OPEN=0, HIGH=1, LOW=2, CLOSE=3). DATE_TIME is not inside the (w,c) tensor.
        - Latest tick is the LAST row of the window (index -1); see sliding_windows.py (baseline time t is last element).  # :contentReference[oaicite:1]{index=1}
        """
        # --------------------------- merge runtime config ---------------------------
        if config:
            self.params.update(config)

        # --------------------------- unpack params ---------------------------------
        w, c = input_shape                                          # window length and channel count
        ph = self.params["predicted_horizons"]                      # list of horizons
        act = self.params.get("activation", "relu")                 # activation
        l2_reg_v = self.params.get("l2_reg", 1e-4)                  # L2 weight
        initial_layer_size = self.params.get("initial_layer_size", 128)
        layer_size_divisor = self.params.get("layer_size_divisor", 2)
        intermediate_layers = int(self.params.get("intermediate_layers", 2))
        head_layers = int(self.params.get("head_layers", 2))
        use_returns = self.params.get("use_returns", False)
        use_pe = self.params.get("positional_encoding", False)

        # Flags per your request
        close_window_only = bool(self.params.get("close_window_only", False))
        process_HF        = bool(self.params.get("process_HF", False))

        # CLOSE column index (DATE_TIME not in tensor). Default: CLOSE=3
        close_idx = int(self.params.get("close_channel", 3))
        if not (0 <= close_idx < c):
            raise ValueError(f"Invalid close_channel={close_idx} for c={c}")

        # --------------------------- single input ----------------------------------
        inputs = Input(shape=(w, c), name="input_layer")            # (batch, w, c)

        # Optional positional encoding
        if use_pe:
            pe = positional_encoding(w, c)                          # (w, c) constant PE
            x_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x_in = inputs

        # ========================================================================== #
        # MODE 1: close_window_only == True -> three Conv1D branches, time preserved #
        # ========================================================================== #
        if close_window_only:
            # Helper: linear time-resize (no pooling, no stride artifacts)
            def resize_time_to(x, target_len, name):
                # x: (batch, time_in, channels) -> resize time dimension to target_len using bilinear interpolation
                def _resize(t):
                    shp = tf.shape(t)
                    b, tin, ch = shp[0], shp[1], shp[2]
                    t4 = tf.expand_dims(t, axis=1)                 # (b, 1, tin, ch) treat time as "width"
                    r4 = tf.image.resize(t4, size=(1, target_len), method="bilinear", antialias=True)
                    r3 = tf.squeeze(r4, axis=1)                    # (b, target_len, ch)
                    return r3
                return Lambda(_resize, name=name)(x)

            # --------------------- Branch A: CLOSE full window -> (w, 64) ----------
            # Slice CLOSE: (batch, w, 1)
            A_seq = Lambda(lambda t, ch=close_idx: t[:, :, ch:ch+1], name="A_slice_close_full")(x_in)
            # Causal Conv1D stack (no stride -> keep time length = w)
            A_tmp = Conv1D(64, kernel_size=7, padding="causal", activation=act,
                        kernel_regularizer=l2(l2_reg_v), name="A_conv1")(A_seq)           # (b, w, 64)
            A_tmp = Conv1D(64, kernel_size=5, padding="causal", activation=act,
                        kernel_regularizer=l2(l2_reg_v), name="A_conv2")(A_tmp)           # (b, w, 64)
            A_tmp = Conv1D(64, kernel_size=3, padding="causal", activation=act,
                        kernel_regularizer=l2(l2_reg_v), name="A_conv3")(A_tmp)           # (b, w, 64)
            A_out = A_tmp  # (b, w, 64)

            # Latest tick row (baseline time t) per sliding_windows builder (last element)
            last_row = Lambda(lambda t: t[:, -1, :], name="last_row_t")(x_in)                # (b, c)

            # --------------------- HF column discovery if requested ----------------
            hf15_idx, hf30_idx = [], []
            if process_HF:
                if c < 16:
                    raise ValueError("process_HF=True but window has fewer than 16 columns.")
                hf_base = c - 16                        # start index of HF block at end
                hf15_idx = list(range(hf_base, hf_base + 8))   # first  8 = 15m ticks
                hf30_idx = list(range(hf_base + 8, c))         # next   8 = 30m ticks

            # --------------------- Branch B15: last-row 8 -> (w, 32) ---------------
            if hf15_idx:
                B15_vec = Lambda(lambda r, idx=hf15_idx: tf.gather(r, indices=idx, axis=1),
                                name="B15_gather_lastrow")(last_row)                         # (b, 8)
                B15_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1), name="B15_as_seq")(B15_vec)  # (b, 8, 1)
                B15_tmp = Conv1D(32, kernel_size=3, padding="same", activation=act,
                                kernel_regularizer=l2(l2_reg_v), name="B15_conv1")(B15_seq)       # (b, 8, 32)
                B15_tmp = Conv1D(32, kernel_size=3, padding="same", activation=act,
                                kernel_regularizer=l2(l2_reg_v), name="B15_conv2")(B15_tmp)       # (b, 8, 32)
                B15_out = resize_time_to(B15_tmp, target_len=w, name="B15_resize_to_w")            # (b, w, 32)
            else:
                B15_out = Lambda(lambda t: tf.zeros((tf.shape(t)[0], w, 32), dtype=t.dtype),
                                name="B15_zero")(A_out)                                            # (b, w, 32)

            # --------------------- Branch B30: last-row 8 -> (w, 32) ---------------
            if hf30_idx:
                B30_vec = Lambda(lambda r, idx=hf30_idx: tf.gather(r, indices=idx, axis=1),
                                name="B30_gather_lastrow")(last_row)                         # (b, 8)
                B30_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1), name="B30_as_seq")(B30_vec)  # (b, 8, 1)
                B30_tmp = Conv1D(32, kernel_size=3, padding="same", activation=act,
                                kernel_regularizer=l2(l2_reg_v), name="B30_conv1")(B30_seq)       # (b, 8, 32)
                B30_tmp = Conv1D(32, kernel_size=3, padding="same", activation=act,
                                kernel_regularizer=l2(l2_reg_v), name="B30_conv2")(B30_tmp)       # (b, 8, 32)
                B30_out = resize_time_to(B30_tmp, target_len=w, name="B30_resize_to_w")            # (b, w, 32)
            else:
                B30_out = Lambda(lambda t: tf.zeros((tf.shape(t)[0], w, 32), dtype=t.dtype),
                                name="B30_zero")(A_out)                                            # (b, w, 32)

            # --------------------- Branch C: point features -> (w, 32) -------------
            def _point_indices(total_c, close_i, hf15, hf30):
                hfset = set(hf15) | set(hf30)
                return [j for j in range(total_c) if j != close_i and j not in hfset]

            point_idx = _point_indices(c, close_idx, hf15_idx, hf30_idx)
            if len(point_idx) > 0:
                C_vec = Lambda(lambda r, idx=point_idx: tf.gather(r, indices=idx, axis=1),
                            name="C_point_gather")(last_row)                                  # (b, M)
                C_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1), name="C_point_expand")(C_vec) # (b, M, 1)
                # Tile along time to length w (constant-in-time contextual features)
                C_tiled = Lambda(lambda x, W=w: tf.tile(tf.expand_dims(x, axis=1), [1, W, 1, 1]),
                                name="C_point_tile4D")(C_seq)                                     # (b, w, M, 1)
                C_back = Lambda(lambda x: tf.squeeze(x, axis=-1), name="C_point_squeeze")(C_tiled) # (b, w, M)
                C_out  = Conv1D(32, kernel_size=1, padding="same", activation=act,
                                kernel_regularizer=l2(l2_reg_v), name="C_point_conv1x1")(C_back)   # (b, w, 32)
            else:
                C_out = Lambda(lambda t: tf.zeros((tf.shape(t)[0], w, 32), dtype=t.dtype),
                            name="C_point_zero")(A_out)                                         # (b, w, 32)

            # --------------------- Merge (keep time axis) --------------------------
            merged_time = Lambda(lambda xs: tf.concat(xs, axis=-1), name="merge_concat_time")(
                [A_out, B15_out, B30_out, C_out]
            )  # (b, w, 64+32+32+32=160)

            # Fuse to sizes[-1] channels so the head stacks see the usual depth
            # Build the same size schedule to get sizes[-1] for consistency with heads
            num_layers = max(1, intermediate_layers)
            sizes = [initial_layer_size] + [
                max(8, initial_layer_size // (layer_size_divisor ** i)) for i in range(1, num_layers)
            ]
            fused = Conv1D(
                filters=sizes[-1], kernel_size=3, strides=1, padding="same",
                activation=act, kernel_regularizer=l2(l2_reg_v), name="fuse_conv1"
            )(merged_time)  # (b, w, sizes[-1])
            fused = Conv1D(
                filters=sizes[-1], kernel_size=3, strides=1, padding="same",
                activation=act, kernel_regularizer=l2(l2_reg_v), name="fuse_conv2"
            )(fused)        # (b, w, sizes[-1])

            merged = fused                               # (b, w, sizes[-1]) for heads
            last_root_filters = sizes[-1]               # keep head sizing logic consistent below

            # --------------------- Heads (unchanged pattern) -----------------------
            outputs = []
            self.output_names = []
            DenseFlipout = tfp.layers.DenseFlipout
            KL_WEIGHT = self.kl_weight_var
            mmd_lambda = self.params.get("mmd_lambda", 0.0)
            sigma_mmd = self.params.get("sigma_mmd", 1.0)

            for horizon in ph:
                suf = f"_h{horizon}"

                head_num_layers = max(1, head_layers)
                base_head_filters = max(8, last_root_filters // 2)
                head_sizes = [base_head_filters] + [
                    max(8, base_head_filters // (layer_size_divisor ** i)) for i in range(1, head_num_layers)
                ]

                h_in = merged
                for j, f_j in enumerate(head_sizes):
                    h_in = Conv1D(
                        filters=f_j, kernel_size=3, strides=2, padding="same",
                        activation=act, kernel_regularizer=l2(l2_reg_v),
                        name=f"head_conv{j+1}{suf}",
                    )(h_in)

                last_head_filters = head_sizes[-1]
                lstm_total_units = max(8, last_head_filters // 2)
                lstm_out = Bidirectional(
                    LSTM(max(1, lstm_total_units // 2), return_sequences=False),
                    name=f"bilstm{suf}",
                )(h_in)

                flip_name = f"flipout{suf}"
                flip_layer = DenseFlipout(
                    units=1, activation="linear",
                    kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                    kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                    name=flip_name,
                )
                bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)
                bias  = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(lstm_out)
                out   = Add(name=f"output_horizon_{horizon}")([bayes, bias])

                outputs.append(out)
                self.output_names.append(f"output_horizon_{horizon}")

            # Compile identical to original
            self.model = Model(inputs=inputs, outputs=outputs, name=f"CNN3BranchConv_{len(ph)}H")
            optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
            loss_dict = {}
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
            return

        # ========================================================================== #
        # MODE 2: close_window_only == False -> original single-stream backbone       #
        # ========================================================================== #
        # (Unchanged from your current plugin)
        num_layers = max(1, intermediate_layers)
        sizes = [initial_layer_size] + [
            max(8, initial_layer_size // (layer_size_divisor ** i)) for i in range(1, num_layers)
        ]
        x = x_in
        for i, filters_i in enumerate(sizes):
            x = Conv1D(
                filters=filters_i, kernel_size=3, strides=2, padding="same",
                activation=act, kernel_regularizer=l2(l2_reg_v),
                name=f"conv_{i+1}",
            )(x)

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
            head_num_layers = max(1, head_layers)
            base_head_filters = max(8, last_root_filters // 2)
            head_sizes = [base_head_filters] + [
                max(8, base_head_filters // (layer_size_divisor ** i)) for i in range(1, head_num_layers)
            ]

            h_in = merged
            for j, f_j in enumerate(head_sizes):
                h_in = Conv1D(
                    filters=f_j, kernel_size=3, strides=2, padding="same",
                    activation=act, kernel_regularizer=l2(l2_reg_v),
                    name=f"head_conv{j+1}{suf}",
                )(h_in)

            last_head_filters = head_sizes[-1]
            lstm_total_units = max(8, last_head_filters // 2)
            lstm_out = Bidirectional(
                LSTM(max(1, lstm_total_units // 2), return_sequences=False),
                name=f"bilstm{suf}",
            )(h_in)

            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1, activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(lstm_out)
            out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"CNNPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
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


if __name__ == "__main__":  # Minimal sanity check
    plugin = Plugin()
    plugin.build_model((24, 3), None, {"predicted_horizons": [1]})
    print(plugin.output_names)