import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
import gc

class MyTimeDistributed(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super(MyTimeDistributed, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, **kwargs):
        # inputs shape: (batch, time, features)
        def apply_fn(x):
            # x is a single time slice; expected shape: (features,)
            # We assume x is already rank 1.
            y = self.layer(x, **kwargs)
            # If the output is a vector with a trailing singleton dimension, squeeze it.
            if tf.rank(y) > 0 and tf.shape(y)[-1] == 1:
                y = tf.squeeze(y, axis=-1)
            return y
        # Apply the function over the time dimension using tf.map_fn.
        return tf.map_fn(apply_fn, inputs, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, time, features)
        # We assume each time slice has shape: (features,)
        child_input_shape = input_shape[2:]  # e.g., (8,)
        # Compute output shape for one time slice (without batch dimension)
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        # child_output_shape is expected to be something like (None, 1); remove the None.
        out_dim = child_output_shape[1]
        return (input_shape[0], input_shape[1], out_dim)





class WrappedDenseFlipout(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_posterior_fn, kernel_prior_fn, kernel_divergence_fn, **kwargs):
        super(WrappedDenseFlipout, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_posterior_fn = kernel_posterior_fn
        self.kernel_prior_fn = kernel_prior_fn
        self.kernel_divergence_fn = kernel_divergence_fn
        self.dense_flipout = tfp.layers.DenseFlipout(
            units=units,
            activation=activation,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_prior_fn=kernel_prior_fn,
            kernel_divergence_fn=kernel_divergence_fn
        )
    def call(self, inputs):
        return self.dense_flipout(inputs)
    def compute_output_shape(self, input_shape):
        return self.dense_flipout.compute_output_shape(input_shape)

# --- Custom Callbacks ---
class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """
    Custom ReduceLROnPlateau callback that prints the patience counter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """
    Custom EarlyStopping callback that prints the patience counter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")

class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()

# --- Named initializer to avoid lambda serialization warnings ---
def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# ---------------------------
# CNN Plugin Definition
# ---------------------------
class Plugin:
    """
    CNN Ioin Plugin using Keras for multi-step forecasting with Bayesian uncertainty estimation,
    MMD loss and additional debug print messages (mirroring the ANN/LSTM plugin).
    """

    # Note: Ensure that 'time_horizon' and 'mmd_lambda' are provided in the parameters.
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,
        'activation': 'tanh',
        'kl_weight': 1e-3,
        'time_horizon': 6,    # final output size
        'mmd_lambda': 0.01,   # MMD loss weight
        'early_patience': 10
    }
    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'intermediate_layers', 'initial_layer_size', 'time_horizon']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape, **kwargs):
        """
        Builds a Bayesian CNN using DenseFlipout for uncertainty estimation.
        The final output layer is replaced by a DenseFlipout (without bias)
        plus a separate deterministic bias layer.
        This method mirrors the LSTM plugin's build_model, except using Conv1D layers.
        """
        # Print version info
        print("DEBUG: tensorflow version:", tf.__version__)
        print("DEBUG: tensorflow_probability version:", tfp.__version__)
        print("DEBUG: numpy version:", np.__version__)
        def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
            print("DEBUG: In posterior_mean_field_custom:")
            print("       dtype =", dtype, "kernel_shape =", kernel_shape)
            print("       Received bias_size =", bias_size, "; overriding to 0")
            print("       trainable =", trainable, "name =", name)
            if not isinstance(name, str):
                print("DEBUG: 'name' is not a string; setting to None")
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            print("DEBUG: posterior: computed n =", n)
            c = np.log(np.expm1(1.))
            print("DEBUG: posterior: computed c =", c)
            loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name="posterior_loc")
            scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name="posterior_scale")
            scale = 1e-3 + tf.nn.softplus(scale + c)
            scale = tf.clip_by_value(scale, 1e-3, 1.0)
            print("DEBUG: posterior: created loc shape:", loc.shape, "scale shape:", scale.shape)
            try:
                loc_reshaped = tf.reshape(loc, kernel_shape)
                scale_reshaped = tf.reshape(scale, kernel_shape)
                print("DEBUG: posterior: reshaped loc to", loc_reshaped.shape, "and scale to", scale_reshaped.shape)
            except Exception as e:
                print("DEBUG: Exception during reshape in posterior:", e)
                raise e
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
                reinterpreted_batch_ndims=len(kernel_shape)
            )
        
        def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
            print("DEBUG: In prior_fn:")
            print("       dtype =", dtype, "kernel_shape =", kernel_shape)
            print("       Received bias_size =", bias_size, "; overriding to 0")
            print("       trainable =", trainable, "name =", name)
            if not isinstance(name, str):
                print("DEBUG: 'name' is not a string in prior_fn; setting to None")
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            print("DEBUG: prior_fn: computed n =", n)
            loc = tf.zeros([n], dtype=dtype)
            scale = tf.ones([n], dtype=dtype)
            try:
                loc_reshaped = tf.reshape(loc, kernel_shape)
                scale_reshaped = tf.reshape(scale, kernel_shape)
                print("DEBUG: prior_fn: reshaped loc to", loc_reshaped.shape, "and scale to", scale_reshaped.shape)
            except Exception as e:
                print("DEBUG: Exception during reshape in prior_fn:", e)
                raise e
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
                reinterpreted_batch_ndims=len(kernel_shape)
            )
        # Optionally convert x_train to numpy and print info if provided
        x_train = kwargs.get("x_train", None)
        if x_train is not None:
            x_train = np.array(x_train)
            print("DEBUG: x_train converted to numpy array. Type:", type(x_train), "Shape:", x_train.shape)

        self.params['input_shape'] = input_shape
        l2_reg = self.params.get('l2_reg', 1e-4)

        # Build layer configuration based on parameters
        layer_sizes = []
        current_size = self.params['initial_layer_size']
        print("DEBUG: Initial layer size:", current_size)
        divisor = self.params.get('layer_size_divisor', 2)
        print("DEBUG: Layer size divisor:", divisor)
        int_layers = self.params.get('intermediate_layers', 3)
        print("DEBUG: Number of intermediate layers:", int_layers)
        time_horizon = self.params['time_horizon']
        print("DEBUG: Time horizon (final layer size):", time_horizon)
        for i in range(int_layers):
            layer_sizes.append(current_size)
            print(f"DEBUG: Appended layer size at layer {i+1}: {current_size}")
            current_size = max(current_size // divisor, 1)
            print(f"DEBUG: Updated current_size after division at layer {i+1}: {current_size}")
        layer_sizes.append(time_horizon)
        print("DEBUG: Final layer sizes:", layer_sizes)

        print("DEBUG: CNN input shape:", input_shape)
        inputs = tf.keras.Input(shape=input_shape, name="model_input", dtype=tf.float32)
        print("DEBUG: Created input layer. Shape:", inputs.shape)
        x = inputs

        # --- CNN Feature Extraction ---
        # Initial Dense layer to mix features before convolutional layers
        
        # Add intermediate Conv1D and MaxPooling1D layers
        for idx, size in enumerate(layer_sizes[:-1]):
            if size > 1:
                x = Conv1D(
                    filters=size,
                    kernel_size=3,
                    activation='tanh',
                    kernel_initializer=HeNormal(),
                    padding='same'
                    #kernel_regularizer=l2(l2_reg)
                )(x)
                print(f"DEBUG: After Conv1D layer {idx+1}, x shape: {x.shape}")
                x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)
                print(f"DEBUG: After MaxPooling1D layer {idx+1}, x shape: {x.shape}")

        x = Conv1D(
                filters=1,
                kernel_size=3,
                activation='tanh',
                kernel_initializer=HeNormal(),
                padding='same',
                #kernel_regularizer=l2(l2_reg)
            )(x)
        #flaten
        x = Flatten()(x)
        # --- Bayesian Output Layer Implementation (copied from ANN/LSTM plugin) ---
                # --- Modified Bayesian Output Block for CNN to Produce Multi-Horizon Predictions ---
        # 'x' is the flattened feature vector from the convolutional layers.
        # Expand dimensions to add a time axis:
        x_expanded = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=1), name="expand_dims")(x)
        print("DEBUG: x_expanded shape:", x_expanded.shape)  # Expected: (batch, 1, features)
        
        # Tile the expanded tensor along the time axis for each forecast horizon:
        x_repeated = tf.keras.layers.Lambda(
            lambda t: tf.tile(t, [1, self.params['time_horizon'], 1]),
            name="tile_time"
        )(x_expanded)
        print("DEBUG: x_repeated shape:", x_repeated.shape)  # Expected: (batch, time_horizon, features)
        
        # --- Optional: Add Horizon Embedding ---
        # Build horizon indices (shape: (batch, time_horizon))
        horizon_indices = tf.keras.layers.Lambda(
            lambda r: tf.tile(tf.expand_dims(tf.range(self.params['time_horizon'], dtype=tf.int32), axis=0),
                              [tf.shape(r)[0], 1]),
            name="horizon_indices"
        )(x_repeated)
        print("DEBUG: horizon_indices shape:", horizon_indices.shape)
        
        # Create an embedding for each horizon (using the same dimension as the flattened features)
        horizon_embedding_dim = tf.shape(x)[-1]  # dynamic dimension from x
        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.params['time_horizon'],
            output_dim=int(x.shape[-1]),  # cast to int if possible
            name="horizon_embedding"
        )
        horizon_embeddings = horizon_embedding_layer(horizon_indices)
        print("DEBUG: horizon_embeddings shape:", horizon_embeddings.shape)
        
        # Combine the repeated features with the horizon embeddings via elementwise addition
        x_repeated = tf.keras.layers.Add(name="add_horizon_embedding")([x_repeated, horizon_embeddings])
        print("DEBUG: x_repeated with horizon embedding shape:", x_repeated.shape)
        
        # --- Apply Bayesian Layer on the Time Axis ---
        # Use our WrappedDenseFlipout inside a TimeDistributed-like block.
        # (We already have a proper 3D tensor now.)
        bayesian_td = tf.keras.layers.TimeDistributed(
            WrappedDenseFlipout(
                units=1,
                activation='linear',
                kernel_posterior_fn=posterior_mean_field_custom,
                kernel_prior_fn=prior_fn,
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.params.get('kl_weight', 1e-3),
                name="td_flipout"
            ),
            name="bayesian_td"
        )(x_repeated)
        print("DEBUG: bayesian_td shape:", bayesian_td.shape)  # Expected: (batch, time_horizon, 1)
        
        # Squeeze the last dimension to get (batch, time_horizon)
        bayesian_output = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="squeeze")(bayesian_td)
        print("DEBUG: bayesian_output shape:", bayesian_output.shape)
        
        # --- Deterministic Bias Branch ---
        # Compute a bias from the flattened features 'x'
        bias_layer = Dense(
            units=self.params['time_horizon'],
            activation='linear',
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias",
            kernel_regularizer=l2(l2_reg)
        )(x)
        print("DEBUG: bias_layer shape:", bias_layer.shape)
        
        # Final output: Bayesian output plus bias
        outputs = tf.keras.layers.Add(name="output_add")([bayesian_output, bias_layer])
        print("DEBUG: Final outputs shape:", outputs.shape)

        
        # Deterministic bias branch from x (the flattened features)
        bias_layer = Dense(
            units=self.params['time_horizon'],
            activation='linear',
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias",
            kernel_regularizer=l2(l2_reg)
        )(x)
        print("DEBUG: Deterministic bias layer output shape:", bias_layer.shape)
        
        # Final outputs: add Bayesian output and bias.
        outputs = tf.keras.layers.Add(name="output_add")([bayesian_output, bias_layer])
        print("DEBUG: Final outputs shape after adding bias:", outputs.shape)



        
        self.model = Model(inputs=inputs, outputs=outputs)
        print("DEBUG: Model created. Input shape:", self.model.input_shape, "Output shape:", self.model.output_shape)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.0001)),
            loss=self.custom_loss,
            metrics=['mae']
        )
        print("DEBUG: Adam optimizer created with learning_rate:", self.params.get('learning_rate', 0.0001))
        print("DEBUG: Model compiled with loss=Huber, metrics=['mae']")
        print("CNN Model Summary:")
        self.model.summary()
        print("✅ Standard CNN model built successfully.")

    def compute_mmd(self, x, y, sigma=1.0, sample_size=256):
        """
        Compute Maximum Mean Discrepancy (MMD) using a Gaussian Kernel with a reduced sample size.
        """
        with tf.device('/CPU:0'):
            idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
            x_sample = tf.gather(x, idx)
            y_sample = tf.gather(y, idx)
            def gaussian_kernel(x, y, sigma):
                x = tf.expand_dims(x, 1)
                y = tf.expand_dims(y, 0)
                dist = tf.reduce_sum(tf.square(x - y), axis=-1)
                return tf.exp(-dist / (2.0 * sigma ** 2))
            K_xx = gaussian_kernel(x_sample, x_sample, sigma)
            K_yy = gaussian_kernel(y_sample, y_sample, sigma)
            K_xy = gaussian_kernel(x_sample, y_sample, sigma)
            return tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)

    def custom_loss(self, y_true, y_pred):
        """
        Custom loss function combining Huber loss and MMD loss.
        """
        huber_loss = Huber()(y_true, y_pred)
        mmd_loss = self.compute_mmd(y_pred, y_true)
        total_loss = huber_loss + self.mmd_lambda * mmd_loss
        return total_loss

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the CNN model with MMD loss incorporated and logged at every epoch.
        """
        import tensorflow as tf

        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]

        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")

        # Initialize MMD lambda
        mmd_lambda = self.params.get('mmd_lambda', 0.01)
        self.mmd_lambda = tf.Variable(mmd_lambda, trainable=False, dtype=tf.float32, name='mmd_lambda')

        class KLAnnealingCallback(tf.keras.callbacks.Callback):
            def __init__(self, plugin, target_kl, anneal_epochs):
                super().__init__()
                self.plugin = plugin
                self.target_kl = target_kl
                self.anneal_epochs = anneal_epochs

            def on_epoch_begin(self, epoch, logs=None):
                new_kl = self.target_kl * min(1.0, (epoch + 1) / self.anneal_epochs)
                self.plugin.kl_weight_var.assign(new_kl)
                print(f"DEBUG: Epoch {epoch+1}: KL weight updated to {new_kl}")

        class MMDLoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, plugin, x_train, y_train):
                super().__init__()
                self.plugin = plugin
                self.x_train = x_train
                self.y_train = y_train

            def on_epoch_end(self, epoch, logs=None):
                preds = self.plugin.model(self.x_train, training=True)
                mmd_value = self.plugin.compute_mmd(preds, self.y_train)
                print(f"                                        MMD Lambda = {self.plugin.mmd_lambda.numpy():.6f}, MMD Loss = {mmd_value.numpy():.6f}")

        anneal_epochs = config.get("kl_anneal_epochs", 10) if config is not None else 10
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        mmd_logging_callback = MMDLoggingCallback(self, x_train, y_train)
        
        min_delta = config.get("min_delta", 1e-4) if config is not None else 1e-4
        early_stopping_monitor = EarlyStoppingWithPatienceCounter(
            monitor='val_loss',
            patience=self.params.get('early_patience', 10),
            restore_best_weights=True,
            verbose=2,
            start_from_epoch=10,
            min_delta=min_delta
        )

        reduce_lr_patience = max(1, self.params.get('early_patience', 10) // 3)
        reduce_lr_monitor = ReduceLROnPlateauWithCounter(
            monitor='val_loss',
            factor=0.1,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [kl_callback, mmd_logging_callback, early_stopping_monitor, reduce_lr_monitor, ClearMemoryCallback()]

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=False,
            callbacks=callbacks,
            validation_data=(x_val, y_val)
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train))
        mmd_training_mode = self.compute_mmd(preds_training_mode, y_train)
        print(f"MAE in Training Mode: {mae_training_mode:.6f}, MMD Lambda: {self.mmd_lambda.numpy():.6f}, MMD Loss: {mmd_training_mode:.6f}")

        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train))
        mmd_eval_mode = self.compute_mmd(preds_eval_mode, y_train)
        print(f"MAE in Evaluation Mode: {mae_eval_mode:.6f}, MMD Lambda: {self.mmd_lambda.numpy():.6f}, MMD Loss: {mmd_eval_mode:.6f}")

        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        train_loss, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MAE: {train_mae}")

        val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_loss, val_mae = val_eval_results

        from sklearn.metrics import r2_score
        #train_predictions = self.predict(x_train)
        mc_samples = config.get("mc_samples", 100)
        train_predictions, train_unc = self.predict_with_uncertainty(x_train, mc_samples=mc_samples)
        #val_predictions = self.predict(x_val)
        val_predictions, val_unc =  self.predict_with_uncertainty(x_val, mc_samples=mc_samples)
        return history, train_predictions, train_unc, val_predictions, val_unc

    def predict_with_uncertainty(self, data, mc_samples=100):
        """
        Perform multiple forward passes through the model to estimate prediction uncertainty.
        """
        import numpy as np
        print("DEBUG: Starting predict_with_uncertainty with mc_samples (expected):", mc_samples)
        predictions = []
        for i in range(mc_samples):
            preds = self.model(data, training=True)
            preds_np = preds.numpy()
            print(f"DEBUG: Sample {i+1}/{mc_samples} prediction. Expected shape: (n_samples, time_horizon), Actual shape:", preds_np.shape)
            predictions.append(preds_np)
        predictions = np.array(predictions)
        print("DEBUG: All predictions collected. Expected shape: (mc_samples, n_samples, time_horizon), Actual shape:", predictions.shape)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates

    def predict(self, data):
        import os
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if isinstance(data, tuple):
            data = data[0]
        preds = self.model.predict(data)
        return preds

    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Ioin model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Ioin model loaded from {file_path}")

    def calculate_r2(self, y_true, y_pred):
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in calculate_r2: y_true={y_true.shape}, y_pred={y_pred.shape}")
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
        r2_scores = 1 - (ss_res / ss_tot)
        r2_scores = np.where(ss_tot == 0, 0, r2_scores)
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
        return r2

# ---------------------------
# Debugging usage example
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=(24, 8))
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
