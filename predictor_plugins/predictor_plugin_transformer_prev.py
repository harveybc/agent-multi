import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, MultiHeadAttention, Add, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
import gc
#LeakyReLU

from tensorflow.keras.layers import LeakyReLU

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
# Transformer Plugin Definition
# ---------------------------
class Plugin:
    """
    Transformer Ioin Plugin using Keras for multi-step forecasting with Bayesian uncertainty estimation,
    MMD loss, KL annealing and detailed debug messages.
    
    The input is expected as a 3D tensor: (window_size, num_features).
    """
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,   # Number of transformer blocks
        'initial_layer_size': 32,   # Also used as the embedding dimension
        'layer_size_divisor': 2,    # Not used inside transformer blocks (kept for compatibility)
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-2,
        'kl_weight': 1e-3,
        'num_heads': 4,
        'time_horizon': 6,          # Final output dimension (forecast horizon)
        'mmd_lambda': 0.01,         # MMD loss weight
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
        debug_info.update(self.get_debug_info())



    def build_model(self, input_shape, **kwargs):
        """
        Builds a Transformer model with a Bayesian output layer.
        The input is expected as a 3D tensor: (window_size, num_features).
        The architecture first projects the input into a fixed embedding dimension,
        adds positional encoding, then applies several transformer blocks, global average pooling,
        and finally a Bayesian output layer.
        """
        print("DEBUG: tensorflow version:", tf.__version__)
        print("DEBUG: tensorflow_probability version:", tfp.__version__)
        print("DEBUG: numpy version:", np.__version__)

        # Optionally log x_train info if provided
        x_train = kwargs.get("x_train", None)
        if x_train is not None:
            x_train = np.array(x_train)
            print("DEBUG: x_train converted to numpy array. Type:", type(x_train), "Shape:", x_train.shape)

        self.params['input_shape'] = input_shape  # (window_size, num_features)
        l2_reg = self.params.get('l2_reg', 1e-4)
        num_heads = self.params['num_heads']
        time_horizon = self.params['time_horizon']
        embedding_dim = self.params.get('initial_layer_size', 32)

        print("DEBUG: Input shape:", input_shape)
        inputs = tf.keras.Input(shape=input_shape, name="model_input", dtype=tf.float32)
        print("DEBUG: Created input layer. Shape:", inputs.shape)
        x = inputs

        # Project input to fixed embedding dimension
        x = Dense(embedding_dim, activation=self.params['activation'],
                kernel_initializer=GlorotUniform(), name="input_projection")(x)
        print("DEBUG: After input projection, x shape:", x.shape)
        # Add positional encoding to capture temporal order
        pos_enc = positional_encoding(input_shape[0], embedding_dim)
        x = x + pos_enc
        print("DEBUG: After adding positional encoding, x shape:", x.shape)
        # Now x is (batch, window_size, embedding_dim)

        # Build transformer blocks (same number as intermediate_layers)
        for idx in range(self.params['intermediate_layers']):
            print(f"DEBUG: Building Transformer block {idx+1} with embedding dim {embedding_dim}")
            # Layer Normalization before attention
            x_norm = LayerNormalization(name=f"layer_norm_{idx+1}")(x)
            key_dim = max(1, embedding_dim // num_heads)
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                name=f"mha_layer_{idx+1}"
            )(x_norm, x_norm)
            print(f"DEBUG: After MultiHeadAttention in block {idx+1}, attn_output shape: {attn_output.shape}")
            # Residual connection for attention sub-layer
            x = Add(name=f"residual_add_attn_{idx+1}")([x, attn_output])
            # Feedforward network
            x_ff_norm = LayerNormalization(name=f"layer_norm_ff_{idx+1}")(x)
            ff_output = Dense(
                units=embedding_dim,
                activation=self.params['activation'],
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name=f"ff_dense_{idx+1}"
            )(x_ff_norm)
            print(f"DEBUG: After feedforward dense in block {idx+1}, ff_output shape: {ff_output.shape}")
            x = Add(name=f"residual_add_ff_{idx+1}")([x, ff_output])
            print(f"DEBUG: After Transformer block {idx+1}, x shape: {x.shape}")

        # Instead of GlobalAveragePooling1D:
        # x = GlobalAveragePooling1D(name="global_avg_pool")(x)
        # Try using Flatten:
        x = tf.keras.layers.Flatten(name="flatten")(x)


        x = BatchNormalization(name="batch_norm_final")(x)
        print("DEBUG: After final BatchNormalization, x shape:", x.shape)

        # --- Bayesian Output Layer Implementation (copied from CNN/LSTM plugins) ---
        def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
            return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
        tfp.layers.DenseFlipout.add_variable = _patched_add_variable

        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight:", self.params.get('kl_weight', 1e-3))

        def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
            print("DEBUG: In posterior_mean_field_custom:")
            print("       dtype =", dtype, "kernel_shape =", kernel_shape)
            print("       Received bias_size =", bias_size, "; overriding to 0")
            print("DEBUG: trainable =", trainable, "name =", name)
            if not isinstance(name, str):
                print("DEBUG: 'name' is not a string; setting to None")
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            print("DEBUG: posterior: computed n =", n)
            c = np.log(np.expm1(1.))
            print("DEBUG: posterior: computed c =", c)
            loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42),
                            dtype=dtype, trainable=trainable, name="posterior_loc")
            scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43),
                                dtype=dtype, trainable=trainable, name="posterior_scale")
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
            print("DEBUG: trainable =", trainable, "name =", name)
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
        
        KL_WEIGHT = self.params.get('kl_weight', 1e-3)
        DenseFlipout = tfp.layers.DenseFlipout
        print("DEBUG: Creating DenseFlipout final layer with units:", self.params['time_horizon'])
        flipout_layer = DenseFlipout(
            units=self.params['time_horizon'],
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
            name="output_layer"
        )
        bayesian_output = tf.keras.layers.Lambda(
            lambda t: flipout_layer(t),
            output_shape=lambda s: (s[0], self.params['time_horizon']),
            name="bayesian_dense_flipout"
        )(x)
        print("DEBUG: After DenseFlipout (via Lambda), bayesian_output shape:", bayesian_output.shape)
        
        bias_layer = Dense(
            units=1,
            activation='linear',
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias",
            kernel_regularizer=l2(l2_reg)
        )(x)
        print("DEBUG: Deterministic bias layer output shape:", bias_layer.shape)
        
        outputs = bayesian_output + bias_layer
        print("DEBUG: Final outputs shape after adding bias:", outputs.shape)
        
        # --- NEW CODE for Multi-Output adaptation ---

        split_layer = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=self.params['time_horizon'], axis=1),
            name="split_layer"
        )
        outputs_list = split_layer(outputs)
        outputs_list = [
            tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1), name=f"output_{i+1}")(o)
            for i, o in enumerate(outputs_list)
        ]
        print("DEBUG: Final model will output a list of tensors (one per horizon).")
        self.model = Model(inputs=inputs, outputs=outputs_list, name="predictor_model")
        # --- END NEW CODE ---

        time_horizon = self.params['time_horizon']
        # add 'mae' time_horizon times to the metrics
        metrics = ['mae' for _ in range(time_horizon)]

        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.0001)),
            loss=[self.custom_loss for _ in range(self.params['time_horizon'])],
            metrics=metrics
        )
        # --- END NEW CODE ---

        print("DEBUG: Adam optimizer created with learning_rate:", self.params.get('learning_rate', 0.0001))
        print("DEBUG: Model compiled with loss=Huber, metrics=['mae']")
        print("Ioin Model Summary:")
        self.model.summary()
        print("✅ Standard Transformer model built successfully.")

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
        total_loss = huber_loss + (self.mmd_lambda * mmd_loss) 
        return total_loss
    

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the Transformer model with MMD loss incorporated and logged at every epoch.
        """
        import tensorflow as tf

        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]

        print(f"Training with data => X: {x_train.shape}, Y: {[a.shape for a in y_train]}")
        exp_horizon = self.params['time_horizon']
        
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
                print(f"MMD Lambda = {self.plugin.mmd_lambda.numpy():.6f}, MMD Loss = {mmd_value.numpy():.6f}\n")

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
            shuffle=True,
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
        predictions = np.array([self.model(data, training=True).numpy() for _ in range(mc_samples)])
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates

    def predict(self, data):
        return self.model.predict(data)

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



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # Apply sin to even indices; cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]  # Shape: (1, position, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)

# ---------------------------
# Debugging usage example
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    # Example: window_size=24, num_features=8
    plugin.build_model(input_shape=(24, 8))
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
