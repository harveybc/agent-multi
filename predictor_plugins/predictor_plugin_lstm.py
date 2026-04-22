#!/usr/bin/env python
"""
Enhanced Multi-Branch Ioin Plugin using Keras for forecasting EUR/USD returns.

This plugin is designed to use the decomposed signals produced by the STL Preprocessor Plugin.
It assumes the input is a multi-channel time window where each channel corresponds to a decomposed component:
  - Trend component
  - Seasonal component
  - Noise (residual) component

The architecture is composed of three branches—each processing one channel through its own Dense sub-network.
The outputs of these branches are concatenated and fed to a final set of layers to produce the predicted return.
The loss is computed using a composite loss (Huber + MMD), and custom metrics (MAE and R²) are calculated
on the predicted return. This implementation is intended for the case when use_returns is True.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K
#bilstm
from tensorflow.keras.layers import Bidirectional, LSTM
import gc
import os
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add
# keras identity
from tensorflow.keras.layers import Identity
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
#reshape 
from tensorflow.keras.layers import GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras.layers import Reshape
from tqdm import tqdm
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization\

from tensorflow.keras.losses import Huber
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, composite_loss_basic, random_normal_initializer_44, composite_loss_noreturns, r2_metric
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor

# Module-level quiet flag (set via env or overridden at runtime)
import os as _os
_QUIET = _os.environ.get('PREDICTOR_QUIET', '0') == '1'

from .common.positional_encoding import positional_encoding



# Define TensorFlow local header output feedback variables(used from the composite loss function):
local_p_control=[]
local_i_control=[]
local_d_control=[]
local_feedback=[] # local feedback values for the model
# ---------------------------
# Custom Callbacks (same as before)
# ---------------------------


class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """Custom ReduceLROnPlateau callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        if self.verbose:
            print(f"ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """Custom EarlyStopping callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        if self.verbose:
            print(f"EarlyStopping patience counter: {self.patience_counter}")

class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()


# ---------------------------
# Custom Metrics and Loss Functions
# ---------------------------
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

def mae_magnitude(y_true, y_pred):
    """Compute MAE on the first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """Compute R² metric on the first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    SS_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    SS_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def compute_mmd(x, y, sigma=1.0, sample_size=32):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples."""
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


# --- Composite Loss Function ---
def composite_loss(y_true, y_pred,
                   # Arguments needed:
                   head_index,
                   mmd_lambda,
                   sigma,
                   p, i, d, # Control parameters for this head (tf.Variable)
                   list_last_signed_error, # List of tf.Variables for metric storage
                   list_last_stddev,       # List of tf.Variables for metric storage
                   list_last_mmd,          # List of tf.Variables for metric storage
                   list_local_feedback     # List of tf.Variables for control action storage/feedback
                   ):
    """
    Global composite loss function for a specific head.
    Calculates metrics (signed_error, stddev, mmd).
    Calls global dummy_feedback_control with metrics and PID params.
    Assigns control function output to list_local_feedback[head_index].
    Also assigns metrics to list_last_xxx[head_index].
    Returns the scalar loss value (MSE + Asymptote + MMD).
    """
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, 1])
    mag_true = y_true
    mag_pred = y_pred

    # --- Calculate Primary Losses ---
    #mse_loss_val = tf.keras.losses.MeanSquaredError()(mag_true, mag_pred)
    huber_loss_val = Huber(delta=1.0)(mag_true, mag_pred)
    #mse_loss_val = huber_loss_val
    #mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    #mmd_loss_val = 0.0


    #mse_min = tf.maximum(huber_loss_val, 1e-10)
    #mse_min = tf.maximum(mse_loss_val, 1e-10)

    # --- Calculate Summary Statistics ---
    #signed_avg_pred = tf.reduce_mean(mag_pred)
    #signed_avg_true = tf.reduce_mean(mag_true)

    # --- Calculate Dynamic Asymptote Penalty (Original User Logic) ---
    #def vertical_dynamic_asymptote(value, center):
    #    res = tf.cond(tf.greater_equal(value, center),
    #        lambda: 3*tf.math.log(tf.abs(value - center) + 1e-9)+20,
    #        lambda: mse_loss_val*1e3 - 1)
    #    res = tf.cond(tf.greater_equal(center, value),
    #        lambda: mse_loss_val*1e3 - 1,
    #        lambda: 3*tf.math.log(tf.abs(value - center) + 1e-9)+20)
    #    return res
    #asymptote = vertical_dynamic_asymptote(signed_avg_pred, signed_avg_true)
    #asymptote = 0.0

    # --- Calculate Feedback Metrics ---
    #feedback_signed_error = 0.0
    #feedback_stddev = 0.0
    #feedback_mmd = 0.0

    # --- Call Control Function ---
    # Pass feedback metrics and head-specific PID parameters (which are tf.Variables)
    #local_control_action = dummy_feedback_control(
    #    feedback_signed_error, feedback_stddev, feedback_mmd, p, i, d
    #)

    # --- Update Feedback Variables (using control dependencies) ---
    #update_ops = [
        # Store calculated metrics
    #    list_last_signed_error[head_index].assign(signed_avg_true-signed_avg_pred),
    #    list_last_stddev[head_index].assign(feedback_stddev),
    #    list_last_mmd[head_index].assign(feedback_mmd),
        # Store the output of the control function - THIS IS THE FEEDBACK FOR THE MODEL
        #list_local_feedback[head_index].assign(local_control_action)
    #]

    #with tf.control_dependencies(update_ops):
        # Calculate final loss term
        #total_loss = 1e4 * mse_min + asymptote + mmd_lambda * mmd_loss_val
    #total_loss = 1e4 * mse_min + asymptote + mmd_lambda * mmd_loss_val
    #total_loss = huber_loss_val+ mmd_lambda * mmd_loss_val
    total_loss = huber_loss_val
    # Return the final scalar loss value
    return total_loss



# --- Can be defined inside the class or outside ---
@tf.function
def dummy_feedback_control(signed_error, stddev, mmd, p, i, d):
    """
    Global dummy control function using only TF ops.
    Takes head feedback values and PID parameters. Returns a placeholder action.
    """
    # Placeholder: Just return the 'p' value.
    control_action = tf.identity(p)
    return control_action


# --- Named initializer to avoid lambda serialization warnings ---
def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# Build Dense branches for trend, seasonal, and noise channels.
def build_branch(branch_input, branch_name, num_branch_layers=2, branch_units=32, activation='relu', l2_reg=1e-5):
    x = Flatten(name=f"{branch_name}_flatten")(branch_input)
    for i in range(num_branch_layers):
        x = Dense(branch_units, activation=activation,
                kernel_regularizer=l2(l2_reg),
                name=f"{branch_name}_dense_{i+1}")(x)
    return x

# Assume necessary imports like tensorflow, numpy, tfp are available

def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
    """Custom posterior distribution function for DenseFlipout kernel."""
    # print(f"DEBUG: posterior_mean_field_custom (name={name}):") # Optional: Keep top-level debug print
    # print(f"       dtype={dtype}, kernel_shape={kernel_shape}, bias_size={bias_size} (overridden to 0), trainable={trainable}") # Optional
    if not isinstance(name, str): name = None # Ensure name is string or None
    bias_size = 0 # Force bias size to 0 for kernel posterior
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.))
    # Use unique variable names based on the layer name if provided
    loc_name = f"{name}_loc" if name else "posterior_loc"
    scale_name = f"{name}_scale" if name else "posterior_scale"
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name=loc_name)
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name=scale_name)
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)

    # --- CORRECTED PRINT STATEMENTS ---
    # Removed access to .name attribute
    if not _QUIET: print(f"DEBUG: posterior: created loc shape: {loc.shape}")
    if not _QUIET: print(f"DEBUG: posterior: created scale shape: {scale.shape}")
    # --- END CORRECTION ---

    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
        # print(f"DEBUG: posterior: reshaped loc to {loc_reshaped.shape} and scale to {scale_reshaped.shape}") # Optional
    except Exception as e:
        print(f"ERROR: Exception during reshape in posterior (name={name}):", e)
        raise e
    # Ensure reinterpreted_batch_ndims matches the rank of the kernel shape
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

# NOTE: Also double-check your 'prior_fn' function definition. If it contains
# similar print statements accessing '.name', remove those as well, although
# based on previous versions, it likely does not.

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    """Custom prior distribution function for DenseFlipout kernel."""
    if not _QUIET: print(f"DEBUG: prior_fn (name={name}):")
    if not _QUIET: print(f"       dtype={dtype}, kernel_shape={kernel_shape}, bias_size={bias_size} (overridden to 0), trainable={trainable}")
    if not isinstance(name, str): name = None # Ensure name is string or None
    bias_size = 0 # Force bias size to 0 for kernel prior
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    if not _QUIET: print(f"DEBUG: prior_fn: computed n={n}")
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
        if not _QUIET: print(f"DEBUG: prior_fn: reshaped loc to {loc_reshaped.shape} and scale to {scale_reshaped.shape}")
    except Exception as e:
        if not _QUIET: print(f"DEBUG: Exception during reshape in prior_fn (name={name}):", e)
        raise e
    # Ensure reinterpreted_batch_ndims matches the rank of the kernel shape
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

# ---------------------------
# Multi-Branch Ioin Plugin Class Definition
# ---------------------------
class Plugin:
    """
    Enhanced Multi-Branch Ioin Plugin.

    This plugin builds a multi-branch model to process STL-decomposed input channels:
      - One branch processes the trend channel.
      - One branch processes the seasonal channel.
      - One branch processes the noise (residual) channel.

    Each branch passes its input through dedicated Dense layers.
    Their outputs are concatenated and then combined with the flattened error and std feedback channels
    (which bypass further Dense processing) and passed to a merged Dense layer.
    The final output is produced by a Bayesian layer (tfp.layers.DenseFlipout) plus a deterministic bias layer,
    so that uncertainty estimates can be derived.
    """
    plugin_params = {
        'batch_size': 32,
        'num_branch_layers': 2,      # Number of Dense layers in each branch
        'branch_units': 32,          # Units in each branch layer
        'merged_units': 64,          # Units in the merged network
        'learning_rate': 0.0001,
        'activation': 'relu',
        'l2_reg': 1e-5,
        'mmd_lambda': 1e-3,
        'time_horizon': 6           # Forecast horizon (in hours)
    }
    plugin_debug_vars = ['batch_size', 'num_branch_layers', 'branch_units', 'merged_units', 'learning_rate', 'l2_reg', 'time_horizon']

    def __init__(self, config):
        """
        Initialize the ioin plugin, including feedback/control lists.
        """
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required for initialization.")

        # Store parameters, update with config
        self.plugin_params = {
            "l2_reg": 0.001, "activation": "relu", "branch_units": 64,
            "merged_units": 128, "learning_rate": 0.001, "mmd_lambda": 0.1,
            "sigma_mmd": 1.0, "predicted_horizons": [1] # Default horizon if not in config
        }
        self.params = self.plugin_params.copy()
        if config:
           self.params.update(config)
        self.quiet = self.params.get('quiet', False)

        # Ensure predicted_horizons exists after potential update
        if 'predicted_horizons' not in self.params:
             raise ValueError("Config must contain 'predicted_horizons' list.")
        predicted_horizons = self.params['predicted_horizons']
        num_outputs = len(predicted_horizons)

        self.model = None
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')

        # --- Initialize Control Parameter & Feedback Lists ---
        #print(f"Initializing control and feedback lists for {num_outputs} outputs in __init__.")

        # Control Parameters per Head (Values are examples)
        self.local_p_control = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"local_p_{i}") for i in range(num_outputs)]
        self.local_i_control = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"local_i_{i}") for i in range(num_outputs)]
        self.local_d_control = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"local_d_{i}") for i in range(num_outputs)]

        # Feedback Metrics Storage per Head (updated by loss)
        self.last_signed_error = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"last_signed_error_{i}") for i in range(num_outputs)]
        self.last_stddev = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"last_stddev_{i}") for i in range(num_outputs)]
        self.last_mmd = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"last_mmd_{i}") for i in range(num_outputs)]

        # Feedback Action Storage per Head (output of control func, INPUT to model) - NEW/REVISED
        # Shape depends on output of dummy_feedback_control. If it returns scalar P, shape is scalar.
        self.local_feedback = [tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"local_feedback_{i}") for i in range(num_outputs)]

        if not _QUIET: print("Control/Feedback lists initialized.")

        # --- Apply DenseFlipout Patch ---
        if not hasattr(tfp.layers.DenseFlipout, '_already_patched_add_variable'):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable
            tfp.layers.DenseFlipout._already_patched_add_variable = True
            if not self.quiet: print("DEBUG: DenseFlipout patched successfully in __init__.")
        else:
            if not self.quiet: print("DEBUG: DenseFlipout already patched.")


    def set_params(self, **kwargs):
        """Update ioin plugin parameters with provided configuration."""
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """Return debug information for the ioin plugin."""
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add ioin plugin debug information to the given dictionary."""
        debug_info.update(self.get_debug_info())


    # --- Define within your YourPredictorPlugin class ---
    # --- Define within your YourPredictorPlugin class ---
    def build_model(self, input_shape, x_train, config):
        """
        Build model: processes features, merges, feeds into heads.
        """
        if not _QUIET: print(f"DEBUG: Entering build_model with input_shape={input_shape}", flush=True)
        # --- Pre-checks ---
        if self.local_feedback is None or self.local_p_control is None:
             raise RuntimeError("Feedback/Control lists were not initialized in __init__.")

        window_size, num_channels = input_shape
        predicted_horizons = config['predicted_horizons']
        num_outputs = len(predicted_horizons)
        if len(self.local_feedback) != num_outputs:
             raise RuntimeError(f"Initialized list length != num_outputs. Re-initialize?")

        # --- Get Parameters ---
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 0.001))
        activation = config.get("activation", self.params.get("activation", "relu"))
        num_intermediate_layers = config['intermediate_layers']
        num_head_intermediate_layers = config['intermediate_layers']
        merged_units = config.get("initial_layer_size", 128)
        branch_units = merged_units//config.get("layer_size_divisor", 2)
        # Add LSTM units parameter (provide a default)
        lstm_units = branch_units//config.get("layer_size_divisor", 2) # New parameter for LSTM size
        embedding_dim = merged_units
        # --- Define Bayesian Layer Components ---
        KL_WEIGHT = self.kl_weight_var
        DenseFlipout = tfp.layers.DenseFlipout
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)

        # --- Input Layer ---
        if not _QUIET: print("DEBUG: Create Input Layer", flush=True)
        inputs = Input(shape=(window_size, num_channels), name="input_layer")
        x = inputs
        
                # Add positional encoding to capture temporal order
        # get static shape tuple via Keras backend
        last_layer_shape = K.int_shape(x)
        feature_dim = last_layer_shape[-1]
        # get the sequence length from the last layer shape
        seq_length = last_layer_shape[1]
        pos_enc = positional_encoding(seq_length, feature_dim)
        x = x + pos_enc
        if not _QUIET: print("DEBUG: Positional encoding added", flush=True)

        # --- Self-Attention Block 1 ---
        num_attention_heads = 2
        # get the last layer shape from the merged tensor
        last_layer_shape = K.int_shape(x)
        # get the feature dimension from the last layer shape as the last component of the shape tuple
        feature_dim = last_layer_shape[-1]
        if not _QUIET: print(f"DEBUG: feature_dim={feature_dim}, num_attention_heads={num_attention_heads}", flush=True)

        # define key dimension for attention    
        attention_key_dim = feature_dim//num_attention_heads
        if not _QUIET: print(f"DEBUG: attention_key_dim={attention_key_dim}", flush=True)
        if attention_key_dim == 0:
             if not self.quiet: print("DEBUG: attention_key_dim is 0, forcing to 1", flush=True)
             attention_key_dim = 1

        # Apply MultiHeadAttention
        attention_output = MultiHeadAttention(
            num_heads=num_attention_heads, # Assumed to be defined
            key_dim=attention_key_dim,      # Assumed to be defined
            kernel_regularizer=l2(l2_reg),
            name=f"multihead_attention_1"
        )(query=x, value=x, key=x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        #AveragePooling1D
        x = AveragePooling1D(pool_size=3, strides=2, padding='same', name=f"average_pooling_1")(x)
        if not _QUIET: print("DEBUG: Self-Attention Block 1 done", flush=True)

        # --- End Self-Attention Block ---
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg),
                    name=f"feature_lstm_1"))(x)

        # --- End Self-Attention Block ---
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg),
                    name=f"feature_lstm_2"))(x)
        
        x = AveragePooling1D(pool_size=3, strides=2, padding='same', name=f"average_pooling_2")(x)

        merged = x
        if not _QUIET: print("DEBUG: Merged features ready", flush=True)

        # --- Build Multiple Output Heads ---
        outputs_list = []
        self.output_names = []

        for i, horizon in enumerate(predicted_horizons):
            if not self.quiet: print(f"DEBUG: Building head for horizon {horizon}", flush=True)
            branch_suffix = f"_h{horizon}"

            # --- Head Intermediate Dense Layers ---
            head_dense_output = merged
            #for j in range(num_head_intermediate_layers):
            #     head_dense_output = Dense(merged_units, activation=activation, kernel_regularizer=l2(l2_reg),
            #                               name=f"head_dense_{j+1}{branch_suffix}")(head_dense_output)

            # --- Add BiLSTM Layer ---
            # Reshape Dense output to add time step dimension: (batch, 1, merged_units) (BEST ONE)
            # TODO: probar (batch, merged_units, 1)
            #reshaped_for_lstm = Reshape((merged_units, 1), name=f"reshape_lstm{branch_suffix}")(head_dense_output) 
            reshaped_for_lstm = head_dense_output
            reshaped_for_lstm = Conv1D(filters=branch_units, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(l2_reg), name=f"conv1d_1{branch_suffix}")(reshaped_for_lstm)
            reshaped_for_lstm = Conv1D(filters=lstm_units, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(l2_reg), name=f"conv1d_2{branch_suffix}")(reshaped_for_lstm)
            # Apply Bidirectional LSTM
            # return_sequences=False gives output shape (batch, 2 * lstm_units)
            lstm_output = Bidirectional(
                LSTM(lstm_units, return_sequences=False), name=f"bidir_lstm{branch_suffix}"
            )(reshaped_for_lstm)
          

          
            # --- Bayesian / Bias Layers ---
            if not self.quiet: print(f"DEBUG: Building Flipout layer {horizon}", flush=True)
            flipout_layer_name = f"bayesian_flipout_layer{branch_suffix}"
            flipout_layer_branch = DenseFlipout(
                units=1, activation='linear',
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flipout_layer_name: posterior_mean_field_custom(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flipout_layer_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT, name=flipout_layer_name
            )
            # Apply DenseFlipout layer via Lambda WITH output_shape specified
            bayesian_output_branch = Lambda(
                lambda t: flipout_layer_branch(t),
                output_shape=lambda s: (s[0], 1), # Explicit output shape
                name=f"bayesian_output{branch_suffix}"
            )(lstm_output)

            bias_layer_branch = Dense(units=1, activation='linear', kernel_initializer=random_normal_initializer_44,
                                      name=f"deterministic_bias{branch_suffix}")(lstm_output)

            # --- Final Head Output ---
            output_name = f"output_horizon_{horizon}"
            final_branch_output = Add(name=output_name)([bayesian_output_branch, bias_layer_branch])

            outputs_list.append(final_branch_output)
            self.output_names.append(output_name) # Store the name
            # --- End of Head ---
        if not _QUIET: print("DEBUG: All heads built", flush=True)

        # --- Model Definition ---
        self.model = Model(inputs=inputs, outputs=outputs_list, name=f"ControlFeedbackPredictor_{len(predicted_horizons)}H")

        # --- Compilation (Using GLOBAL composite_loss) ---
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
        if not _QUIET: print("DEBUG: Compiling model...", flush=True)
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        if not _QUIET: print("DEBUG: Model compiled. Printing summary...", flush=True)
        self.model.summary(line_length=140)
        if not _QUIET: print("DEBUG: build_model completed.", flush=True)


    # --- Method within YourPredictorPlugin class ---
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the multi-output model using provided data and configuration.

        Expects y_train and y_val to be dictionaries mapping output layer names
        to their corresponding target numpy arrays (e.g., shape [num_samples, 1]).
        Utilizes KL annealing and other callbacks during training.
        Calculates final metrics based on the specific output head designated
        by config['plotted_horizon'].

        Args:
            x_train (np.ndarray): Training input features.
            y_train (dict): Dictionary of training target arrays for each output head.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            threshold_error: (Not used in provided snippet, kept for signature consistency).
            x_val (np.ndarray): Validation input features.
            y_val (dict): Dictionary of validation target arrays for each output head.
            config (dict): Configuration dictionary for training parameters, MUST contain
                           'predicted_horizons' (list) and 'plotted_horizon' (int).

        Returns:
            tuple: Contains history object, list of train predictions per head,
                   list of train uncertainties (placeholders), list of validation
                   predictions per head, list of validation uncertainties (placeholders).

        Raises:
            ValueError: If config is missing required keys or if 'plotted_horizon'
                        is not found within 'predicted_horizons'.
            TypeError: If y_train or y_val are not dictionaries.
            AttributeError: If self.output_names was not set by build_model.
        """
        # --- Configuration Validation ---
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required for training.")
        if 'predicted_horizons' not in config:
            raise ValueError("Config dictionary must contain the key 'predicted_horizons' (list of ints).")
        if 'plotted_horizon' not in config:
            raise ValueError("Config dictionary must contain the key 'plotted_horizon' (int).")

        predicted_horizons = config['predicted_horizons']
        plotted_horizon = config['plotted_horizon'] # Horizon used for final metric reporting

        # Validate that the plotted_horizon is one of the predicted horizons
        if plotted_horizon not in predicted_horizons:
            raise ValueError(
                f"Invalid configuration: 'plotted_horizon' ({plotted_horizon}) "
                f"is not present in the 'predicted_horizons' list ({predicted_horizons}). "
                f"Please ensure 'plotted_horizon' matches one of the values in 'predicted_horizons'."
            )

        # Find the index corresponding to the plotted horizon
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
        except ValueError:
             # This case should be caught by the 'in' check above, but added for robustness
             raise ValueError(f"'plotted_horizon' {plotted_horizon} not found in {predicted_horizons} (index error).")


        # --- Inner Class for KL Annealing Callback ---
        class KLAnnealingCallback(tf.keras.callbacks.Callback):
            # ... (keep implementation as provided) ...
            def __init__(self, plugin, target_kl, anneal_epochs):
                super().__init__()
                self.plugin = plugin
                self.target_kl = target_kl
                self.anneal_epochs = anneal_epochs
            def on_epoch_begin(self, epoch, logs=None):
                new_kl = self.target_kl * min(1.0, (epoch + 1) / self.anneal_epochs)
                self.plugin.kl_weight_var.assign(new_kl)

        # --- Setup Callbacks ---
        anneal_epochs = config.get("kl_anneal_epochs", self.params.get("kl_anneal_epochs", 10))
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        min_delta_early_stopping = config.get("min_delta", self.params.get("min_delta", 1e-4))
        patience_early_stopping = self.params.get('early_patience', 10)
        start_from_epoch_es = self.params.get('start_from_epoch', 10)
        patience_reduce_lr = config.get("reduce_lr_patience", max(1, int(patience_early_stopping / 4)))

        # Verbosity: quiet mode suppresses per-step progress bars and per-epoch LR prints
        quiet_mode = config.get("quiet", self.params.get("quiet", False))
        fit_verbose = 0 if quiet_mode else 1
        cb_verbose = 0 if quiet_mode else 1

        # Instantiate callbacks WITHOUT ClearMemoryCallback
        # Assumes relevant Callback classes are imported/defined
        callbacks = [
            EarlyStoppingWithPatienceCounter(
                monitor='val_loss', patience=patience_early_stopping, restore_best_weights=True,
                verbose=cb_verbose, start_from_epoch=start_from_epoch_es, min_delta=min_delta_early_stopping
            ),
            ReduceLROnPlateauWithCounter(
                monitor="val_loss", factor=0.5, patience=patience_reduce_lr, cooldown=5, min_delta=min_delta_early_stopping, verbose=cb_verbose
            ),
            kl_callback
        ]
        if not quiet_mode:
            callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs:
                           print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")))

        # --- Input Data Verification ---
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
             raise TypeError("y_train and y_val must be dictionaries.")
        if not hasattr(self, 'output_names') or not self.output_names:
             raise AttributeError("self.output_names not set by build_model.")
        plotted_output_name = f"output_horizon_{plotted_horizon}"
        if plotted_output_name not in y_train or plotted_output_name not in y_val:
             raise ValueError(f"Target dicts missing key: '{plotted_output_name}'")
        # Optional: Check all keys match
        if set(y_train.keys()) != set(self.output_names) or set(y_val.keys()) != set(self.output_names):
             if not self.quiet: print("WARN: Target data dictionary keys may not perfectly match all model output names.")

        # --- Model Training ---
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=callbacks,
                                 verbose=fit_verbose)

        # --- Post-Training Predictions ---
        # Note: Predicting on full train/val sets uses memory. Consider alternatives if needed.
        #list_train_preds = self.model.predict(x_train, batch_size=batch_size)
        #list_val_preds = self.model.predict(x_val, batch_size=batch_size)
        mc_samples = config.get("mc_samples", 100)
        list_train_preds,list_train_uncertainty  = self.predict_with_uncertainty(x_train, mc_samples)
        list_val_preds, list_val_uncertainty = self.predict_with_uncertainty(x_val, mc_samples)   

        # Placeholder uncertainties (as these weren't generated during training)
        #list_train_uncertainty = [np.zeros_like(preds) for preds in list_train_preds]
        #list_val_uncertainty = [np.zeros_like(preds) for preds in list_val_preds]

        # --- Post-Training Metrics (for the configured 'plotted_horizon') ---
        # Assumes self.calculate_mae and self.calculate_r2 methods exist
        try:
            y_train_plotted = y_train[plotted_output_name]
            train_preds_plotted = list_train_preds[plotted_index] # Use pre-calculated index
            if not self.quiet: print(f"Calculating final MAE/R2 for plotted horizon: {plotted_horizon} (Index: {plotted_index})")
            if hasattr(self, 'calculate_mae') and callable(self.calculate_mae):
                self.calculate_mae(y_train_plotted, train_preds_plotted)
            if hasattr(self, 'calculate_r2') and callable(self.calculate_r2):
                self.calculate_r2(y_train_plotted, train_preds_plotted)
        except Exception as e:
             print(f"ERROR during post-training metric calculation: {e}")

        # Return history and lists of predictions/uncertainties
        return history, list_train_preds, list_train_uncertainty, list_val_preds, list_val_uncertainty
    

    # --- Method within PredictorPluginANN class ---
    # --- Method within PredictorPluginANN class ---
    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Performs Monte Carlo dropout predictions for the multi-output model
        using an incremental approach to avoid large memory allocation.

        Runs the model multiple times with dropout enabled (training=True)
        to estimate predictive uncertainty (standard deviation) for each output head.

        Args:
            x_test (np.ndarray): Input data for prediction.
            mc_samples (int): Number of Monte Carlo samples to perform.

        Returns:
            tuple: (list_mean_predictions, list_uncertainty_estimates)
                   Lists containing numpy arrays (one per output head)
                   for mean predictions and standard deviations (uncertainty).
                   Shape of each array: [num_samples, output_dim (usually 1)].
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        if mc_samples <= 0:
            return [], []

        # Get dimensions from a single sample run
        try:
            first_run_output_tf = self.model(x_test[:1], training=True) # Predict on one sample
            if not isinstance(first_run_output_tf, list): first_run_output_tf = [first_run_output_tf]
            num_heads = len(first_run_output_tf)
            if num_heads == 0: return [], []
            first_head_output = first_run_output_tf[0].numpy()
            num_test_samples = x_test.shape[0]
            output_dim = first_head_output.shape[1] if first_head_output.ndim > 1 else 1
        except Exception as e:
            print(f"ERROR getting model output shape in predict_with_uncertainty: {e}")
            raise ValueError("Could not determine model output structure.") from e

        # Initialize accumulators for mean and variance calculation (Welford's algorithm components)
        # Using lists to store per-head accumulators
        means = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        m2s = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        counts = [0] * num_heads # Use a single count across heads, assuming samples are drawn together

        # print(f"Running {mc_samples} MC samples for uncertainty (incremental)...") # Informative print
        for i in tqdm(range(mc_samples), desc="MC Samples"):
            # Get predictions for all heads in this sample
            batch_size = 256  # ✅ Use safe batch size
            ## Initialize a list for each output head
            head_outputs_lists = None
            for i in range(0, len(x_test), batch_size):
                batch_x = x_test[i:i + batch_size]
                preds = self.model(batch_x, training=False)
                if not isinstance(preds, list):
                    preds = [preds]
                if head_outputs_lists is None:
                    head_outputs_lists = [[] for _ in range(len(preds))]
                for h, pred in enumerate(preds):
                    head_outputs_lists[h].append(pred)


            # Concatenate outputs for each head along the batch dimension
            head_outputs_tf = [tf.concat(head_list, axis=0) for head_list in head_outputs_lists]



            if not isinstance(head_outputs_tf, list): head_outputs_tf = [head_outputs_tf]

            # Process each head's output for this sample
            for h in range(num_heads):
                head_output_np = head_outputs_tf[h].numpy()
                # Reshape if necessary
                if head_output_np.ndim == 1:
                    head_output_np = np.expand_dims(head_output_np, axis=-1)
                if head_output_np.shape != (num_test_samples, output_dim):
                     raise ValueError(f"Shape mismatch in MC sample {i}, head {h}: Expected {(num_test_samples, output_dim)}, got {head_output_np.shape}")

                # Welford's online algorithm update
                counts[h] += 1
                delta = head_output_np - means[h]
                means[h] += delta / counts[h]
                delta2 = head_output_np - means[h] # New delta using updated mean
                m2s[h] += delta * delta2

            # Optional progress print
            # if (i + 1) % (mc_samples // 10 or 1) == 0: print(f"  MC sample {i+1}/{mc_samples}")

        # Finalize calculations: variance = M2 / (n - 1), stddev = sqrt(variance)
        list_mean_predictions = means # The mean is already calculated
        list_uncertainty_estimates = []
        for h in range(num_heads):
             if counts[h] < 2: # Need at least 2 samples for variance/stddev
                 variance = np.full((num_test_samples, output_dim), np.nan, dtype=np.float32)
             else:
                 variance = m2s[h] / (counts[h] - 1)
             stddev = np.sqrt(np.maximum(variance, 0)) # Ensure variance isn't negative due to float issues
             list_uncertainty_estimates.append(stddev.astype(np.float32))

        # print("MC sampling finished.") # Informative print
        return list_mean_predictions, list_uncertainty_estimates
    
    
    def save(self, file_path):
        self.model.save(file_path)
        if not self.quiet: print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={
            "composite_loss": composite_loss,
            "compute_mmd": compute_mmd,
            "r2_metric": r2_metric,
            "mae_magnitude": mae_magnitude
        })
        if not self.quiet: print(f"Ioin model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        if not _QUIET: print(f"DEBUG: y_true (sample): {mag_true.flatten()[:5]}")
        if not _QUIET: print(f"DEBUG: y_pred (sample): {mag_pred.flatten()[:5]}")
        mae = np.mean(np.abs(mag_true.flatten() - mag_pred.flatten()))
        if not self.quiet: print(f"Calculated MAE (magnitude): {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        if not self.quiet: print(f"Calculating R²: y_true shape={mag_true.shape}, y_pred shape={mag_pred.shape}")
        SS_res = np.sum((mag_true - mag_pred) ** 2, axis=0)
        SS_tot = np.sum((mag_true - np.mean(mag_true, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (SS_res / (SS_tot + np.finfo(float).eps))
        r2 = np.mean(r2_scores)
        if not self.quiet: print(f"Calculated R² (magnitude): {r2}")
        return r2

# ---------------------------
# Debugging usage example (if run as main)
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    # For debugging, assume input shape (window_size, num_channels) where num_channels=3.
    # Example: window_size=24, 3 channels (trend, seasonal, noise).
    plugin.build_model(input_shape=(24, 3), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    if not _QUIET: print(f"Debug Info: {debug_info}")