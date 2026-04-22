import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score

class Plugin:
    """
    A ioin plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,     # L2 regularization factor
        'activation': 'tanh'
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'intermediate_layers', 'initial_layer_size', 'time_horizon']

    def __init__(self):
        """
        Initializes the Plugin with default parameters and no model.
        """
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Updates the plugin parameters with provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to update plugin parameters.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Retrieves the current values of debug variables.
        
        Returns:
            dict: Dictionary containing debug information.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds the plugin's debug information to an external debug_info dictionary.
        
        Args:
            debug_info (dict): External dictionary to update with debug information.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build a CNN-based model with sliding window input.

        Parameters:
            input_shape (tuple): Shape of the input data (window_size, features).
        """
        if len(input_shape) != 2:
            raise ValueError(f"Invalid input_shape {input_shape}. CNN requires input with shape (window_size, features).")

        self.params['input_shape'] = input_shape
        print(f"CNN input_shape: {input_shape}")

        layers = []
        current_size = self.params['initial_layer_size']
        l2_reg = self.params.get('l2_reg', 1e-4)
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        # Output layer size is time_horizon
        layers.append(self.params['time_horizon'])

        # Debugging message
        print(f"CNN Layer sizes: {layers}")

        # Define the Input layer
        inputs = Input(shape=input_shape, name="model_input")
        x = inputs
        x = Dense(
            units=layers[0],
            activation=self.params['activation'],
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
        )(x)
        # Add intermediate Conv1D and MaxPooling1D layers
        for idx, size in enumerate(layers[:-1]):
            if size > 1:
                x = Conv1D(
                    filters=size, 
                    kernel_size=3, 
                    activation='relu', 
                    kernel_initializer=HeNormal(), 
                    padding='same',
                    kernel_regularizer=l2(self.params.get('l2_reg', 1e-4)),
                    name=f"conv1d_{idx+1}"
                )(x)
                x = MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)
        x = Dense(
            units=size,
            activation=self.params['activation'],
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
        )(x)

        #add batch normalization
        x = BatchNormalization()(x)
        
        # Flatten the output from Conv layers
        x = Flatten(name="flatten")(x)
                
        # Output layer => shape (N, time_horizon)
        model_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)


        # Create the Model
        self.model = Model(inputs=inputs, outputs=model_output, name="cnn_model")

        # Define the Adam optimizer
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        # Compile the model with Huber loss and evaluation metrics
        self.model.compile(
            optimizer=adam_optimizer, 
            loss=Huber(), 
            metrics=['mse','mae'], 
            run_eagerly=False  # Set to False for better performance unless debugging
        )

        # Debugging messages to trace the model configuration
        print("CNN Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the CNN model with Early Stopping to prevent overfitting.

        Parameters:
            x_train (numpy.ndarray): Training input data with shape (samples, window_size, features).
            y_train (numpy.ndarray): Training target data with shape (samples, time_horizon).
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            threshold_error (float): Threshold for loss to trigger warnings.
            x_val (numpy.ndarray, optional): Validation input data.


            y_val (numpy.ndarray, optional): Validation target data.
        """
        if x_train.ndim != 3:
            raise ValueError(f"x_train must be 3D with shape (samples, window_size, features). Found: {x_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(
                f"y_train shape {y_train.shape}, expected (N,{exp_horizon})."
            )
        callbacks = []

        # Early Stopping based on loss or validation loss
        patience = self.params.get('early_patience', 25)  # default patience is 10 epochs
        monitor_metric = 'val_loss'
        early_stopping_monitor = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)

        print(f"Training CNN model with data shape: {x_train.shape}, target shape: {y_train.shape}")

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_split = 0.2
        )
        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Force the model to run in "training mode"
        print("Forcing training mode for MAE calculation...")
        preds_training_mode = self.model(x_train, training=True).numpy()
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train[:len(preds_training_mode)]))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        # Compare with evaluation mode
        print("Forcing evaluation mode for MAE calculation...")
        preds_eval_mode = self.model(x_train, training=False).numpy()
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train[:len(preds_training_mode)]))
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        # Evaluate on the full training dataset for consistency
        print("Evaluating on the full training dataset...")
        train_eval_results = self.model.evaluate(x_train, y_train[:len(preds_training_mode)], batch_size=batch_size, verbose=0)
        train_loss, train_mse, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MSE: {train_mse}, MAE: {train_mae}")
        
        # Only evaluate validation data if it exists
        if x_val is not None and y_val is not None:
            val_eval_results = self.model.evaluate(x_val, y_val[:x_val.shape[0]], batch_size=batch_size, verbose=0)
            _, _, val_mae = val_eval_results
            val_predictions = self.predict(x_val)  # Predict validation data
            val_r2 = r2_score(y_val[:x_val.shape[0]], val_predictions)
        else:
            val_mae = None
            val_r2 = None
            val_predictions = None

        # Predict training data for evaluation
        train_predictions = self.predict(x_train)  # Predict train data

        # Calculate R² scores - adjust target shapes to match predictions
        train_r2 = r2_score(y_train[:x_train.shape[0]], train_predictions)
        val_r2 = None if val_predictions is None else r2_score(y_val[:x_val.shape[0]], val_predictions)
        
        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions



    def predict(self, data):
        """
        Generate predictions using the trained CNN model.

        Parameters:
            data (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted outputs.
        """
        # CNN expects data to be (samples, window_size, features)

        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
        return predictions


    def calculate_mse(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error (MSE) between true and predicted values.
        
        Args:
            y_true (numpy.ndarray): True target values of shape (N, time_horizon).
            y_pred (numpy.ndarray): Predicted target values of shape (N, time_horizon).
        
        Returns:
            float: Calculated MSE.
        
        Raises:
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        print(f"Calculating MSE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mse: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        
        # Flatten the arrays to 1D for MSE calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")
        
        # Calculate Mean Squared Error
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Calculates the Mean Absolute Error (MAE) between true and predicted values.
        
        Args:
            y_true (numpy.ndarray): True target values of shape (N, time_horizon).
            y_pred (numpy.ndarray): Predicted target values of shape (N, time_horizon).
        
        Returns:
            float: Calculated MAE.
        
        Raises:
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        print(f"Calculating MAE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mae: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        
        # Flatten the arrays to 1D for MAE calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")
        
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        #print(f"Calculated MAE: {mae}")
        return mae


    def save(self, file_path):
        """
        Saves the trained model to the specified file path.

        Args:
            file_path (str): Path to save the model.
        """
        save_model(self.model, file_path)
        print(f"Ioin model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a trained model from the specified file path.

        Args:
            file_path (str): Path to load the model from.
        """
        self.model = load_model(file_path)
        print(f"Ioin model loaded from {file_path}")

    def calculate_r2(self, y_true, y_pred):
        """
        Calculates the R² (Coefficient of Determination) score between true and predicted values.

        Args:
            y_true (numpy.ndarray): True target values of shape (N, time_horizon).
            y_pred (numpy.ndarray): Predicted target values of shape (N, time_horizon).

        Returns:
            float: Calculated R² score.

        Raises:
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_r2: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )

        # Calculate R² score for each sample and then average
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
        r2_scores = 1 - (ss_res / ss_tot)

        # Handle cases where ss_tot is zero
        r2_scores = np.where(ss_tot == 0, 0, r2_scores)

        # Calculate the average R² score
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
        return r2


# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=(24, 8))  # Example input_shape for CNN
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
