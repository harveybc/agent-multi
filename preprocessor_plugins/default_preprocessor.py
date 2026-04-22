#!/usr/bin/env python
"""
Default Preprocessor Plugin

Este plugin se encarga del preprocesamiento de datos para la predicción de EUR/USD.
Implementa el método process_data que:
  1. Carga archivos CSV según las rutas definidas en la configuración.
  2. Extrae la columna 'typical_price' como serie univariada.
  3. Genera ventanas deslizantes para la predicción de un único paso,
     aplicando opcionalmente la conversión a retornos.
  4. Reestructura los datos para que sean compatibles con los modelos (reshape a 3D para redes recurrentes, CNN, etc.).

Además, se incluye la generación de codificación posicional (útil para modelos basados en atención)
y se definen los métodos de interfaz para la configuración del plugin.

Nota: La descomposición STL se contempla en un plugin separado.
"""

import numpy as np
import pandas as pd
from app.data_handler import load_csv, write_csv  # Asegúrate de tener implementada esta función en tu proyecto.
import json

class PreprocessorPlugin:
    # Parámetros por defecto específicos del preprocesador.
    plugin_params = {
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "headers": True,
        "max_steps_train": None,
        "max_steps_val": None,
        "max_steps_test": None,
        "window_size": 24,
        "time_horizon": 1,
        "use_returns": False,
        "pos_encoding_dim": 16  # Dimensión para la codificación posicional.
    }
    # Variables de debug (pueden ser extendidas según se requiera)
    plugin_debug_vars = ["window_size", "time_horizon", "use_returns"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del plugin combinando los parámetros específicos
        con la configuración global.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Devuelve información de debug de los parámetros relevantes.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Añade información de debug al diccionario debug_info.
        """
        debug_info.update(self.get_debug_info())

    @staticmethod
    def create_sliding_windows_single(data, window_size, time_horizon, date_times=None):
        """
        Crea ventanas deslizantes para una serie univariada con objetivo de un solo paso.

        Args:
            data (np.ndarray): Serie 1D de valores.
            window_size (int): Número de pasos pasados para la ventana.
            time_horizon (int): Número de pasos adelante para el target.
            date_times (pd.DatetimeIndex, optional): Índices de fecha para la serie.

        Returns:
            tuple: (windows, targets, date_windows)
              - windows: np.ndarray de forma (n_samples, window_size)
              - targets: np.ndarray de forma (n_samples,) con el valor futuro.
              - date_windows: lista de fechas correspondientes a cada ventana (si se provee).
        """
        windows = []
        targets = []
        date_windows = []
        n = len(data)
        for i in range(0, n - window_size - time_horizon + 1):
            window = data[i : i + window_size]
            target = data[i + window_size + time_horizon - 1]
            windows.append(window)
            targets.append(target)
            if date_times is not None:
                date_windows.append(date_times[i + window_size - 1])
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows

    @staticmethod
    def generate_positional_encoding(num_features, pos_dim=16):
        """
        Genera una codificación posicional para un número dado de features.

        Args:
            num_features (int): Número de features en la entrada.
            pos_dim (int): Dimensión de la codificación posicional.

        Returns:
            np.ndarray: Codificación posicional de forma (1, num_features * pos_dim).
        """
        position = np.arange(num_features)[:, np.newaxis]
        div_term = np.exp(np.arange(0, pos_dim, 2) * -(np.log(10000.0) / pos_dim))
        pos_encoding = np.zeros((num_features, pos_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        pos_encoding_flat = pos_encoding.flatten().reshape(1, -1)
        return pos_encoding_flat

    def process_data(self, config):
        """
        Procesa los datos para el preprocesamiento de la predicción de EUR/USD.

        Pasos:
          1. Carga de archivos CSV definidos en la configuración.
          2. Conversión de la columna 'typical_price' a array NumPy (float32).
          3. Creación de ventanas deslizantes para un forecast de un solo paso.
             - Si config["use_returns"] es True, se calcula el target como la diferencia entre el valor futuro y el último valor de la ventana.
          4. Remodelado de las ventanas a (samples, window_size, 1) para inputs univariados.
          5. Retorna un diccionario con los datasets procesados y metadatos.

        Args:
            config (dict): Diccionario de configuración que debe incluir:
                - "x_train_file", "x_validation_file", "x_test_file"
                - "headers": bool, indica si los CSV tienen encabezado.
                - "max_steps_train", "max_steps_val", "max_steps_test": máximas filas a cargar.
                - "window_size": longitud de la ventana deslizante.
                - "time_horizon": horizonte de predicción.
                - "use_returns": bool, indica si el target es el retorno (diferencia).

        Returns:
            dict: Diccionario con:
                "x_train", "y_train", "x_val", "y_val", "x_test", "y_test",
                "dates_train", "dates_val", "dates_test",
                "y_train_array", "y_val_array", "y_test_array",
                "test_close_prices"
        """
        # 1. Cargar archivos CSV
        x_train_df = load_csv(config["x_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
        x_val_df   = load_csv(config["x_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
        x_test_df  = load_csv(config["x_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))

        # 2. Intentar convertir el índice a datetime
        for df in (x_train_df, x_val_df, x_test_df):
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    df.index = None

        # 3. Extraer la columna 'typical_price'
        if "typical_price" not in x_train_df.columns:
            raise ValueError("Column 'typical_price' not found in training data.")
        close_train = x_train_df["typical_price"].astype(np.float32).values
        close_val   = x_val_df["typical_price"].astype(np.float32).values
        close_test  = x_test_df["typical_price"].astype(np.float32).values

        train_dates = x_train_df.index if x_train_df.index is not None else None
        val_dates   = x_val_df.index   if x_val_df.index is not None else None
        test_dates  = x_test_df.index  if x_test_df.index is not None else None

        # 4. Crear ventanas deslizantes para forecast de un solo paso.
        window_size = config["window_size"]
        time_horizon = config["time_horizon"]
        use_returns = config.get("use_returns", False)

        X_train, y_train, dates_train = self.create_sliding_windows_single(close_train, window_size, time_horizon, train_dates)
        X_val, y_val, dates_val       = self.create_sliding_windows_single(close_val, window_size, time_horizon, val_dates)
        X_test, y_test, dates_test      = self.create_sliding_windows_single(close_test, window_size, time_horizon, test_dates)

        # 5. Si se usa returns, calcular la diferencia respecto al último valor de la ventana.
        if use_returns:
            baseline_train = X_train[:, -1]
            baseline_val   = X_val[:, -1]
            baseline_test  = X_test[:, -1]
            y_train = y_train - baseline_train
            y_val   = y_val - baseline_val
            y_test  = y_test - baseline_test

        # 6. Remodelar las ventanas a (samples, window_size, 1)
        X_train = X_train.reshape(-1, window_size, 1)
        X_val   = X_val.reshape(-1, window_size, 1)
        X_test  = X_test.reshape(-1, window_size, 1)

        # 7. Crear las listas de targets y arrays de target
        y_train_list = [y_train]
        y_val_list   = [y_val]
        y_test_list  = [y_test]
        y_train_array = y_train.reshape(-1, 1)
        y_val_array   = y_val.reshape(-1, 1)
        y_test_array  = y_test.reshape(-1, 1)

        # 8. Para precios de test, utilizar el último valor de cada ventana
        test_close_prices = close_test[window_size - 1 : len(close_test) - time_horizon]

        # Debug messages
        print("Processed datasets:")
        print(" X_train shape:", X_train.shape, " y_train shape:", y_train_array.shape)
        print(" X_val shape:  ", X_val.shape,   " y_val shape:  ", y_val_array.shape)
        print(" X_test shape: ", X_test.shape,  " y_test shape: ", y_test_array.shape)
        print(" Test close prices shape:", test_close_prices.shape)

        ret = {
            "x_train": X_train,
            "y_train": y_train_list,
            "x_val": X_val,
            "y_val": y_val_list,
            "x_test": X_test,
            "y_test": y_test_list,
            "dates_train": dates_train,
            "dates_val": dates_val,
            "dates_test": dates_test,
            "y_train_array": y_train_array,
            "y_val_array": y_val_array,
            "y_test_array": y_test_array,
            "test_close_prices": test_close_prices
        }
        if use_returns:
            ret["baseline_train"] = X_train[:, -1]
            ret["baseline_val"]   = X_val[:, -1]
            ret["baseline_test"]  = X_test[:, -1]
        return ret

    def run_preprocessing(self, config):
        """
        Método de conveniencia para ejecutar el procesamiento de datos.
        Retorna el diccionario generado por process_data.
        """
        return self.process_data(config)


# Ejemplo de uso cuando se ejecuta el plugin directamente (modo debug)
if __name__ == "__main__":
    # Crear una instancia del plugin y probar process_data con una configuración mínima
    plugin = PreprocessorPlugin()
    # Se puede definir una configuración de prueba (se pueden sobreescribir estos valores vía CLI en el sistema real)
    test_config = {
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "headers": True,
        "max_steps_train": 1000,
        "max_steps_val": 500,
        "max_steps_test": 500,
        "window_size": 24,
        "time_horizon": 1,
        "use_returns": False
    }
    datasets = plugin.process_data(test_config)
    debug_info = plugin.get_debug_info()
    print("Debug Info:", debug_info)
    print("Datasets keys:", list(datasets.keys()))
