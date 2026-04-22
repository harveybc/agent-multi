import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.data_processor import process_data, train_autoencoder
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_plugin
from app.reconstruction import unwindow_data

sample_data = pd.DataFrame({
    'column1': np.random.rand(100),
    'column2': np.random.rand(100)
})

@patch('app.plugin_loader.load_plugin')
@patch('app.data_handler.load_csv')
@patch('app.data_handler.write_csv')
@patch('app.reconstruction.unwindow_data')
def test_process_data(mock_unwindow_data, mock_write_csv, mock_load_csv, mock_load_plugin):
    config = {
        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
        'csv_output_path': './output',
        'decoder_plugin': 'mock_decoder',
        'encoder_plugin': 'mock_encoder',
        'window_size': 10,
        'headers': False,
        'force_date': False
    }

    mock_load_csv.return_value = sample_data
    mock_load_plugin.side_effect = [(MagicMock(), []), (MagicMock(), [])]
    mock_unwindow_data.return_value = pd.DataFrame({'Output': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    processed_data, debug_info = process_data(config)

    assert isinstance(processed_data, dict)
    for value in processed_data.values():
        assert isinstance(value, np.ndarray)

def test_train_autoencoder():
    autoencoder_manager = MagicMock()
    autoencoder_manager.calculate_mse.return_value = 0.05  # Mock the return value of calculate_mse

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    trained_autoencoder_manager = train_autoencoder(
        autoencoder_manager=autoencoder_manager, data=data, mse_threshold=0.1, initial_size=4, step_size=2, incremental_search=False, epochs=10
    )

    assert trained_autoencoder_manager is not None
    autoencoder_manager.calculate_mse.assert_called()  # Ensure calculate_mse was called
