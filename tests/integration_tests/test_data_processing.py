import pytest
import pandas as pd
import numpy as np
from app.plugin_loader import load_plugin
from app.autoencoder_manager import AutoencoderManager
from app.data_processor import train_autoencoder

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.rand(1000, 10))

@pytest.fixture
def config():
    return {
        'csv_file': 'path/to/mock_csv.csv',
        'window_size': 128,
        'initial_encoding_dim': 4,
        'encoding_step_size': 4,
        'mse_threshold': 0.005,
        'epochs': 10,
        'headers': False,
        'force_date': False,
        'incremental_search': True
    }

def test_train_autoencoder(mock_data, config):
    encoder_plugin, encoder_params = load_plugin('feature_extractor.encoders', 'default')
    decoder_plugin, decoder_params = load_plugin('feature_extractor.decoders', 'default')

    print(f"[test_train_autoencoder] Encoder params: {encoder_params}")
    print(f"[test_train_autoencoder] Decoder params: {decoder_params}")

    autoencoder_manager = AutoencoderManager(input_dim=mock_data.shape[1], encoding_dim=config['initial_encoding_dim'])
    print(f"[test_train_autoencoder] AutoencoderManager initialized")

    assert autoencoder_manager.input_dim == mock_data.shape[1]
    assert autoencoder_manager.encoding_dim == config['initial_encoding_dim']

    trained_manager = train_autoencoder(autoencoder_manager, mock_data.values, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])
    assert trained_manager is not None
    print("Checking if build_autoencoder was called...")
    assert autoencoder_manager.autoencoder_model is not None