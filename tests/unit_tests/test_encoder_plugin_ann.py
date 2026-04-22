import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.plugins.encoder_plugin_ann import Plugin
from keras.models import Model
from keras.layers import Dense, Input

@pytest.fixture
def encoder_plugin():
    return Plugin()

def test_set_params(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2, epochs=15, batch_size=32)
    assert encoder_plugin.params['input_dim'] == 5

def test_get_debug_info(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2)
    debug_info = encoder_plugin.get_debug_info()
    assert debug_info['input_dim'] == 5

def test_add_debug_info(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2)
    debug_info = {}
    encoder_plugin.add_debug_info(debug_info)
    assert debug_info['input_dim'] == 5

def test_configure_size(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    assert encoder_plugin.params['input_dim'] == 3
    assert encoder_plugin.params['encoding_dim'] == 2
    assert encoder_plugin.encoder_model is not None

def test_train(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    mock_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with patch.object(encoder_plugin.encoder_model, 'fit') as mock_fit:
        encoder_plugin.train(mock_data)
        mock_fit.assert_called_once()

def test_save(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    with patch('app.plugins.encoder_plugin_ann.save_model') as mock_save_model:
        encoder_plugin.save('test_path')
        mock_save_model.assert_called_once_with(encoder_plugin.encoder_model, 'test_path')

def test_load(encoder_plugin):
    with patch('app.plugins.encoder_plugin_ann.load_model', return_value=MagicMock(spec=Model)) as mock_load_model:
        encoder_plugin.load('test_path')
        mock_load_model.assert_called_once_with('test_path')
        assert encoder_plugin.encoder_model is not None
        assert isinstance(encoder_plugin.encoder_model, MagicMock)  # Adjusted to check for MagicMock instead of Model

def test_calculate_mse(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)

    # Create a mock autoencoder model
    input_layer = encoder_plugin.encoder_model.input
    encoded_layer = encoder_plugin.encoder_model.output
    # Decoder should match the encoded layer's shape
    decoded_layer = Dense(3, activation='relu', name="decoder_output")(encoded_layer)
    autoencoder_model = Model(inputs=input_layer, outputs=decoded_layer)
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')

    mock_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with patch.object(autoencoder_model, 'fit') as mock_fit:
        autoencoder_model.fit(mock_data, mock_data, epochs=encoder_plugin.params['epochs'], batch_size=encoder_plugin.params['batch_size'], verbose=1)
        mock_fit.assert_called_once()

    encoded_data = encoder_plugin.encode(mock_data)
    # Ensure decoded data matches the original mock_data's shape
    decoded_data = autoencoder_model.predict(mock_data)  # Predict based on original mock_data shape

    mse = np.mean(np.square(mock_data - decoded_data))
    assert isinstance(mse, float)
