import pytest
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins
from app.config import DEFAULT_VALUES

def test_load_plugin_success():
    plugin_class, required_params = load_plugin('feature_extractor.encoders', 'default')
    assert plugin_class.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert required_params == ['epochs', 'batch_size']

def test_load_encoder_decoder_plugins():
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('default', 'default')
    assert encoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert decoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}
