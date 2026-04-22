import pytest
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins, get_plugin_params

def test_load_plugin_success():
    plugin_class, required_params = load_plugin('feature_extractor.encoders', 'default')
    assert plugin_class.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert required_params == ['epochs', 'batch_size']

def test_load_plugin_key_error():
    with pytest.raises(ImportError):
        load_plugin('feature_extractor.encoders', 'non_existent_plugin')

def test_load_plugin_general_exception():
    with pytest.raises(ImportError) as excinfo:
        load_plugin('feature_extractor.encoders', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group feature_extractor.encoders.' in str(excinfo.value)

def test_load_encoder_decoder_plugins():
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('default', 'default')
    assert encoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert decoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}

def test_get_plugin_params_success():
    params = get_plugin_params('feature_extractor.encoders', 'default')
    assert params == {'epochs': 10, 'batch_size': 256}

def test_get_plugin_params_key_error():
    with pytest.raises(ImportError):
        get_plugin_params('feature_extractor.encoders', 'non_existent_plugin')

def test_get_plugin_params_general_exception():
    with pytest.raises(ImportError) as excinfo:
        get_plugin_params('feature_extractor.encoders', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group feature_extractor.encoders.' in str(excinfo.value)
