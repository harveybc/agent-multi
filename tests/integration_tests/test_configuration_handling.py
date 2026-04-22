import pytest
import json
from unittest.mock import patch, mock_open
from app import config_handler
from app.config import DEFAULT_VALUES

@pytest.fixture
def default_config():
    return DEFAULT_VALUES.copy()

def test_load_default_config(default_config):
    with patch('builtins.open', mock_open(read_data=json.dumps(default_config))):
        loaded_config = config_handler.load_config(DEFAULT_VALUES['config_load_path'])
        assert loaded_config == default_config

def test_save_config(default_config):
    m = mock_open()
    with patch('builtins.open', m):
        config_handler.save_config(default_config, DEFAULT_VALUES['config_save_path'])
    m.assert_called_once_with(DEFAULT_VALUES['config_save_path'], 'w')
    handle = m()
    handle.write.assert_called_once_with(json.dumps({k: v for k, v in default_config.items() if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]}, indent=4))

def test_configure_with_args(default_config):
    args = {
        'csv_input_path': './new_input.csv',
        'csv_output_path': './new_output.csv',
        'epochs': 20,
        'quiet_mode': True
    }
    updated_config = config_handler.configure_with_args(default_config, args)
    assert updated_config['csv_input_path'] == './new_input.csv'
    assert updated_config['csv_output_path'] == './new_output.csv'
    assert updated_config['epochs'] == 20
    assert updated_config['quiet_mode'] is True
