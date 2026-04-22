import pytest
import json
import requests
from unittest.mock import patch, mock_open
from app.config_handler import load_config, save_config, merge_config, save_debug_info, load_remote_config, save_remote_config, log_remote_data
from app.config import DEFAULT_VALUES

# Mock data for tests
mock_config = {
    'csv_input_path': './test_input.csv',
    'csv_output_path': './test_output.csv',
    'encoder_plugin': 'test_encoder',
    'decoder_plugin': 'test_decoder',
    'training_batch_size': 64,
    'epochs': 20
}

mock_debug_info = {
    'mean_squared_error_0': 0.01,
    'mean_absolute_error_0': 0.005
}

mock_remote_config = {
    'status': 'success',
    'config': mock_config
}

# Test loading configuration from a file
def test_load_config():
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        config = load_config('config.json')
        assert config == mock_config

# Test saving configuration to a file
def test_save_config():
    with patch("builtins.open", mock_open()) as mocked_file:
        config, path = save_config(mock_config, 'config_out.json')
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == mock_config
        assert config == mock_config
        assert path == 'config_out.json'

# Test merging configuration
def test_merge_config():
    cli_args = {'csv_file': './cli_input.csv', 'epochs': 30}
    plugin_params = {'max_error': 0.1}
    merged_config = merge_config(mock_config, cli_args, plugin_params)
    expected_config = {**DEFAULT_VALUES, **mock_config, **cli_args, **plugin_params}
    assert merged_config == expected_config

# Test saving debug information to a file
def test_save_debug_info():
    with patch("builtins.open", mock_open()) as mocked_file:
        save_debug_info(mock_debug_info, 'debug_out.json')
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == mock_debug_info

# Test loading remote configuration
def test_load_remote_config():
    with patch('requests.get') as mocked_get:
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.json.return_value = mock_remote_config
        config = load_remote_config('http://example.com/config', 'user', 'pass')
        assert config == mock_remote_config

# Test saving remote configuration
def test_save_remote_config():
    with patch('requests.post') as mocked_post:
        mocked_post.return_value.status_code = 200
        result = save_remote_config(mock_config, 'http://example.com/config', 'user', 'pass')
        assert result == True

# Test logging remote data
def test_log_remote_data():
    with patch('requests.post') as mocked_post:
        mocked_post.return_value.status_code = 200
        result = log_remote_data(mock_debug_info, 'http://example.com/log', 'user', 'pass')
        assert result == True

if __name__ == "__main__":
    pytest.main()
