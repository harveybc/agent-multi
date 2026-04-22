import pytest
from unittest.mock import patch, MagicMock
from app.main import main
from app.config import DEFAULT_VALUES

@patch('app.main.parse_args')
@patch('app.main.load_config')
@patch('app.main.save_config')
@patch('app.main.process_data')
@patch('app.main.DEFAULT_VALUES', {
    'csv_input_path': './csv_input.csv',
    'csv_output_path': './csv_output.csv',
    'config_save_path': './config_out.json',
    'config_load_path': './config_in.json',
    'encoder_plugin': 'default',
    'decoder_plugin': 'default',
    'training_batch_size': 128,
    'epochs': 10,
    'plugin_directory': 'app/plugins/',
    'remote_log_url': None,
    'remote_config_url': None,
    'window_size': 128,
    'initial_encoding_dim': 4,
    'encoding_step_size': 4,
    'mse_threshold': 0.3,
    'quiet_mode': False,
    'remote_username': 'test',
    'remote_password': 'pass',
    'save_encoder_path': './encoder_ann.keras',
    'save_decoder_path': './decoder_ann.keras',
    'force_date': False,
    'headers': False,
    'incremental_search': True
})
def test_main_with_invalid_arguments(mock_process_data, mock_save_config, mock_load_config, mock_parse_args):
    mock_args = MagicMock()
    mock_parse_args.return_value = (mock_args, ['--invalid_argument'])
    mock_process_data.return_value = (MagicMock(), {})
    mock_load_config.return_value = {}

    mock_args.csv_input_path = 'tests/data/csv_sel_unb_norm_512.csv'
    mock_args.save_encoder = './encoder_ann.keras'
    mock_args.save_decoder = './decoder_ann.keras'

    with patch('sys.argv', ['script_name', 'tests/data/csv_sel_unb_norm_512.csv', '--save_encoder', './encoder_ann.keras', '--save_decoder', './encoder_ann.keras', '--invalid_argument']):
        with pytest.raises(SystemExit):  # Expect SystemExit due to sys.exit(1)
            main()
