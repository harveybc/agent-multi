import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Ioin: A tool for timeseries prediction with plugin support.")
    parser.add_argument('--x_train_file', type=str, help='Path to the input CSV file that is used for training the model (x_train).')
    parser.add_argument('-ytf', '--y_train_file', type=str, help='Path to the input CSV file that is used for training the model (y_train), IMPORTANT: it is not shifted, must coincide 1 to 1|with the training data.')
    parser.add_argument('-xvf', '--x_validation_file', type=str, help='Path to the input CSV file that is used for validation (x_validation).')
    parser.add_argument('-yvf', '--y_validation_file', type=str, help='Path to the input CSV file that is used for validation (y_validation), IMPORTANT: it is not shifted, must coincide 1 to 1|with the validation data.')
    parser.add_argument('-tc', '--target_column', type=str, help='If used, assumes no input_timeseries is used but the input_timeseries is a target column in the input CSV file, in all cases, each row in the input_csv must correspond with the exact tick time of the timeseries.')
    parser.add_argument('-of', '--output_file', type=str, help='Path to the output CSV file with the prediction vlues.')
    parser.add_argument('-rf', '--results_file', type=str, help='Path to the output CSV file with the result training and  validationstatistics.')
    parser.add_argument('-sm', '--save_model', type=str, help='Filename to save the trained ioin model.')
    parser.add_argument('-lm', '--load_model', type=str, help='Filename to load a trained ioin model from (does not perform training, just evaluate input data).')
    parser.add_argument('-ef', '--evaluate_file', type=str, help='Filename for outputting loaded model evaluation results.')
    parser.add_argument('-pl', '--plugin', type=str,  help='Name of the encoder plugin to use.')
    parser.add_argument('-th', '--time_horizon', type=int, help='Number of ticks ahead to predict.')
    parser.add_argument('-te', '--threshold_error', type=float, help='MSE error threshold to stop the training process.')
    parser.add_argument('-rl', '--remote_log', type=str, help='URL of a remote API endpoint for saving debug variables in JSON format.')
    parser.add_argument('-rlc', '--remote_load_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('-rsc', '--remote_save_config', type=str, help='URL of a remote API endpoint for saving configuration in JSON format.')
    parser.add_argument('-u', '--username', type=str, help='Username for the API endpoint.')
    parser.add_argument('-p', '--password', type=str, help='Password for Username for the API endpoint.')
    parser.add_argument('-lc', '--load_config', type=str, help='Path to load a configuration file.')
    parser.add_argument('-sc', '--save_config', type=str, help='Path to save the current configuration.')
    parser.add_argument('-sl', '--save_log', type=str, help='Path to save the current debug info.')
    parser.add_argument('-qm', '--quiet_mode', action='store_true', help='Suppress output messages.')
    parser.add_argument('-fd', '--force_date', action='store_true', help='Include date in the output CSV files.')
    parser.add_argument('-hdr', '--headers', action='store_true', help='Indicate if the CSV file has headers.')
    parser.add_argument('-io', '--input_offset', type=int,help='Offset for input data to account for feature extraction window size.')
    parser.add_argument('-mstr', '--max_steps_train', type=int,help='Offset for input data to account for feature extraction window size.')
    parser.add_argument('-mste', '--max_steps_test', type=int,help='Offset for input data to account for feature extraction window size.')
    parser.add_argument('-it', '--iterations', type=int,help='number of times the whole process is made and after that the training and validation MAE are averaged and also the std dev, max and min is shown.')
    parser.add_argument('-e', '--epochs', type=int,help='number of epochs for the plugin model training.')
    parser.add_argument('-ud', '--use_daily', action='store_true',help='isntead of predicting the next time_horizon hours, predict the next time_horizon days.')
    # the predicted_hrizons parameter is a list of integers, each integer is a number of hours to predict
    parser.add_argument('-ph', '--predicted_hrizons', type=int, nargs='*', help='list of predicted hrizons to predict.')
    # "feature_extractor_file": "examples/results/phase_3_2_daily/phase_3_2_cnn_25200_1d_encoder_model.h5.keras",
    # "train_fe" : false
    parser.add_argument('--feature_extractor_file', type=str, help='Path to the feature extractor file.')
    parser.add_argument('--train_fe', action='store_true', help='Train the feature extractor.')



    return parser.parse_known_args()
