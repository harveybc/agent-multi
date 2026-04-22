import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from io import StringIO
from app.data_handler import load_csv, write_csv, sliding_window

# Mock data for tests
mock_data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1]
})

# Test loading CSV file with headers
def test_load_csv_with_headers():
    with patch("builtins.open", mock_open(read_data=mock_data.to_csv(index=False))):
        data = load_csv('test.csv', headers=True)
        pd.testing.assert_frame_equal(data, mock_data)

# Test loading CSV file without headers
def test_load_csv_without_headers():
    with patch("builtins.open", mock_open(read_data=mock_data.to_csv(index=False, header=False))):
        data = load_csv('test.csv', headers=False)
        pd.testing.assert_frame_equal(data, pd.read_csv('test.csv', header=None))

# Test writing CSV file
def test_write_csv():
    with patch("builtins.open", mock_open()) as mocked_file:
        write_csv('test_write.csv', mock_data.values, include_date=False, headers=mock_data.columns)
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        written_df = pd.read_csv(StringIO(written_content))
        pd.testing.assert_frame_equal(written_df, mock_data)

# Test sliding window with sufficient data
def test_sliding_window_sufficient_data():
    data = np.array([1, 2, 3, 4, 5])
    window_size = 3
    expected_output = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])
    result = sliding_window(data, window_size)
    result = result.reshape(result.shape[0], result.shape[1])
    np.testing.assert_array_equal(result, expected_output)

# Test sliding window with insufficient data
def test_sliding_window_insufficient_data():
    data = np.array([1, 2])
    window_size = 3
    expected_output = np.empty((0, window_size))
    result = sliding_window(data, window_size)
    np.testing.assert_array_equal(result, expected_output)

# Test sliding window with exact data size
def test_sliding_window_exact_data_size():
    data = np.array([1, 2, 3])
    window_size = 3
    expected_output = np.array([[1, 2, 3]])
    result = sliding_window(data, window_size)
    result = result.reshape(result.shape[0], result.shape[1])
    np.testing.assert_array_equal(result, expected_output)

# Test sliding window with 2D data
def test_sliding_window_2d_data():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    window_size = 2
    expected_output = np.array([
        [[1, 2], [3, 4]],
        [[3, 4], [5, 6]],
        [[5, 6], [7, 8]]
    ])
    result = sliding_window(data, window_size)
    np.testing.assert_array_equal(result, expected_output)

if __name__ == "__main__":
    pytest.main()
