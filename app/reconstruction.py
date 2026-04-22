import numpy as np
import pandas as pd

def unwindow_data(windowed_df):
    """
    Transform a windowed dataset into a non-windowed dataset by following a precise procedure.
    
    Parameters:
    windowed_df (DataFrame): The input dataset with windowed data.
    
    Returns:
    DataFrame: The resulting non-windowed dataset.
    """
    window_size = windowed_df.shape[1]
    num_rows = len(windowed_df)
    total_rows_out = num_rows + window_size

    output_dataset = pd.DataFrame(0, index=range(total_rows_out-1), columns=['Output'])
    print("Un-Windowing output data")
    percen_val = num_rows // 100
    count=0
    for row in range(num_rows):
        if count == percen_val:
            print(f"{row//percen_val}% done", end="\r", flush=True)
            count = 0
        count += 1
        extended_row = np.zeros(total_rows_out-1)
        extended_row[row:row + window_size] = windowed_df.iloc[row].values
        output_dataset['Output'] += extended_row
    
    print("calculating averages in the first segment")
    for row in range(window_size - 2):
        output_dataset.iloc[row] /= (row + 1)
    print("calculating averages in the second segment")
    for row in range(window_size - 2, total_rows_out - window_size):
        if count == percen_val:
            print(f"{row//percen_val}% done", end="\r", flush=True)
            count = 0
        count += 1
        output_dataset.iloc[row] /= window_size
    print("calculating averages in the last segment")        
    for row in range(total_rows_out - window_size, total_rows_out-1):
        output_dataset.iloc[row] /= (total_rows_out - row)

    return output_dataset
