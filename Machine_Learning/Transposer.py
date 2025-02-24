import pandas as pd

# Function to read specific number of lines and transpose them
def read_and_transpose(df, num_lines):
    transposed_data = []
    for i in range(min(num_lines, len(df))):  # Ensure not to exceed the number of available rows
        transposed_data.append(df.iloc[i].values)
    return pd.DataFrame(transposed_data).transpose()


def Transpose(dataset):
    
    original_df = dataset

    num_lines_to_read = 600  # Specify the number of lines to read at once


    # Create an empty DataFrame to store the transposed data
    transposed_df = pd.DataFrame()

    # Loop through the original dataset, reading and transposing data
    num_rows = len(original_df)
    start_idx = 0
    while start_idx < num_rows:
        end_idx = min(start_idx + num_lines_to_read, num_rows)
        chunk_df = original_df.iloc[start_idx:end_idx]
        transposed_chunk = read_and_transpose(chunk_df, num_lines_to_read)
        transposed_df = pd.concat([transposed_df, transposed_chunk], ignore_index=True)
        start_idx += num_lines_to_read

    return transposed_df