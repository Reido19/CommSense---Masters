
import struct 
import numpy as np
import pandas as pd
import sys
import os
import Transposer

def read_complex_values_from_binary_file(filename):
    complex_numbers = []
    real_numbers = []
    real_row = []
    imag_row = []
    imag_numbers = []
    

    count = 0
    with open(filename, 'rb') as file:
        # Determine the size of each complex number in bytes
        size_of_double = 8  # size of float64 in bytes
        size_of_complex = 2 * size_of_double  # since complex number has a real and imaginary part

        # Read the entire file content
        file_content = file.read()

        # Calculate the number of complex numbers in the file
        num_complex_numbers = len(file_content) // size_of_complex
        column_len = int(num_complex_numbers/14)
        # Unpack the binary data
        for i in range(num_complex_numbers):
            real, imag = struct.unpack('dd', file_content[i * size_of_complex:(i+1) * size_of_complex])
            real_row.append(real)
            imag_row.append((imag)*-1)
            complex_numbers.append(complex(real, (imag)*-1))

            if (count+1) % column_len == 0:
                real_numbers.append(real_row.copy())
                imag_numbers.append(imag_row.copy())
                real_row.clear()
                imag_row.clear()
            count += 1
    return complex_numbers, real_numbers, imag_numbers


# Usage example

def get_pandas_dataFrame(folder_path):

    complex_place = []
    real_place = []
    imag_place = []
    complex_values = []
    real_values = []
    imag_values = []
    target_box = []
    ori_target = []
    target_zero = "_0"
    target_one = "_1"
    runs = 0
        
    files = os.listdir(folder_path)
    files = sorted(files)
    for i in files:
        filename = folder_path + i
        complex_place, real_place, imag_place = read_complex_values_from_binary_file(filename)
        complex_values = np.hstack((complex_values, complex_place))
        x = np.array(real_place)
        y = int(x.shape[1]/600)

        if(runs == 0):
            real_values = real_place
            imag_values = imag_place
        else:
            real_values = np.hstack((real_values, real_place))
            imag_values = np.hstack((imag_values, imag_place))
        runs+=1
        
        target_values = [int(i[-5])] * (len(real_place)*y)
        target_box.extend(target_values)

        ori_target_values = [int(i[-5])] * x.shape[1]
        ori_target.extend(ori_target_values)

    
    test = np.vstack((real_values, imag_values))
    test_T = test.T

    orginal_df = pd.DataFrame(test.T)
    df = pd.DataFrame(test_T)
    
    final_data_frame = Transposer.Transpose(df)

    real_parts = []
    imag_parts = []

    for i in range (0, len(final_data_frame), 28):
        real_part1 =final_data_frame.iloc[i:i+14, :].reset_index(drop=True)
        imag_part =final_data_frame.iloc[i+14:i+28, :].reset_index(drop=True)

        real_parts.append(real_part1)
        imag_parts.append(imag_part)

    combined_real = pd.concat(real_parts, ignore_index=True)
    combined_imag = pd.concat(imag_parts, ignore_index=True)

    # final_df = pd.concat([combined_real, combined_imag], axis=1)
    complex_data=combined_real+1j*combined_imag
    final_df = pd.DataFrame(complex_data)

    headings = []
    for i in range(600):
        headings.append(i)

    targets = np.array(target_box)
    targets = targets.transpose()
    final_df.columns = headings

    final_df["Target"] = targets
    orginal_df["Target"] = ori_target

    return final_df, orginal_df

