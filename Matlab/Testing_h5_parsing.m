clear all; clf; close all;

filename = "LTETest1.h5";
temp = h5read(filename, '/SSNC_data');
block1 = double(temp.real) + 1i * double(temp.imag);
for (idx = 2:67)
    % Read a block of data
    block_name = sprintf("/SSNC_data%d", idx);
    temp = h5read(filename, block_name);
    block = double(temp.real) + 1i * double(temp.imag);
    block1 = [block1; block];
end % for