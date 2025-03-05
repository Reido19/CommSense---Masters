

fileID = fopen('/home/torps/CommSense/Datadump/Random_Testing/4/dat/reid_receive_1.dat', 'rb');  % Open the file in binary mode

% Read the entire data as double (64 bits = 8 bytes per value)
data = fread(fileID, 'double');  % 'double' reads 64-bit floating-point values
fclose(fileID);

% Reshape the data into two rows (I and Q interleaved)
IQ = reshape(data, 2, []);  % Each column now holds [I; Q] for a sample

% Extract I and Q components
I = IQ(1, :);  % In-phase components
Q = IQ(2, :);  % Quadrature components

% Optional: Combine into complex samples (I + jQ)
complexData = I + 1i * Q;

% Display or process as needed
% disp(complexData);
complex_reshape = reshape(complexData,[],1);

complex_reshape = complex_reshape(10e6:20e6);