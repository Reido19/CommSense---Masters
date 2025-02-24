
% Loop to collect multiple sets of LTE DATA
num_data_captures = 1;
filename = '/home/torps/CommSense/Datadump/Random_Testing/2/Bin/LTE_3_Omni_CRG_0_0.bin';
filenameCSV = '/home/torps/CommSense/Datadump/Random_Testing/2/CSV/LTE_3_Omni_CRG_0_0.csv';
file_raw = '/home/torps/CommSense/Datadump/Random_Testing/2/Raw/LTE_3_Omni_CRG_0_0.bin';
dataDump = [];
rms_evm = [];
peak_evm = [];
eNode_raw = [];
IQ_matrix = [];
matrix = [];
rows = 14;
columns = 600;

for n = 1:num_data_captures

    % %Function to Colled IQ Data
    % IQ = Band3Collector();
    IQ = block1(1:1.28e6);
    % %Matlab Receiver to Demodulated Data
    LTE_struct = struct();
    [LTE_struct, hest, rmsevm, peakevm] = SIB1RecoveryExample_edited(IQ, LTE_struct);
    % SIB1RecoveryExample_edited();
    dataDump = [dataDump;hest(:,:,1)];
    rms_evm=[rms_evm;rmsevm];
    peak_evm = [peak_evm;peakevm];
    
    IQ_matrix = reshape(hest(:,:,1), columns, rows).'; 
    matrix = [matrix;IQ_matrix]
    % pause(0.2);
end

% publish_struct(LTE_struct); 

data_rows = size(dataDump,1); 
extra1 = ones(data_rows,1);
extra0 = zeros(data_rows,1);

writematrix(matrix, filenameCSV);

disp("CSV File Written")

dataDump = dataDump(:)';
% Separate real and imaginary parts
realPart = real(dataDump);
imagPart = imag(dataDump);

% Open a file for writing
fileID = fopen(filename, 'w');

% Check if the file was successfully opened
if fileID == -1
    error('File could not be opened.');
end

for k = 1:length(dataDump)
    % Write real and imaginary parts to the file
    fwrite(fileID, realPart(k), 'double');
    fwrite(fileID, imagPart(k), 'double');
end
% Close the file
fclose(fileID);

% fprintf('Read real part     : %.15f\n', realPart);
% fprintf('Read imaginary part: %.15f\n', imagPart);
disp(size(realPart))
disp(size(imagPart))
disp('Complex data has been written to complex.bin');

realRaw = real(IQ);
imagRaw = imag(IQ);

% Open a file for writing
fileID_raw = fopen(file_raw, 'w');

% Check if the file was successfully opened
if fileID_raw == -1
    error('File raw could not be opened.');
end

for l = 1:length(IQ)
    % Write real and imaginary parts to the file
    fwrite(fileID_raw, realRaw(l), 'double');
    fwrite(fileID_raw, imagRaw(l), 'double');
end
% Close the file
fclose(fileID_raw);


% % % % % % Open a CSV file for writing
% % % % % fileID_csv = fopen(filenameCSV, 'w');
% % % % % 
% % % % % % Check if the file was successfully opened
% % % % % if fileID_csv == -1
% % % % %     error('File csv could not be opened.');
% % % % % end
% % % % % 
% % % % % % Combine real and imaginary parts into a single matrix
% % % % % IQ_data = [real(IQ(:)), imag(IQ(:))];
% % % % % 
% % % % % % Write the IQ data to a CSV file
% % % % % writematrix(IQ_data, filenameCSV);
% % % % % 
% % % % % % Close the file
% % % % % fclose(fileID_csv);


% Calculate SNR based on RMS and Peak EVM values
snr_rms_dB = -20 * log10(rms_evm / 100);
snr_peak_dB = -20 * log10(peak_evm / 100);

% Display the results
disp(rms_evm)
fprintf('SNR from PDSCH RMS EVM: %.2f dB\n', snr_rms_dB);
disp(peak_evm)
fprintf('SNR from PDSCH Peak EVM: %.2f dB\n', snr_peak_dB);

