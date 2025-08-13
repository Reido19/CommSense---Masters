
% Loop to collect multiple sets of LTE DATA

Datadump_filename = "Testing/Wheelchair_Testing/6";

Signal = "LTE";
Band = "3";
Location = "SnapeParking";
Antenna = "Omni";
Iteration = "11";  % CHANGE THIS VALUE WITH EACH CAPTURE
Object = "0";  % 0 with no object and 1 with object.

filename = '/home/torps/CommSense/Datadump/'+Datadump_filename+'/Bin/'+Signal+'_'+Band+'_'+Location+'_'+Antenna+'_'+Iteration+'_'+Object+'.bin';
filename_eq = '/home/torps/CommSense/Datadump/'+Datadump_filename+'/Bin_eq/'+Signal+'_'+Band+'_Equalized'+Location+'_'+Antenna+'_'+Iteration+'_'+Object+'.bin';
filenameCSV = '/home/torps/CommSense/Datadump/'+Datadump_filename+'/CSV/'+Signal+'_'+Band+'_'+Location+'_'+Antenna+'_'+Iteration+'_'+Object+'.csv';
file_raw = '/home/torps/CommSense/Datadump/'+Datadump_filename+'/Raw/'+Signal+'_'+Band+'_'+Location+'_'+Antenna+'_'+Iteration+'_'+Object+'.bin';

num_data_captures = 1;

dataDump = [];
dataDump_eq = [];
rms_evm = [];
peak_evm = [];
eNode_raw = [];
IQ_matrix = [];
matrix = [];
rows = 14;
columns = 600;


for n = 1:num_data_captures

    %Function to Colled IQ Data
    % IQ = 0;                   % Uncomment to run default matlab example
    % sr = 15.36e6;
    % IQ = complex_reshape;  
    % sr = 12.5e6;
    IQ = Band3Collector();    % Uncomment to collect fresh data using the USRP
    sr = 100e6/4;
    % IQ = block1(1:1.28e6);    % Uncomment to retreives IQ Data from
    % sr = 12.8e6;              % pre-collected data (block 1)

     
    % %Matlab Receiver to Demodulated Data
    LTE_struct = struct();
    [LTE_struct,eqGrid, hest1, rmsevm, peakevm] = SIB1RecoveryExample_edited(IQ, LTE_struct,sr);
    % dataDump = [dataDump;hest(:,:,1)];
    dataDump = [dataDump;hest1];
    % dataDump_eq = [dataDump;eqGrid(:,:,1)];
    rms_evm=[rms_evm;rmsevm];
    peak_evm = [peak_evm;peakevm];
    
    % IQ_matrix = reshape(hest1(:,:,1), columns, rows).'; 
    % matrix = [matrix;IQ_matrix];
    % pause(0.2);
end

% publish_struct(LTE_struct); 

% Publish Data to Files
% ------------------------------------------------------
writematrix(matrix, filenameCSV);
disp("CSV File Written")

write_bin(dataDump_eq, filename_eq);

write_bin(dataDump, filename);

write_bin(IQ, file_raw);
% -------------------------------------------------------



% 
% realRaw = real(IQ);
% imagRaw = imag(IQ);
% 
% % Open a file for writing
% fileID_raw = fopen(file_raw, 'w');
% 
% % Check if the file was successfully opened
% if fileID_raw == -1
%     error('File raw could not be opened.');
% end
% 
% for l = 1:length(IQ)
%     % Write real and imaginary parts to the file
%     fwrite(fileID_raw, realRaw(l), 'double');
%     fwrite(fileID_raw, imagRaw(l), 'double');
% end
% % Close the file
% fclose(fileID_raw);


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


% % Calculate SNR based on RMS and Peak EVM values
% snr_rms_dB = -20 * log10(rms_evm / 100);
% snr_peak_dB = -20 * log10(peak_evm / 100);
% 
% % Display the results
% disp(rms_evm)
% fprintf('SNR from PDSCH RMS EVM: %.2f dB\n', snr_rms_dB);
% disp(peak_evm)
% fprintf('SNR from PDSCH Peak EVM: %.2f dB\n', snr_peak_dB);

