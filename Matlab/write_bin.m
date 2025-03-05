function write_bin(dataDump,filename)
    

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

end