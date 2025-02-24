
function publish_struct(LTE_struct)

   fprintf('Struct Name: %s \n', LTE_struct.name) 
   receivedSignal = LTE_struct.raw;
   downsampled = LTE_struct.downsampled;
   rxgrid = LTE_struct.rxgrid;
   rxgrid_2 = LTE_struct.rxgrid_2;
   sr = 100e6/4;
   dsr = 1.92e6;
   % Time-domain signal
    % plot(real(receivedSignal)); % Plot I-component
    % hold on;
    % plot(imag(receivedSignal)); % Plot Q-component
    % title('Received Signal (Time Domain)');
    % xlabel('Sample Index');
    % ylabel('Amplitude');
    % grid on;

    % Frequency domain
    % pspectrum(receivedSignal, sr); % Spectrogram or Power Spectrum
    
    % disp(dsr);
    persistent scope;
    scope = spectrumAnalyzer(SampleRate=sr);
    scope(receivedSignal);
    persistent scope_2;
    scope_2 = spectrumAnalyzer(SampleRate=dsr);
    scope_2(downsampled);


    disp(size(rxgrid));
    
    % Select a single antenna port for visualization
    antennaPort = 1;
    gridToDisplay = abs(rxgrid(:,:,antennaPort));  % Magnitude of resource elements
    
    % Display the resource grid
    figure;
    imagesc(gridToDisplay);
    colormap(jet);  % Use a color map for better visualization
    colorbar;       % Add a color bar to indicate magnitude
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
    title(['LTE Resource Grid for Antenna Port ', num2str(antennaPort)]);

    % Select a single antenna port for visualization
    antennaPort_2 = 1;
    gridToDisplay_2 = abs(rxgrid_2(:,:,antennaPort_2));  % Magnitude of resource elements

    % Display the resource grid
    figure;
    imagesc(gridToDisplay_2);
    colormap(jet);  % Use a color map for better visualization
    colorbar;       % Add a color bar to indicate magnitude
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
    title(['LTE Resource Grid 2 for Antenna Port ', num2str(antennaPort_2)]);

end