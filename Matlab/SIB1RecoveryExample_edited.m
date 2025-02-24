%% Cell Search, MIB and SIB1 Recovery 

function [LTE_struct,hest, rmsevm, peakevm] = SIB1RecoveryExample_edited(IQ, LTE_struct,sr)
    
    loadFromFile = 1; % Set to 0 to generate the eNodeB output locally
    LTE_struct.name = 'LTE Structure';
    LTE_struct.raw = IQ;
    
    if loadFromFile

    % Custom IQ Data
    %--------------------------------------
        % eNodeBOutput = IQ;
        % sr = 100e6/4;
        % % sr = 12.8e6;
    % --------------------------------------
    % Inbuilt LTE Example Data
        load eNodeBOutput.mat           % Load I/Q capture of eNodeB output
        eNodeBOutput = double(eNodeBOutput)/32768; % Scale samples
        % sr = 15.36e6; 
    %---------------------------------------
    else
        rmc = lteRMCDL('R.3'); %#ok<UNRCH>
        rmc.NCellID = 17;
        rmc.TotSubframes = 41;
        rmc.PDSCH.RNTI = 61;
        % SIB parameters
        rmc.SIB.Enable = 'On';
        rmc.SIB.DCIFormat = 'Format1A';
        rmc.SIB.AllocationType = 0;
        rmc.SIB.VRBStart = 8;
        rmc.SIB.VRBLength = 8;
        rmc.SIB.Gap = 0;
        % SIB data field filled with random bits, this is not a valid SIB
        % message
        rmc.SIB.Data = randi([0 1],176,1);
        [eNodeBOutput,~,info] = lteRMCDLTool(rmc,[1;0;0;1]);
        sr = info.SamplingRate;     % Sampling rate of generated samples
    end
    
    
    % Set up some housekeeping variables:
    % separator for command window logging
    separator = repmat('-',1,50);
    % plots
    if (~exist('channelFigure','var') || ~isvalid(channelFigure))
        channelFigure = figure('Visible','off');        
    end
    [spectrumScope,synchCorrPlot,pdcchConstDiagram] = ...
        hSIB1RecoveryExamplePlots(channelFigure,sr);
    % PDSCH EVM
    pdschEVM = comm.EVM();
    pdschEVM.MaximumEVMOutputPort = true;
    
    enb = struct;                   % eNodeB config structure
    enb.NDLRB = 6;                  % Number of resource blocks
    ofdmInfo = lteOFDMInfo(setfield(enb,'CyclicPrefix','Normal')); %#ok<SFLD>
    
    if (isempty(eNodeBOutput))
        fprintf('\nReceived signal must not be empty.\n');
        return;
    end
    
    % Display received signal spectrum
    fprintf('\nPlotting received signal spectrum...\n');  % Comment out to remove display graphs
    % spectrumScope(awgn(eNodeBOutput, 100.0));             % Comment out to remove display graphs
    spectrumScope(eNodeBOutput);

    if (sr~=ofdmInfo.SamplingRate)
        if (sr < ofdmInfo.SamplingRate)
            warning('The received signal sampling rate (%0.3fMs/s) is lower than the desired sampling rate for cell search / MIB decoding (%0.3fMs/s); cell search / MIB decoding may fail.',sr/1e6,ofdmInfo.SamplingRate/1e6);
        end
        fprintf('\nResampling from %0.3fMs/s to %0.3fMs/s for cell search / MIB decoding...\n',sr/1e6,ofdmInfo.SamplingRate/1e6);
    else
        fprintf('\nResampling not required; received signal is at desired sampling rate for cell search / MIB decoding (%0.3fMs/s).\n',sr/1e6);
    end
    % Downsample received signal
    nSamples = ceil(ofdmInfo.SamplingRate/round(sr)*size(eNodeBOutput,1));
    nRxAnts = size(eNodeBOutput, 2);
    downsampled = zeros(nSamples, nRxAnts);
    for i=1:nRxAnts
        downsampled(:,i) = resample(eNodeBOutput(:,i), ofdmInfo.SamplingRate, round(sr));
    end
    LTE_struct.downsampled = downsampled;

    %% Cell Search, Cyclic Prefix Length and Duplex Mode Detection
    
    fprintf('\nPerforming cell search...\n');
    
    if (~isfield(enb,'DuplexMode'))
        duplexModes = {'TDD' 'FDD'};
    else
        duplexModes = {enb.DuplexMode};
    end
    if (~isfield(enb,'CyclicPrefix'))
        cyclicPrefixes = {'Normal' 'Extended'};
    else
        cyclicPrefixes = {enb.CyclicPrefix};
    end
    
    searchalg.MaxCellCount = 1;
    searchalg.SSSDetection = 'PostFFT';
    peakMax = -Inf;
    for duplexMode = duplexModes
        for cyclicPrefix = cyclicPrefixes
            enb.DuplexMode = duplexMode{1};
            enb.CyclicPrefix = cyclicPrefix{1};
            [enb.NCellID, offset, peak] = lteCellSearch(enb, downsampled,searchalg); %,searchalg
            enb.NCellID = enb.NCellID(1);
            offset = offset(1);
            peak = peak(1);
            if (peak>peakMax)
                enbMax = enb;
                offsetMax = offset;
                peakMax = peak;
            end
        end
    end
    
    % Use the cell identity, cyclic prefix length, duplex mode and timing
    % offset which gave the maximum correlation during cell search
    enb = enbMax;
    offset = offsetMax;
    
    corr = cell(1,3);
    idGroup = floor(enbMax.NCellID/3);
    for i = 0:2
        enb.NCellID = idGroup*3 + mod(enbMax.NCellID + i,3);
        [~,corr{i+1}] = lteDLFrameOffset(enb, downsampled);
        corr{i+1} = sum(corr{i+1},2);
    end
    threshold = 1.3 * max([corr{2}; corr{3}]); % multiplier of 1.3 empirically obtained
    if (max(corr{1})<threshold)    
        warning('Synchronization signal correlation was weak; detected cell identity may be incorrect.');
    end
    % Return to originally detected cell identity
    enb.NCellID = enbMax.NCellID;
    
    % Plot PSS/SSS correlation and threshold
    synchCorrPlot.YLimits = [0 max([corr{1}; threshold])*1.1];  % Comment out to remove display graphs
    synchCorrPlot([corr{1} threshold*ones(size(corr{1}))]);     % Comment out to remove display graphs
    

    % Perform timing synchronization
    fprintf('Timing offset to frame start: %d samples\n',offset);
    downsampled = downsampled(1+offset:end,:); 
    enb.NSubframe = 0;
    % Show cell-wide settings
    fprintf('Cell-wide settings after cell search:\n');
    disp(enb);
    
    %% Frequency Offset Estimation and Correction
    
    fprintf('\nPerforming frequency offset estimation...\n');
    if (strcmpi(enb.DuplexMode,'TDD'))
        enb.TDDConfig = 0;
        enb.SSC = 0;
    end
    delta_f = lteFrequencyOffset(enb, downsampled);
    fprintf('Frequency offset: %0.3fHz\n',delta_f);
    downsampled = lteFrequencyCorrect(enb, downsampled, delta_f);    
    
    %% OFDM Demodulation and Channel Estimation  
    
    % Channel estimator configuration
    cec.PilotAverage = 'UserDefined';     % Type of pilot averaging
    cec.FreqWindow = 13;                  % Frequency window size    
    cec.TimeWindow = 9;                   % Time window size    
    cec.InterpType = 'cubic';             % 2D interpolation type
    cec.InterpWindow = 'Centered';        % Interpolation window type
    cec.InterpWinSize = 1;                % Interpolation window size  
    
    enb.CellRefP = 4;   
                        
    fprintf('Performing OFDM demodulation...\n\n');
    
    griddims = lteResourceGridSize(enb); % Resource grid dimensions
    L = griddims(2);                     % Number of OFDM symbols in a subframe 
    % OFDM demodulate signal 
    rxgrid = lteOFDMDemodulate(enb, downsampled);
    LTE_struct.rxgrid = rxgrid;
    if (isempty(rxgrid))
        fprintf('After timing synchronization, signal is shorter than one subframe so no further demodulation will be performed.\n');
        return;
    end
    
    [hest, nest] = lteDLChannelEstimate(enb, cec, rxgrid(:,1:L,:));
    
    
    %% PBCH Demodulation, BCH Decoding, MIB Parsing
    
    % Decode the MIB
    fprintf('Performing MIB decoding...\n');
    pbchIndices = ltePBCHIndices(enb);
    [pbchRx, pbchHest] = lteExtractResources( ...
        pbchIndices, rxgrid(:,1:L,:), hest(:,1:L,:,:));
    
    % Decode PBCH
    [bchBits, bchBits_before_fec, pbchSymbols, nfmod4, mib, enb.CellRefP] = ltePBCHDecode_edited( ...
        enb, pbchRx, pbchHest, nest); 

    figure()
    scatter(real(pbchSymbols), imag(pbchSymbols))     % Comment out to display graph
    % scatter(real(pbchHest), imag(pbchHest))

    % Parse MIB bits
    enb = lteMIB(mib, enb); 
    
    enb.NFrame = enb.NFrame+nfmod4;
    
    % Display cell wide settings after MIB decoding
    fprintf('Cell-wide settings after MIB decoding:\n');
    disp(enb);
    
    if (enb.CellRefP==0)
        fprintf('MIB decoding failed (enb.CellRefP=0).\n\n');
        return;
    end
    if (enb.NDLRB==0)
        fprintf('MIB decoding failed (enb.NDLRB=0).\n\n');
        return;
    end
    
    try
    
        resourceGrid = lteDLResourceGrid(enb);
    
        rsAnt0 = lteCellRS(enb,0);
        indAnt0 = lteCellRSIndices(enb,0);
        resourceGrid(indAnt0) = rsAnt0;
    
        pss_scale = 0.2;
        sss_scale = 0.4;
        pdsch_scale = 0.8;
    
        pss = ltePSS(enb);
        pss_arrayIndex = 0:length(pss)-1;
        pss_sym_ind = ltePSSIndices(enb,0,{'1based','re'});
        resourceGrid(pss_sym_ind) = pss_scale;
    
        sss = ltePSS(enb);
        sss_arrayIndex = 0:length(sss)-1;
        sss_sym_ind = lteSSSIndices(enb,0,{'1based','re'});
        resourceGrid(sss_sym_ind) = sss_scale;

        % pdsch = ltePDSCH(enb);
        % pdsch_arrayIndex = 0:length(pdsch)-1;
        % pdsch_sym_ind = ltePDSCHIndices(enb,0,{'1based','re'});
        % resourceGrid(pdsch_sym_ind) = pdsch_scale;
    
        % ----------------------------------------------------------------------
        % % Script to display LTE resource Block
        % xStep = 0:13;
        % yStep = 0:(enb.NDLRB*12-1);
        % figure;
        % surface(xStep,yStep,abs(resourceGrid));
        % axis([0 13 0 (enb.NDLRB*12-1) 0 1]);
        % view([0,90]);
        % set(gca,'xtick',[0 1 2 3 4 5 6 7 8 9 10 11 12 13]);
        % set(gca,'ytick',[[0:15:enb.NDLRB*15-1] [enb.NDLRB*15-1]]);
        %-----------------------------------------------------------------------
        disp('Succesful LTEResourceGridPlot')
        % disp(size(resourceGrid));
        % disp(yStep);
        % pause;
    catch exception
        disp('LTEResourceGrid Failed')
        disp(exception.message)
    end
    
    %% OFDM Demodulation on Full Bandwidth
    
    fprintf('Restarting reception now that bandwidth (NDLRB=%d) is known...\n',enb.NDLRB);
    
    % Resample now we know the true bandwidth
    ofdmInfo = lteOFDMInfo(enb);
    if (sr~=ofdmInfo.SamplingRate)
        if (sr < ofdmInfo.SamplingRate)
            warning('The received signal sampling rate (%0.3fMs/s) is lower than the desired sampling rate for NDLRB=%d (%0.3fMs/s); PDCCH search / SIB1 decoding may fail.',sr/1e6,enb.NDLRB,ofdmInfo.SamplingRate/1e6);
        end    
        fprintf('\nResampling from %0.3fMs/s to %0.3fMs/s...\n',sr/1e6,ofdmInfo.SamplingRate/1e6);
    else
        fprintf('\nResampling not required; received signal is at desired sampling rate for NDLRB=%d (%0.3fMs/s).\n',enb.NDLRB,sr/1e6);
    end
    nSamples = ceil(ofdmInfo.SamplingRate/round(sr)*size(eNodeBOutput,1));
    resampled = zeros(nSamples, nRxAnts);
    for i = 1:nRxAnts
        resampled(:,i) = resample(eNodeBOutput(:,i), ofdmInfo.SamplingRate, round(sr));
    end
    
    % Perform frequency offset estimation and correction
    fprintf('\nPerforming frequency offset estimation...\n');
    delta_f = lteFrequencyOffset(enb, resampled);
    fprintf('Frequency offset: %0.3fHz\n',delta_f);
    resampled = lteFrequencyCorrect(enb, resampled, delta_f);
    
    % Find beginning of frame
    fprintf('\nPerforming timing offset estimation...\n');
    offset = lteDLFrameOffset(enb, resampled); 
    fprintf('Timing offset to frame start: %d samples\n',offset);
    % Aligning signal with the start of the frame
    resampled = resampled(1+offset:end,:);   
    
    % OFDM demodulation
    fprintf('\nPerforming OFDM demodulation...\n\n');
    rxgrid = lteOFDMDemodulate(enb, resampled);   
    
    LTE_struct.rxgrid_2 = rxgrid;
    %% SIB1 Decoding
    
    if (mod(enb.NFrame,2)~=0)                    
        if (size(rxgrid,2)>=(L*10))
            rxgrid(:,1:(L*10),:) = [];   
            fprintf('Skipping frame %d (odd frame number does not contain SIB1).\n\n',enb.NFrame);
        else        
            rxgrid = [];
        end
        enb.NFrame = enb.NFrame + 1;
    end
    
    % Advance to subframe 5, or terminate if we have less than 5 subframes  
    if (size(rxgrid,2)>=(L*5))
        rxgrid(:,1:(L*5),:) = [];   % Remove subframes 0 to 4        
    else    
        rxgrid = [];
    end
    enb.NSubframe = 5;
    
    if (isempty(rxgrid))
        fprintf('Received signal does not contain a subframe carrying SIB1.\n\n');
    end
    
    % Reset the HARQ buffers
    decState = [];
    
    % While we have more data left, attempt to decode SIB1
    while (size(rxgrid,2) > 0)
    
        fprintf('%s\n',separator);
        fprintf('SIB1 decoding for frame %d\n',mod(enb.NFrame,1024));
        fprintf('%s\n\n',separator);
    
        % Reset the HARQ buffer with each new set of 8 frames as the SIB1
        % info may be different
        if (mod(enb.NFrame,8)==0)
            fprintf('Resetting HARQ buffers.\n\n');
            decState = [];
        end
    
        % Extract current subframe
        rxsubframe = rxgrid(:,1:L,:);
        
        % Perform channel estimation
        [hest,nest] = lteDLChannelEstimate(enb, cec, rxsubframe);    
        
        fprintf('Decoding CFI...\n\n');
        pcfichIndices = ltePCFICHIndices(enb);  % Get PCFICH indices
        [pcfichRx, pcfichHest] = lteExtractResources(pcfichIndices, rxsubframe, hest);
        % Decode PCFICH
        cfiBits = ltePCFICHDecode(enb, pcfichRx, pcfichHest, nest);
        cfi = lteCFIDecode(cfiBits); % Get CFI
        if (isfield(enb,'CFI') && cfi~=enb.CFI)
            release(pdcchConstDiagram);
        end
        enb.CFI = cfi;
        fprintf('Decoded CFI value: %d\n\n', enb.CFI);   
        
        if (strcmpi(enb.DuplexMode,'TDD'))
            tddConfigs = [1 6 0];
        else
            tddConfigs = 0; % not used for FDD, only used to control while loop
        end    
        alldci = {};
        while (isempty(alldci) && ~isempty(tddConfigs))
            % Configure TDD uplink-downlink configuration
            if (strcmpi(enb.DuplexMode,'TDD'))
                enb.TDDConfig = tddConfigs(1);
            end
            tddConfigs(1) = [];        
            pdcchIndices = ltePDCCHIndices(enb); % Get PDCCH indices
            [pdcchRx, pdcchHest] = lteExtractResources(pdcchIndices, rxsubframe, hest);
            % Decode PDCCH and plot constellation
            [dciBits, pdcchSymbols] = ltePDCCHDecode(enb, pdcchRx, pdcchHest, nest);
            pdcchConstDiagram(pdcchSymbols);        % Comment out to remove display graphs
    
            fprintf('PDCCH search for SI-RNTI...\n\n');
            pdcch = struct('RNTI', 65535);  
            pdcch.ControlChannelType = 'PDCCH';
            pdcch.EnableCarrierIndication = 'Off';
            pdcch.SearchSpace = 'Common';
            pdcch.EnableMultipleCSIRequest = 'Off';
            pdcch.EnableSRSRequest = 'Off';
            pdcch.NTxAnts = 1;
            alldci = ltePDCCHSearch(enb, pdcch, dciBits); % Search PDCCH for DCI                
        end
        
        % If DCI was decoded, proceed with decoding PDSCH / DL-SCH
        for i = 1:numel(alldci)
            
            dci = alldci{i};
            fprintf('DCI message with SI-RNTI:\n');
            disp(dci);
            % Get the PDSCH configuration from the DCI
            [pdsch, trblklen] = hPDSCHConfiguration(enb, dci, pdcch.RNTI);
            
            % If a PDSCH configuration was created, proceed with decoding PDSCH
            % / DL-SCH
            if ~isempty(pdsch)
                
                pdsch.NTurboDecIts = 5;
                fprintf('PDSCH settings after DCI decoding:\n');
                disp(pdsch);
    
                fprintf('Decoding SIB1...\n\n');        
                % Get PDSCH indices
                [pdschIndices,pdschIndicesInfo] = ltePDSCHIndices(enb, pdsch, pdsch.PRBSet);
                [pdschRx, pdschHest] = lteExtractResources(pdschIndices, rxsubframe, hest);
                % Decode PDSCH 
                [dlschBits,pdschSymbols] = ltePDSCHDecode(enb, pdsch, pdschRx, pdschHest, nest);
                % Decode DL-SCH with soft buffer input/output for HARQ combining
                if ~isempty(decState)
                    fprintf('Recombining with previous transmission.\n\n');
                end        
                [sib1, crc, decState] = lteDLSCHDecode(enb, pdsch, trblklen, dlschBits, decState);
    
                % Compute PDSCH EVM
                recoded = lteDLSCH(enb, pdsch, pdschIndicesInfo.G, sib1);
                remod = ltePDSCH(enb, pdsch, recoded);
                [~,refSymbols] = ltePDSCHDecode(enb, pdsch, remod);
                [rmsevm,peakevm] = pdschEVM(refSymbols{1}, pdschSymbols{1});
                fprintf('PDSCH RMS EVM: %0.3f%%\n',rmsevm);
                fprintf('PDSCH Peak EVM: %0.3f%%\n\n',peakevm);
    
                fprintf('SIB1 CRC: %d\n',crc);
                if crc == 0
                    fprintf('Successful SIB1 recovery.\n\n');
                else
                    fprintf('SIB1 decoding failed.\n\n');
                end
                
            else
                % Indicate that creating a PDSCH configuration from the DCI
                % message failed
                fprintf('Creating PDSCH configuration from DCI message failed.\n\n');
            end
            
        end
        if (numel(alldci)==0)
            % Indicate that DCI decoding failed 
            fprintf('DCI decoding failed.\n\n');
        end
        
        % Update channel estimate plot 
        figure(channelFigure);                          % Comment out to remove display graphs
        surf(abs(hest(:,:,1,1)));                       % Comment out to remove display graphs
        hSIB1RecoveryExamplePlots(channelFigure);       % Comment out to remove display graphs
        % channelFigure.CurrentAxes.XLim = [0 size(hest,2)+1];
        % channelFigure.CurrentAxes.YLim = [0 size(hest,1)+1];

        % eqGrid = lteEqualizeMMSE(rxsubframe,hest,nest);
        % figure(channelFigure);                          % Equalization Sauce
        % surf(abs(eqrid(:,:,1,1)));                       % Comment out to remove display graphs
        % hSIB1RecoveryExamplePlots(channelFigure); 

        % Skip 2 frames and try SIB1 decoding again, or terminate if we
        % have less than 2 frames left. 
        if (size(rxgrid,2)>=(L*20))
            rxgrid(:,1:(L*20),:) = [];   % Remove 2 more frames
        else
            rxgrid = []; % Less than 2 frames left
        end
        enb.NFrame = mod(enb.NFrame + 2,1024);
            
    end
    
    fprintf('Done-Done\n\n');

end
