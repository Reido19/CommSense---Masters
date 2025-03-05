
function eNodeBOutput=Band3Collector() 

radiolist = findsdru;
radio = comm.SDRuReceiver('Platform',radiolist.Platform, 'IPAddress', "192.168.10.2"); 
radio.MasterClockRate = 100e6;
radio.DecimationFactor = 4;
radio.ChannelMapping = [1:1];     % Receive signals from both channels
radio.CenterFrequency = 1862.10009e6; %874.3MHz, 1810.1MHz
radio.Gain = 0;
radio.SamplesPerFrame = 250000; % Sampling rate is 1.92 MHz. LTE frames are 10 ms long
radio.OutputDataType = 'double';
radio.EnableBurstMode = true;
radio.NumFramesInBurst = 10; %4;
radio.OverrunOutputPort = true;
info(radio)
%% Capture Signal

burstCaptures = zeros(250000,10,1);

len = 0;
for frame = 1:10
    while len == 0
        [data,len,lostSamples] = step(radio);
        burstCaptures(:,frame,:) = data;
    end
    len = 0;
end
release(radio);

% disp(size(burstCaptures));
fprintf('Size of burst Capture: \n');
eNodeBOutput = reshape(burstCaptures,[],1);

end

