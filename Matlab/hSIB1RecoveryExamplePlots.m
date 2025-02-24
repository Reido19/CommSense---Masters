% hSIB1RecoveryExamplePlots configure plots for SIB1RecoveryExample

% Copyright 2014-2022 The MathWorks, Inc.

function [spectrumScope,synchCorrPlot,pdcchConstDiagram] = hSIB1RecoveryExamplePlots(channelFigure,varargin)
    persistent spectrumHandle synchPlot  pdcchPlot;
    if (nargin==2)
        
        sr = varargin{1};
        
        % plot sizing and positioning
        [repositionPlots,plotPositions] = hPlotPositions();

        % Received signal spectrum
        spectrumHandle = spectrumAnalyzer();
        spectrumHandle.Name = 'Received signal spectrum';
        spectrumHandle.SampleRate = sr;
        spectrumHandle.PlotMaxHoldTrace = true;
        spectrumHandle.PlotMinHoldTrace = true;
        spectrumHandle.ShowGrid = true;
        if (repositionPlots)
            spectrumHandle.Position = plotPositions(1,:);
        end
        spectrumScope = spectrumHandle;

        % PSS/SSS correlation 
        synchPlot = dsp.ArrayPlot();
        synchPlot.Name = 'PSS/SSS correlation';
        synchPlot.XLabel = 'Timing offset (samples)';
        synchPlot.YLabel = 'Correlation level';
        synchPlot.PlotType = 'Line';
        synchPlot.ShowGrid = true;
        if (repositionPlots)
            synchPlot.Position = plotPositions(2,:);
        end
        synchCorrPlot = synchPlot;

        % PDCCH constellation
        pdcchPlot = comm.ConstellationDiagram();
        pdcchPlot.Name = 'PDCCH constellation';
        pdcchPlot.ShowReferenceConstellation = true; % Changed from False
        pdcchPlot.ShowGrid = true;
        if (repositionPlots)
            pdcchPlot.Position = plotPositions(3,:);
        end
        pdcchConstDiagram = pdcchPlot;
        
        % Channel magnitude response
        channelFigure.Name = 'Channel magnitude response';
        channelFigure.NumberTitle = 'off';
        channelFigure.Color = [40 40 40]/255;   
        channelFigure.Visible = 'off';
        if (repositionPlots)
            channelFigure.Position = plotPositions(4,:);      
        end
        
    else
        
        figure(channelFigure);
        shading flat;
        channelFigure.CurrentAxes.XColor = [175 175 175]/255;
        channelFigure.CurrentAxes.YColor = [175 175 175]/255;
        channelFigure.CurrentAxes.ZColor = [175 175 175]/255;
        channelFigure.CurrentAxes.Color = [0 0 0];
        channelFigure.CurrentAxes.XGrid = 'on';
        channelFigure.CurrentAxes.YGrid = 'on';
        channelFigure.CurrentAxes.ZGrid = 'on';
        channelFigure.CurrentAxes.XLabel.String = 'OFDM symbols';
        channelFigure.CurrentAxes.XLabel.FontSize = 8;
        channelFigure.CurrentAxes.YLabel.String = 'Subcarriers';
        channelFigure.CurrentAxes.YLabel.FontSize = 8;
        channelFigure.CurrentAxes.ZLabel.String = 'Magnitude';
        channelFigure.CurrentAxes.ZLabel.FontSize = 8;
        channelFigure.CurrentAxes.View = [-30 60];
        
    end

end
