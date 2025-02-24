%ltePBCHDecode Physical broadcast channel decoding
%   [BITS,SYMBOLS,NFMOD4,TRBLK,CELLREFP] = ltePBCHDecode(...) returns a
%   vector of soft bits BITS and received constellation of complex symbols
%   vector SYMBOLS resulting from performing the inverse of Physical
%   Broadcast Channel (PBCH) processing (TS 36.211 6.6, see <a href="matlab: help('ltePBCH')">ltePBCH</a> for 
%   details). Frame number modulo 4 NFMOD4, decoded BCH information bits
%   TRBLK and number of cell-specific reference signal antenna ports
%   CELLREFP are decoded by performing the inverse of Broadcast Channel
%   (BCH) processing (TS 36.212 5.3.1, see <a href="matlab:help('lteBCH')">lteBCH</a> for details). BITS is 
%   optionally scaled by channel state information (CSI) calculated during
%   the equalization process.
%   
%   [BITS,SYMBOLS,NFMOD4,TRBLK,CELLREFP] = ltePBCHDecode(ENB,SYM) performs
%   the inverse of Physical Broadcast Channel (PBCH) processing on the
%   matrix of complex modulated PBCH symbols SYM and cell-wide settings
%   structure ENB. The channel inverse processing includes deprecoding,
%   symbol demodulation and descrambling.
%   
%   ENB must be a structure including the fields:
%   NCellID      - Physical layer cell identity
%   CellRefP     - Optional. Number of cell-specific reference signal 
%                  antenna ports (1,2,4) (default is to establish CellRefP
%                  by decoding the input symbols SYM). 
%   CyclicPrefix - Optional. Cyclic prefix length 
%                  ('Normal'(default),'Extended')
%   
%   SYM must be a matrix of NRE-by-NRxAnts complex modulated PBCH symbols.
%   NRE is the number of QPSK symbols per antenna assigned to the PBCH and
%   NRxAnts is the number of receive antennas. SYM can contain 1 to 4
%   subframes worth of PBCH data. When multiple subframes are provided,
%   they must be consecutive subframes within the same coded BCH block.
%
%   BITS is a vector of soft bits and SYMBOLS is a vector of received
%   constellation complex symbols. NFMOD4 is the system frame number modulo
%   4 (i.e. mod(NFrame,4)) obtained when determining the scrambling phase
%   of the input PBCH symbols SYM. TRBLK is the decoded BCH information
%   bits (24 bits) and CELLREFP is the number of cell-specific reference
%   signal antenna ports determined during the BCH decoding process. The
%   NFMOD4, TRBLK and CELLREFP outputs are determined by successful
%   synchronization with the scrambling sequence, which gets initialized
%   every 40msec. The true number of transmitted cell-specific reference
%   signals will be returned in output CELLREFP, and is searched for by
%   attempting decoding with each of CELLREFP={1 2 4}. ENB.CellRefP will be
%   attempted first; this ensures that the SYMBOLS output contains the
%   expected constellation and BITS contains the expected soft bit
%   estimates for the specified value of ENB.CellRefP (decoding may succeed
%   with a different value of CELLREFP under good conditions, but will give
%   unexpected BITS and SYMBOLS outputs). Note that ENB.CellRefP is
%   optional; if it is not provided, the true number of transmitted
%   cell-specific reference signals will be established by the search and
%   returned in CELLREFP. If the function detects a cyclic redundancy check
%   (CRC) error, it returns the CELLREFP output as 0.
%
%   [BITS,SYMBOLS,NFMOD4,TRBLK,CELLREFP] = ...
%   ltePBCHDecode(ENB,SYM,HEST,NOISEEST)
%   performs the decoding of the complex PBCH symbols SYM using cell-wide
%   settings ENB, the channel estimate HEST and the noise estimate
%   NOISEEST. For the TxDiversity transmission scheme (CELLREFP=2 or
%   CELLREFP=4), the reception is performed using an OSFBC (Orthogonal
%   Space Frequency Block Code) decoder and for the Port0 transmission
%   scheme (CELLREFP=1), the reception is performed using MMSE
%   equalization.
%
%   HEST is a 3-dimensional NRE-by-NRxAnts-by-P array where NRE are the
%   frequency and time locations corresponding to the PBCH RE positions (a
%   total of NRE positions), NRxAnts is the number of receive antennas, and
%   P is the number of cell-specific reference signal antennas.
%   
%   NOISEEST is an estimate of the noise power spectral density per RE on
%   received subframe; such an estimate is provided by the 
%   <a href="matlab: help('lteDLChannelEstimate')">lteDLChannelEstimate</a> function.
%   
%   [BITS,SYMBOLS,NFMOD4,TRBLK,CELLREFP] = ...
%   ltePBCHDecode(ENB,SYM,HEST,NOISEEST,ALG) 
%   is the same as above except it provides control over weighting the
%   output soft bits BITS with Channel State Information (CSI) calculated
%   during the equalization stage using algorithmic configuration structure
%   ALG.
%   
%   ALG must be a structure including the field:
%   CSI - Optional. Determines if soft bits should be weighted by CSI 
%         ('Off','On'(default))
%   
%   Example:
%   % Decode the number of cell-specific reference ports from the Master
%   % Information Block (MIB):
%
%   enb = lteRMCDL('R.14');
%   mib = lteMIB(enb);
%   bchBits = lteBCH(enb,mib);
%   quarterLen = length(bchBits)/4;
%   pbchSymbols = ltePBCH(enb,bchBits(1:quarterLen));
%   [bits,symbols,nfmod4,trblk,cellrefp] = ltePBCHDecode(enb,pbchSymbols);
%   cellrefp
%
%   See also ltePBCH, ltePBCHIndices, ltePBCHPRBS, lteBCHDecode.

%   Copyright 2010-2019 The MathWorks, Inc.

function [bits, bits_before_fec,symbols,nfmod4,trblk,cellrefp] = ltePBCHDecode_edited(enb,sym,varargin)
       
    % Provide default value for CSI field if absent
    if(nargin>4)
        alg = varargin{3};
    else
        alg = struct();
    end
    alg = mwltelibrary('validateLTEParameters',alg,'CSI','optionalnowarning');
    
    cellrefps=[1 2 4];
    if (isfield(enb,'CellRefP'))
        cellrefps(find(cellrefps==enb.CellRefP,1))=[];
        cellrefps=[enb.CellRefP cellrefps];
    end
    
    for c=cellrefps
        
        enb.CellRefP=c;
        
        if(nargin == 2)

            if (enb.CellRefP==2 || enb.CellRefP==4)
                symbols = lteTransmitDiversityDecode(sym,ones(size(sym,1),size(sym,2),enb.CellRefP)/size(sym,2));
            else
                symbols = lteEqualizeZF(sym,ones(size(sym,1),size(sym,2),1));  
            end        
            csi=ones(size(symbols));

        elseif(nargin >= 4)

            hest=varargin{1};
            noiseEst=varargin{2};        
            if((enb.CellRefP==2 || enb.CellRefP==4) && size(hest,3)>1)                       
                [symbols,csi] = lteTransmitDiversityDecode(sym,hest(:,:,1:min(c,size(hest,3))));            
            else            
                [symbols,csi] = lteEqualizeMMSE(sym,hest(:,:,1:min(c,size(hest,3))),noiseEst);
            end

        end

        % Soft demodulation of rec bits
        demod = lteSymbolDemodulate(symbols,'QPSK','Soft');
        bits_before_fec = lteSymbolDemodulate(symbols, 'QPSK', 'Hard');
        Qm=2;
        csi=repmat(csi.',Qm,1);
        csi=reshape(csi,numel(csi),1);

        % decoding/search
        [bits,nfmod4,trblk,cellrefp] = decode(enb, demod, alg, csi);    
        if (cellrefp~=0)
            break;
        end
        
    end
    
end

function [sbits,nfmod4,trblk,cellrefp] = decode(enb, demod, alg, csi)
      
    % Calculate the actual 40ms period in coded bits
    if isfield(enb,'CyclicPrefix') && strcmpi(enb.CyclicPrefix,'Extended')
        period = 1728;
    else
        period = 1920;
    end
        
    % Create periodic scrambling sequence that will cover the data under 4 period/4 shifts
    descramblingSeq = repmat(ltePBCHPRBS(enb,period,'signed'), ceil(length(demod)/period) + 1, 1);
            
    % Start decoding process
    period = period/4;
    for nfmod4=3:-1:0
        sbits = demod.*descramblingSeq(nfmod4*period + 1: nfmod4*period + length(demod));
        % Scaling LLRs by CSI
        if (strcmpi(alg.CSI,'On'))
            sbits=sbits.*csi;    
        end
        % Prepend LLR data with zeros to account for rate matching offset
        % associated with subframes (relevant to extended cyclic prefix numerology)
        [trblk, cellrefp] = lteBCHDecode(enb,[zeros(mod(nfmod4*period,120),1); sbits]);
        if cellrefp ~= 0
            break;
        end
    end
    
end
