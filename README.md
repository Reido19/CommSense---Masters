# CommSense---Masters
Backup and version control for CommSense Masters Project


Pipeline structure:

	Collect_Bin.m   
		-Control file, running this file should call everything else.
		-Writes results to various formats (bin, csv, raw) stored at the specified locations
	
	Band3Collector.m
		- Interacts with the USRP N210
		- Sets all the radio parameters 
	
	SIB1RecoveryExample_edited.m 
		- Slightly edited version of the SIB1 Recovery Example Script from Matlab
		- Demodulates the LTE data and returns channel estimations.
		
	Other files
		- data and plotting files usually called from SIB1RecoveryExample_edited.m
	
	
	To switch between various data capturing modes see commented out lines in Collect_Bin.mat  (19-26)
