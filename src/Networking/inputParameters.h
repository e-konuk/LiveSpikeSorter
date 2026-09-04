#ifndef INPUTPARAMETERS_H
#define INPUTPARAMETERS_H

typedef unsigned short  uint16;

#include <string>
#include <vector>
#include <map>
#include <limits>

struct InputParameters {
	std::string					sInputFolder{},
								sImecFile{},
								sNidqFile{},
								sSpikesFile{},
								sLogFile{},
								sEventFile{},
								sDataAccquisitionHost{},
								sDecoderWorkFolder{},
								sDecoderInputFolder{},
								sOSSOutputFolder{},
								sdmIP{},
								sdmProcessorType{ "zscore" },
								sdmMode{ "median" },
								sdmStatsPath{},
								sdmRsFsPath{},
								sSdmSpikesFile{},
								sSdmEventFile{},
								sSdmDecoderWorkFolder{};

	uint16						uDataAccquisitionPort{ 0 },
								uSelectedDevice{ 0 };

	std::vector<uint16>			vSelectedDevices,
								vChannelSubset;
	std::vector<long>			vSdmActivitySubset{};

	std::map<uint16, std::string>	mapDeviceFilePaths,
								mapOSSOutputFolders,
								mapDecoderInputFolders,
								mapSpikeFiles;

	double						dTau,
								dThreshold,
								dRatioToMax;

	float						sdmTriggerZ{ 1.0f },
								sdmBaselineMinSeconds{ 10.0f },
								sdmOffset{ 0.0f };

	// Per-population (FS/RS) thresholds used for closed loop demo.
	//   median mode: FS low  if pop < fsMedian - sdmOffsetFsLow
	//                FS high if pop > fsMedian + sdmOffsetFsHigh  (RS analogous)
	//   zscore mode: FS low  if z   < -sdmTriggerZFsLow
	//                FS high if z   >  sdmTriggerZFsHigh          (RS analogous)
	float						sdmOffsetFsLow{ std::numeric_limits<float>::quiet_NaN() },
								sdmOffsetFsHigh{ std::numeric_limits<float>::quiet_NaN() },
								sdmOffsetRsLow{ std::numeric_limits<float>::quiet_NaN() },
								sdmOffsetRsHigh{ std::numeric_limits<float>::quiet_NaN() },
								sdmTriggerZFsLow{ std::numeric_limits<float>::quiet_NaN() },
								sdmTriggerZFsHigh{ std::numeric_limits<float>::quiet_NaN() },
								sdmTriggerZRsLow{ std::numeric_limits<float>::quiet_NaN() },
								sdmTriggerZRsHigh{ std::numeric_limits<float>::quiet_NaN() };
	int							sdmTriggerBinMs{ 50 },
								sdmDecoderWindowMs{ 300 };

	float						fImecSamplingRate,
								fNidqSamplingRate,
								fThresholdStd;

	int							iSubstream,
								iNidqRefreshRate,
								iMinScanWindow,
								iMaxScanWindow,
								iConvolutionTimes,
								iDownsampling,
								iMaxIts,
								iTimeBehind,
								iAvgWindowTime,
								iRedundancy,
								iWindowLength,
								iBinLength,
								iWindowOffset,
								iSorterType,
								iNumTemplates;

	bool						bReadFromFile,
								bIsDecoding,
								bIsSendingFeedback,
								bSmallskip;

	uint16_t					sdmPort{};

	// Skip the ImGui input window and use CLI-populated params directly
	bool						bSkipInputGui{ false };

	// For (live) (rigid) drift estimation
	bool						bDriftEstimation{ false };
	float						fDriftWindowSeconds{ 10.0f };
	float						fDriftMaxShiftUm{ 50.0f };
	// Suggestion threshold (for retraining)
	float						fDriftRetrainThresholdUm{ 50.0f };
};
#endif