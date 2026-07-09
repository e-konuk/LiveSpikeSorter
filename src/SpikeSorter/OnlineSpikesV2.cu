#include "OnlineSpikesV2.h"

#include "../Helpers/Utils.h"

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <iterator>
#include <chrono>
#include <queue>
#include <algorithm>
#include <typeinfo>
#include <vector>
#include <thread>
#include "../Helpers/Timer.h"
#include <fstream>
#include <experimental/filesystem>

#ifdef WINDOWS
#include <windows.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

//#include <device_functions.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cufft.h>
#include "../Helpers/TimeHelpers.h"
#include "../Networking/onlineSpikesPayload.h"
#include "../Networking/NetworkHelpers.h"
#include "../Networking/FragmentManager.h"
#include "../NetClient/NetClient.h"
#include "myCudnnConvolution.h"
#include "SorterHelpers.h"

#include <atomic>
#include <mutex>
#include <numeric>

#ifdef WINDOWS
#ifndef _WIN_CLOCK
#define _WIN_CLOCK
#endif
#endif

#define BLOCK_SIZE      16
#define DEFAULT_TPB		256 // CUDA kernel default threads per block

#define _CUBLAS_CALL(call, errorMessage) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        char errMsg[256]; \
        snprintf(errMsg, sizeof(errMsg), "%s (%s at %s:%d)", \
                 errorMessage, cublasGetErrorString(status), __FILE__, __LINE__); \
        _RUN_ERROR(ptLabel, errMsg); \
    } \
} while(0)

// Helper function to convert cuBLAS error codes to strings
const char* cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
	default: return "Unknown cuBLAS error";
	}
}

__global__ void fwd_max_pool_1d_kernel(float* __restrict__ d_matrix, float* __restrict__ d_output, int W, int M) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < W) {
		float maxVal = -FLT_MAX;

		for (int offset = -M; offset <= M; offset++) {
			int newIdx = idx + offset;
			if (newIdx < 0 || newIdx >= W) continue;
			float entry = d_matrix[newIdx];
			if (entry > maxVal) maxVal = entry;
		}

		d_output[idx] = maxVal;
	}
}

__global__ void compute_amps_kernel(const float* __restrict__ d_B, const float* __restrict__ d_nm, const long* __restrict__ d_spikeIndices, float* d_result, long numSpikes, long T, long W) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numSpikes) {
		long spikeIdx = d_spikeIndices[idx];
		long sampleIdx = spikeIdx % W;
		long templateIdx = spikeIdx / W;
		d_result[idx] = d_B[templateIdx * W + sampleIdx] * d_nm[templateIdx];
	}
}

__global__ void compute_xfeat_kernel(
	int numSpikes,
	int numNearestChans,
	long M,
	long K,
	long unclu_T,
	long C,
	const int64_t* __restrict__ d_iCC,
	const int64_t* __restrict__ d_iU,
	const float* __restrict__ d_Ucc,
	const float* __restrict__ d_Xres,
	int currBatchNumSamples,
	const float* __restrict__ d_wPCA_transposed,
	const long* __restrict__ d_spikeTimes,
	const long* __restrict__ d_spikeTemplates,
	const float* __restrict__ d_amps,
	float* __restrict__ d_xfeat)
{
	// Each thread works on one (spike, nearest-channel) pair.
	int i = blockIdx.y * blockDim.y + threadIdx.y; // spike index
	int j = blockIdx.x * blockDim.x + threadIdx.x; // nearest channel index

	if (i < numSpikes && j < numNearestChans) {
		// Get the spike template for this spike.
		long tmpl = d_spikeTemplates[i];
		// Use d_iU to get an intermediate index into d_iCC.
		long mostActiveChan = d_iU[tmpl];
		// Get the actual channel index for this nearest channel:
		// d_iCC is stored as [numNearestChans, n_templates_iU] in row-major order.
		long channel_extracted = d_iCC[mostActiveChan + j * C];

		// Precompute half the window length (integer division).
		long half_M = M / 2;

		// Loop over principal components.
		for (long k = 0; k < K; k++) {
			float dot = 0.0f;
			// Loop over the waveform samples.
			for (long r = 0; r < M; r++) {
				// Compute the time index: spike time plus offset (r - half_M).
				long time_index = d_spikeTimes[i] + (r - half_M);
				// Access the value from d_Xres.
				// d_Xres is assumed stored as a 2D array [numChannels x currBatchNumSamples] (row-major).
				float x_val = d_Xres[channel_extracted * currBatchNumSamples + time_index];
				// d_wPCA_transposed is stored as [M x K] (row-major); its element at row r, col k is:
				float w_val = d_wPCA_transposed[r * K + k];
				dot += x_val * w_val;
			}
			// Compute the offset term using d_Ucc.
			// d_Ucc is assumed stored as a 3D array of shape [numNearestChans, n_templates, K] in row-major order.
			// Indexing: (j, tmpl, k) -> j + tmpl*numNearestChans + k*numNearestChans.
			float offset = d_amps[i] * d_Ucc[k + tmpl * K + j * unclu_T * K];

			// Write the result into d_xfeat.
			// d_xfeat is stored as [numSpikes x numNearestChans x K] in row-major order:
			// index = i * (numNearestChans * K) + j * K + k.
			d_xfeat[j * (numSpikes * K) + i * K + k] = dot + offset;
		}
	}
}

__global__ void transpose_xfeat(
	const float* __restrict__ d_xfeat,
	float* __restrict__ d_tF,
	long numNearestChans,
	long numSpikes,
	long K)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < K * numSpikes * numNearestChans) {
		int pcIdx = idx % K;
		int spikeIdx = (idx / K) % numSpikes;
		int chanIdx = idx / (numSpikes * K);
		d_tF[pcIdx + chanIdx * K + spikeIdx * K * numNearestChans] = d_xfeat[pcIdx + spikeIdx * K + chanIdx * K * numSpikes];
	}
}

__global__ void compute_spike_positions_kernel(
	int numSpikes,
	int numNearestChans,
	long K,
	const float* __restrict__ d_tF,          // shape: [numSpikes, numNearestChans, K]
	const long* __restrict__ d_spikeTemplates, // shape: [numSpikes]
	const int64_t* __restrict__ d_iU,             // shape: [numTemplates]
	const int64_t* __restrict__ d_iCC,            // shape: [numNearestChans, numTemplates] in row-major order
	const float* __restrict__ d_xc,            // channel x positions, length: numChannels
	const float* __restrict__ d_yc,            // channel y positions, length: numChannels
	float* __restrict__ d_xs,                  // output: spike x positions [numSpikes]
	float* __restrict__ d_ys,                  // output: spike y positions [numSpikes]
	long numTemplates,                         // total number of templates (for indexing purposes)
	long C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // spike index
	if (i < numSpikes) {
		// For spike i, retrieve its template index.
		long tmpl = d_spikeTemplates[i];
		// Get the corresponding index into d_iCC via d_iU.
		long iU_val = d_iU[tmpl];

		float total_mass = 0.0f;
		// Assume numNearestChans is small; use a fixed-size temporary array.
		const int MAX_NEAREST = 16;  // adjust if needed
		float mass_arr[MAX_NEAREST];

		// First pass: compute mass for each channel (sum over K PCs) and total mass.
		for (int j = 0; j < numNearestChans; j++) {
			float mass = 0.0f;
			for (long k = 0; k < K; k++) {
				// d_tF is stored in row-major order as:
				// index = i * (numNearestChans * K) + j * K + k
				long idx = i * (numNearestChans * K) + j * K + k;
				float val = d_tF[idx];
				mass += val * val;
			}
			mass_arr[j] = mass;
			total_mass += mass;
		}

		float xs_val = 0.0f;
		float ys_val = 0.0f;
		// Second pass: compute weighted sum of channel coordinates.
		// The channel for nearest channel j is determined by d_iCC:
		// channel = d_iCC[j + iU_val * numNearestChans]
		for (int j = 0; j < numNearestChans; j++) {
			float weight = (total_mass > 0.0f) ? (mass_arr[j] / total_mass) : 0.0f;
			long chan = d_iCC[iU_val + j * C];
			xs_val += d_xc[chan] * weight;
			ys_val += d_yc[chan] * weight;
		}
		d_xs[i] = xs_val;
		d_ys[i] = ys_val;
	}
}

// constructor (for ctrl-f purposes)
OnlineSpikesV2::OnlineSpikesV2(
	InputParameters params,
	sockaddr_in      mainAddr,
	DataSocket*      sharedSocket
)
	: cudnnConvObj(0),
	minWindow(params.iMinScanWindow),              // Minimum samples per window
	maxWindow(params.iMaxScanWindow),              // Maximum samples per window
	redundancy(params.iRedundancy),
	timeBehind(params.iTimeBehind),                // Allowed ms behind; 0 = skip, >100000 = never skip
	downsampling(params.iDownsampling),            // Temporal downsampling factor
	samplingRate(params.fImecSamplingRate / downsampling), // Derived Imec Hz
	nidqRefreshRate(params.iNidqRefreshRate),
	imecSock(Sock::UDP),
	nidqSock(Sock::UDP),
	imecFm(&imecSock),
	nidqFm(&nidqSock),
	cublasHandle(),                                // cuBLAS context
	sglxSock(sharedSocket),
	smallSkip(params.bSmallskip),
	p2p(0),
	W(params.iMinScanWindow + params.iMaxScanWindow),
	rootMeanSquared(0),
	ossOutputDir(params.sOSSOutputFolder),
	spikesFileOut(),
	recordingOffset(0),
	substream(params.iSubstream),
	lookback(0), // this will be set when we read in M
	m_driftEnabled(params.bDriftEstimation),
	m_deviceIndex(params.uSelectedDevice),
	m_driftWindowSeconds(params.fDriftWindowSeconds),
	m_driftMaxShiftUm(params.fDriftMaxShiftUm),
	m_estWindowSamples(0),
	m_windowStartCt(0),
	m_sigInterp(20.0f),
	m_binningDepth(5.0f),
	m_ycMin(0.0f),
	m_ycMax(0.0f),
	m_dshiftLast(0.0f),
	m_fpDmax(0),
	m_nAmpBins(20),
	m_driftTranspose(false),
	m_driftJobReady(false),
	m_driftStop(false),
	m_snapWindowEndCt(0),
	m_activeDriftMatrix(nullptr),
	m_totalShiftUm(0.0f),
	m_estDriftUm(0.0f),
	m_atomicDriftUpdateCt(0)
{
	static const char *ptLabel = { "OnlineSpikesV2::OnlineSpikesV2" };

	initializeSorter(params);
	establishDecoderConnection(mainAddr);
}

OnlineSpikesV2::~OnlineSpikesV2()
{
	static const char *ptLabel = { "OnlineSpikesV2::~OnlineSpikesV2" };

	// Stop the drift estimation worker before freeing device memory it may use
	if (m_driftWorker.joinable()) {
		{
			std::lock_guard<std::mutex> lk(m_driftMutex);
			m_driftStop = true;
		}
		m_driftCV.notify_all();
		m_driftWorker.join();
	}

	// Deallocate Host Memory
	#define X(type, name, memType, size) \
		if (memType == Host) { \
			delete[] name; \
			name = nullptr; \
		}
		MEMORY_VARIABLES
	#undef X

	// Deallocate Pinned Host Memory
	#define X(type, name, memType, size) \
		if (memType == Pinned) { \
			if (name) { \
				_CUDA_CALL(cudaFreeHost(name)); \
				name = nullptr; \
			} \
		}
		MEMORY_VARIABLES
	#undef X

	// Deallocate Device Memory
	#define X(type, name, memType, size) \
		if (memType == Device) { \
			if (name) { \
				_CUDA_CALL(cudaFree(name)); \
				name = nullptr; \
			} \
		}
		MEMORY_VARIABLES
	#undef X

	// Close files and perform other cleanup
	spikesFileOut.close();
}

void OnlineSpikesV2::initializeSorter(InputParameters params) {
	static const char* ptLabel = { "OnlineSpikesV2::initializeSorter" };
	// Ensure output directory exists and open the spike output file
	{
		namespace fs = std::experimental::filesystem;
		fs::path outputDir(ossOutputDir);
		if (!ossOutputDir.empty() && !fs::exists(outputDir)) {
			std::cout << "Creating output directory: " << ossOutputDir << std::endl;
			fs::create_directories(outputDir);
		}

		std::string spikesFilePath = ossOutputDir + "spikeOutput.txt";

		// If a previous spikeOutput.txt exists, rename it with an incremented suffix
		if (fs::exists(spikesFilePath)) {
			int suffix = 1;
			std::string renamedPath;
			do {
				renamedPath = ossOutputDir + "spikeOutput_" + std::to_string(suffix) + ".txt";
				suffix++;
			} while (fs::exists(renamedPath));
			std::cout << "Renaming existing " << spikesFilePath << " -> " << renamedPath << std::endl;
			fs::rename(spikesFilePath, renamedPath);
		}

		spikesFileOut.open(spikesFilePath);
		if (!spikesFileOut.is_open()) {
			std::cerr << "ERROR: Failed to open spike output file: " << spikesFilePath << std::endl;
			throw std::runtime_error("Failed to open spike output file: " + spikesFilePath);
		}
		std::cout << "Writing spikes to " << spikesFilePath << std::endl;
	}

	// Set CUDA device to the one that was chosen
	setDevice(params.uSelectedDevice, &cudnnConvObj);
	_CUDA_CALL(cudaSetDevice(params.uSelectedDevice));

	std::cout << "OSS started with device number " << params.uSelectedDevice << " and input directory "
		<< params.sInputFolder << std::endl;

	// Initialize cuBLAS handle
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "createCuBLAS: Failed to initialize cuBLAS handle");

	if (cusolverDnCreate(&cuSolverHandle) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "Failed to initialized cusolver handle");

	// Need to load C, M, T, unclu_T, K, and some KS params first before allocating memory
	loadTemplatesShape(params.sInputFolder + "templates.npy");
	loadKilosortParameters(params.sInputFolder);
	loadPreclusterShapes(params.sInputFolder + "Wall3.npy");

	// Allocate memory for Kilosort output, DO NOT MOVE THIS, as 
	// one must have the following variables defined in order for the X-macros to work:
	//    C, M, T, unclu_T, K, numNearestChans
	allocateMemory();
	initializeStaticMemory(params);

	// No idea what this does ngl
	for (int i = 0; i < C; i++) activeChannels.push_back(i);

	// --- Real-time drift estimation setup ---
	m_activeDriftMatrix.store(d_driftMatrix, std::memory_order_release);
	m_totalShiftUm = m_dshiftLast;

	if (m_driftEnabled) {
		loadDriftData(params.sInputFolder);   // may disable drift if tensors missing
	}
	if (m_driftEnabled) {
		m_estWindowSamples = (long)(m_driftWindowSeconds * samplingRate);
		if (m_estWindowSamples < 1) m_estWindowSamples = 1;
		m_fpDepths.reserve(1 << 16);
		m_fpAmps.reserve(1 << 16);
		m_driftWorker = std::thread(&OnlineSpikesV2::driftWorkerLoop, this);
		std::cout << "[Drift] Real-time drift estimation enabled: window "
		          << m_driftWindowSeconds << " s (" << m_estWindowSamples
		          << " samples), max shift " << m_driftMaxShiftUm << " um, "
		          << m_fpDmax << " depth bins." << std::endl;
	}
}

void OnlineSpikesV2::establishDecoderConnection(sockaddr_in mainAddr)
{
	// Connect to the main Server (so mainAddr can give decoder the sorter's addresses)
	sendConnectMsg(&imecSock, mainAddr, _SPIKE_SORTER_IMEC);
	sendConnectMsg(&nidqSock, mainAddr, _SPIKE_SORTER_NIDQ);

	// Receive connection from decoder to accquire decoder's addresses
	decoderImecAddr = recvConnectMsg(&imecSock, _DECODER_IMEC);
	decoderNidqAddr = recvConnectMsg(&nidqSock, _DECODER_NIDQ);

	// Start up assembler to handle larger packets
	std::thread imecFmThread = imecFm.assemblerThread();
	imecFmThread.detach();

	std::thread imecRetransThread = imecFm.retransmitterThread();
	imecRetransThread.detach();

	// Send SorterParams to Decoder
	SorterParameters sorterParams = getSorterParams();

	// Send parameters to decoder, which will be sent to OutputGUI
	sendPayload(&imecFm, sorterParams, decoderImecAddr);
}

// Google "X-macros" to get an idea of how this works. This function will automatically allocate and initialize memory
// for variables listed in OnlineSpikesV2MemoryList.h
void OnlineSpikesV2::allocateMemory()
{
	static const char *ptLabel = { "OnlineSpikesV2::allocateMemory" };

	int maxBufferSize;
	nppsMaxIndxGetBufferSize_32f(W * T, &maxBufferSize);

	int minBufferSize;
	nppsMaxIndxGetBufferSize_32f(W * T, &minBufferSize);

		// Host memory allocations
#define X(type, name, memType, size) \
        if (memType == Host) { \
            name = new type[size]; \
            std::memset(name, 0, sizeof(type) * size); \
        }
		MEMORY_VARIABLES
#undef X

		// Pinned Host memory allocations
#define X(type, name, memType, size) \
        if (memType == Pinned) { \
            _CUDA_CALL(cudaMallocHost((void**)&name, sizeof(type) * size)); \
            std::memset(name, 0, sizeof(type) * size); \
        }
		MEMORY_VARIABLES
#undef X

		// Device memory allocations
#define X(type, name, memType, size) \
        if (memType == Device) { \
            _CUDA_CALL(cudaMalloc((void**)&name, sizeof(type) * size)); \
        } \
		if (!name) { \
			std::cerr << "Failed to allocate " << size << " bytes for variable " << name << std::endl; \
		} 
		MEMORY_VARIABLES
#undef X
										
	templateMap.resize(T);
	channelMap.resize(C);
	lastSpikeTime.resize(T);
}

void OnlineSpikesV2::loadTemplatesShape(std::string filepath)
{
	std::cout << "Loading template shapes from directory " << filepath << std::endl;
	cnpy::NpyArray npTemplates = getTemplates(filepath);
	T = npTemplates.shape[0];
	M = npTemplates.shape[1];
	C = (npTemplates.shape.size() == 3) ? npTemplates.shape[2] : 1;
	lookback = 2 * M;
}

// TODO: Make this more "conventional"... idk put everything into a JSON instead of a text file or soemthing
void OnlineSpikesV2::loadKilosortParameters(std::string directoryPath)
{
	std::ifstream inf(directoryPath + "misc.txt");
	if (!inf.is_open()) {
		std::cerr << "Unable to open file " << directoryPath << "/data.txt" << std::endl;
		exit(EXIT_SUCCESS);
	}

	std::map<std::string, std::string> params;
	std::string line;

	while (std::getline(inf, line)) {
		size_t pos = line.find(":");
		std::string key = line.substr(0, pos);
		std::string value = line.substr(pos + 1);
		params[key] = value;
	}

	nt0min          = std::stoi(params["nt0min"]);
	numNearestChans = std::stoi(params["numNearestChans"]);
	Th_learned      = std::stoi(params["Th_learned"]);
	dt              = std::stoi(params["duplicate_spike_bins"]);

	// Drift-estimation parameters (if present)
	if (params.count("sig_interp"))    m_sigInterp    = std::stof(params["sig_interp"]);
	if (params.count("binning_depth")) m_binningDepth = std::stof(params["binning_depth"]);
	if (params.count("yc_min"))        m_ycMin        = std::stof(params["yc_min"]);
	if (params.count("yc_max"))        m_ycMax        = std::stof(params["yc_max"]);
	if (params.count("dshift_last"))   m_dshiftLast   = std::stof(params["dshift_last"]);
}

void OnlineSpikesV2::loadPreclusterShapes(std::string filepath)
{
	std::cout << "Loading precluster shapes from file " << filepath << std::endl;
	cnpy::NpyArray npWall = cnpy::npy_load(filepath);
	unclu_T = npWall.shape[0];
	K = npWall.shape[1];

	std::cout << "Detected " << unclu_T << " templates prior to clustering and " << K << " principal components." << std::endl;
}

// Allocate memory for and initialize memory that will only need to be written to once
void OnlineSpikesV2::initializeStaticMemory(InputParameters params)
{
	static const char *ptLabel = { "OnlineSpikesV2::initializeStaticMemory" };

	loadChannelMap(params.sInputFolder + "channelMap.npy");
	loadTemplates(params.sInputFolder + "templates.npy");
	loadWhitening(params.sInputFolder + "whiteningMat.npy");
	loadTemplateMap(params.sInputFolder + "templateMap.npy");
	loadKilosortTrainingData(params.sInputFolder);
	loadKilosortClusteringData(params.sInputFolder);
}

/*
	Parses the templates read from Kilosort into memory formats that allow for
	vectorized computations later on
*/
void OnlineSpikesV2::loadTemplates(std::string filepath)
{
	static const char *ptLabel = { "OnlineSpikesV2::loadTemplates" };

	// Allocate memory for template copies onto pinned memory
	cnpy::NpyArray npTemplates = getTemplates(filepath);
	float *flattenedTemplates = npTemplates.data<float>();

	T = npTemplates.shape[0];
	M = npTemplates.shape[1];
	C = (npTemplates.shape.size() == 3) ? npTemplates.shape[2] : 1;
	std::cout << "Detected " << T << " templates, each on " << M << " samples and " << C << " channels." << std::endl;

	/* Copy the templates to flattened arrays of differing orderedness to vectorize future GPU operations */
	for (long sampleInd = 0; sampleInd < M; sampleInd++) {          //m_lM = #Samples/template
		for (long templateInd = 0; templateInd < T; templateInd++) { //m_lT = #Templates
			for (long chanInd = 0; chanInd < C; chanInd++) { //m_lC = #Channels
				float entry = flattenedTemplates[chanInd + sampleInd * C + templateInd * C * M];
				D_chan_samp_temp[chanInd + sampleInd * C + templateInd * C * M] = entry;
			}
		}
	}
}

void OnlineSpikesV2::loadWhitening(std::string filepath)
{
	static const char *ptLabel = { "OnlineSpikesV2::loadWhitening" };

	cnpy::NpyArray npWhiteningMat = getWhitening(filepath);
	if (npWhiteningMat.shape[0] != C || npWhiteningMat.shape[1] != C)
		_RUN_ERROR(ptLabel, "getWhitening: Incorrect whitening size");

	// Copy to device
	_CUDA_CALL(cudaMemcpyAsync(d_whitening, npWhiteningMat.data<float>(), C * C * sizeof(float), cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaDeviceSynchronize());

	std::cout << "The following data was loaded successfully: Whitening matrix " << C << " x " << C << std::endl;
}

void OnlineSpikesV2::loadChannelMap(std::string filepath)
{
	static const char *ptLabel = { "OnlineSpikesV2::loadChannelMap" };

	cnpy::NpyArray npChannelMap = getChannelMap(filepath);

	// Check if size is correct
	if (npChannelMap.shape[0] != C)
		_RUN_ERROR(ptLabel, "loadChannelMap: Wrong size, size should be: " + std::to_string(C) + ", but is: " + std::to_string(npChannelMap.shape[0]));

	memcpy(channelMap.data(), npChannelMap.data<int>(), C * sizeof(int));
}

void OnlineSpikesV2::loadTemplateMap(std::string filepath)
{
	static const char *ptLabel = { "OnlineSpikesV2::loadTemplateMap" };

	cnpy::NpyArray npTemplateMap = getTemplateMap(filepath);
	if (npTemplateMap.shape[0] != T)
		_RUN_ERROR(ptLabel, "templateMapFile: Incorrect template map size");

	memcpy(templateMap.data(), npTemplateMap.data<int>(), T * sizeof(int));
}


void OnlineSpikesV2::loadKilosortTrainingData(std::string directoryPath)
{
    static const char* ptLabel =
        "OnlineSpikesV2::loadKilosortIntermediateTensors";

    std::cout << "Loading Kilosort tensors from " << directoryPath << std::endl;

    // --- Load .npy files ---
    auto npctc                   = cnpy::npy_load(directoryPath + "ctc.npy");
    auto npWall3                 = cnpy::npy_load(directoryPath + "Wall3.npy");
    auto npDriftMatrix           = cnpy::npy_load(directoryPath + "drift_matrix.npy");
    auto npiCC                   = cnpy::npy_load(directoryPath + "iCC.npy");
    auto npiU                    = cnpy::npy_load(directoryPath + "iU.npy");
    auto npUcc                   = cnpy::npy_load(directoryPath + "Ucc.npy");
    auto npwPCA                  = cnpy::npy_load(directoryPath + "wPCA.npy");
    auto npwPCA_permuted         = cnpy::npy_load(directoryPath + "wPCA_permuted.npy");
    auto nptemplateWaveforms     = cnpy::npy_load(directoryPath + "preclustered_template_waveforms.npy");
    auto npHpFilter              = cnpy::npy_load(directoryPath + "hp_filter.npy");
    auto npxc                    = cnpy::npy_load(directoryPath + "xc.npy");
    auto npyc                    = cnpy::npy_load(directoryPath + "yc.npy");
	auto npclusterCentroidsPca	 = cnpy::npy_load(directoryPath + "cluster_centroids_pca.npy");

    filterLen = npHpFilter.num_vals;

    std::cout << "Copying tensors to device..." << std::endl;

    // --- Copy to GPU ---
    _CUDA_CALL(cudaMemcpy(d_ctc,                   npctc.data<float>(),                 unclu_T * unclu_T * (2 * M + 1) * sizeof(float), cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_Wall3,                 npWall3.data<float>(),               unclu_T * K * C * sizeof(float),                 cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_driftMatrix,           npDriftMatrix.data<float>(),         C * C * sizeof(float),                           cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_iCC,                   npiCC.data<int64_t>(),               10 * C * sizeof(int64_t),                        cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_iU,                    npiU.data<int64_t>(),                unclu_T * sizeof(int64_t),                       cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_Ucc,                   npUcc.data<float>(),                 10 * K * unclu_T * sizeof(float),                cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_wPCA,                  npwPCA.data<float>(),                K * M * sizeof(float),                           cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_wPCA_permuted,         npwPCA_permuted.data<float>(),       K * M * sizeof(float),                           cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_templateWaveforms,     nptemplateWaveforms.data<float>(),   unclu_T * M * C * sizeof(float),                 cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_hpFilterFull,          npHpFilter.data<float>(),            filterLen * sizeof(float),                       cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_xc,                    npxc.data<float>(),                  C * sizeof(float),                               cudaMemcpyHostToDevice));
    _CUDA_CALL(cudaMemcpy(d_yc,                    npyc.data<float>(),                  C * sizeof(float),                               cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaMemcpy(d_clusterCentroidsPca,   npclusterCentroidsPca.data<float>(), T * K * sizeof(float),							 cudaMemcpyHostToDevice));

    _CUDA_CALL(cudaDeviceSynchronize());

    // --- Precompute norms (nm) ---
    for (int t = 0; t < unclu_T; ++t) {
		_CUBLAS_CALL(cublasSdot(
			cublasHandle,
			K * C,
			d_Wall3 + t * K * C, 1,
			d_Wall3 + t * K * C, 1,
			d_nm + t
		), "Error pre-computing template norms");
        _CUDA_CALL(cudaDeviceSynchronize());
    }

    // --- Precompute reciprocal of nm ---
	reciprocal(d_nm, unclu_T);
    _CUDA_CALL(cudaDeviceSynchronize());
}

// TODO: Get rid of function since we no longer use clusterMap
void OnlineSpikesV2::loadKilosortClusteringData(std::string directoryPath)
{
	static const char *ptLabel = { "OnlineSpikesV2::loadKilosortClusteringData" };

	// --- Load data from Kilosort
	cnpy::NpyArray npSpikeTemplates				= cnpy::npy_load(directoryPath + "spike_templates.npy");
	cnpy::NpyArray npSpikeDetectionTemplates	= cnpy::npy_load(directoryPath + "spike_detection_templates.npy");
	cnpy::NpyArray npClusterCenters				= cnpy::npy_load(directoryPath + "cluster_centroids.npy");

	long numSpikes						= npSpikeTemplates.shape[0];
	long* spikeTemplates				= npSpikeTemplates.data<long>();
	long* spikeDetectionTemplates		= npSpikeDetectionTemplates.data<long>();
	float* clusterCenters				= npClusterCenters.data<float>();
	
	// cluster centers are provided as [ x_1, y_1, x_2, y_2, ... ]
	for (int i = 0; i < npClusterCenters.num_vals; i++) {
		if (i & 1) ys.push_back(clusterCenters[i]);
		else       xs.push_back(clusterCenters[i]);
	}
}

// Sign applied to the measured residual shift
static const float DRIFT_SIGN = 1.0f;

// Load iKxx and the reference activity fingerprint required for estimation
void OnlineSpikesV2::loadDriftData(std::string directoryPath)
{
	static const char* ptLabel = { "OnlineSpikesV2::loadDriftData" };
	try {
		auto npiKxx = cnpy::npy_load(directoryPath + "iKxx.npy");
		if (npiKxx.shape.size() != 2 || (long)npiKxx.shape[0] != C || (long)npiKxx.shape[1] != C) {
			std::cout << "[Drift] iKxx.npy has unexpected shape; disabling drift estimation." << std::endl;
			m_driftEnabled = false;
			return;
		}
		_CUDA_CALL(cudaMemcpy(d_iKxx, npiKxx.data<float>(), (size_t)C * C * sizeof(float), cudaMemcpyHostToDevice));

		auto npRef = cnpy::npy_load(directoryPath + "reference_fingerprint.npy");
		if (npRef.shape.size() != 2) {
			std::cout << "[Drift] reference_fingerprint.npy has unexpected shape; disabling drift estimation." << std::endl;
			m_driftEnabled = false;
			return;
		}
		m_fpDmax   = (long)npRef.shape[0];
		m_nAmpBins = (int)npRef.shape[1];
		m_refFingerprint.assign(npRef.data<float>(), npRef.data<float>() + (m_fpDmax * m_nAmpBins));
		_CUDA_CALL(cudaDeviceSynchronize());
	}
	catch (const std::exception& e) {
		std::cout << "[Drift] Could not load drift tensors (" << e.what()
		          << "); disabling drift estimation." << std::endl;
		m_driftEnabled = false;
	}
}

void OnlineSpikesV2::accumulateFingerprint(long numSpikesInBatch)
{
	for (long i = 0; i < numSpikesInBatch; ++i) {
		m_fpDepths.push_back(closest_y[i]);
		m_fpAmps.push_back(spikeAmplitudes[i]);
	}
}

// Build a depth x amplitude fingerprint from a window's spikes
void OnlineSpikesV2::buildFingerprint(const std::vector<float>& depths,
                                      const std::vector<float>& amps,
                                      std::vector<float>& F)
{
	std::fill(F.begin(), F.end(), 0.0f);
	const long N = (long)depths.size();
	if (N == 0) return;

	const float dmin = m_ycMin - 1.0f;

	// amplitude -> percentile rank in [0, 1)
	std::vector<long> order(N);
	for (long i = 0; i < N; ++i) order[i] = i;
	std::sort(order.begin(), order.end(),
	          [&](long a, long b) { return amps[a] < amps[b]; });
	std::vector<float> rank(N);
	for (long r = 0; r < N; ++r) rank[order[r]] = (float)r / (float)N;

	for (long i = 0; i < N; ++i) {
		long row = (long)floorf((depths[i] - dmin) / m_binningDepth);
		if (row < 0) row = 0;
		if (row >= m_fpDmax) row = m_fpDmax - 1;
		long col = (long)floorf(rank[i] * m_nAmpBins);
		if (col < 0) col = 0;
		if (col >= m_nAmpBins) col = m_nAmpBins - 1;
		F[row * m_nAmpBins + col] += 1.0f;
	}
	for (size_t k = 0; k < F.size(); ++k) F[k] = log2f(1.0f + F[k]);

	// mean-subtract along depth, per amplitude column
	for (int c = 0; c < m_nAmpBins; ++c) {
		double mean = 0.0;
		for (long r = 0; r < m_fpDmax; ++r) mean += F[r * m_nAmpBins + c];
		mean /= (double)m_fpDmax;
		for (long r = 0; r < m_fpDmax; ++r) F[r * m_nAmpBins + c] -= (float)mean;
	}
}

// Cross-correlate window fingerprint against the reference over integer depth-bin shifts
// Returns the residual shift in MICRONS
float OnlineSpikesV2::registerFingerprint(const std::vector<float>& F)
{
	int n = (int)ceilf(m_driftMaxShiftUm / m_binningDepth);
	if (n > 15) n = 15;
	if (n < 1)  n = 1;

	const int dmax = (int)m_fpDmax;
	const int A = m_nAmpBins;

	std::vector<float> dc(2 * n + 1, 0.0f);
	for (int t = -n; t <= n; ++t) {
		double s = 0.0;
		for (int r = 0; r < dmax; ++r) {
			int rr = ((r - t) % dmax + dmax) % dmax; // torch.roll(F, t): F_rolled[r] = F[r - t]
			for (int c = 0; c < A; ++c)
				s += (double)m_refFingerprint[r * A + c] * (double)F[rr * A + c];
		}
		dc[t + n] = (float)s;
	}

	// Sub-bin peak via parabolic interpolation around the integer argmax.
	//
	// The previous x10 Gaussian-weighted upsampling applied a kernel of sigma
	// ~= 1 full bin over a +/-n-bin window (here n=2). That kernel is so wide
	// relative to the search window that it just smooths the correlation curve
	// toward its center, dragging every estimate toward zero -- a confirmed
	// ~3x under-gain: a known rigid shift of +/-10 um was recovered as only
	// +/-3 um. The integer-bin correlation `dc` itself recovers the shift
	// exactly, so we keep it and interpolate the peak with a 3-point parabola,
	// which restores ~unit gain on a rigid shift.
	int iBest = 0;
	float dcBest = dc[0];
	for (int k = 1; k < 2 * n + 1; ++k)
		if (dc[k] > dcBest) { dcBest = dc[k]; iBest = k; }

	float offsetBins = 0.0f;
	if (iBest > 0 && iBest < 2 * n) {
		float ym1 = dc[iBest - 1], y0 = dc[iBest], yp1 = dc[iBest + 1];
		float denom = ym1 - 2.0f * y0 + yp1; // < 0 at a genuine peak (concave down)
		if (fabsf(denom) > 1e-12f) {
			offsetBins = 0.5f * (ym1 - yp1) / denom;
			if (offsetBins >  1.0f) offsetBins =  1.0f;
			if (offsetBins < -1.0f) offsetBins = -1.0f;
		}
	}
	float bestBins = (float)(iBest - n) + offsetBins;

	float shiftUm = bestBins * m_binningDepth;
	if (shiftUm >  m_driftMaxShiftUm) shiftUm =  m_driftMaxShiftUm;
	if (shiftUm < -m_driftMaxShiftUm) shiftUm = -m_driftMaxShiftUm;
	return shiftUm;
}

// Worker-thread body: waits for a window snapshot, estimates drift, rebuilds the correction matrix
void OnlineSpikesV2::driftWorkerLoop()
{
	static const char* ptLabel = { "OnlineSpikesV2::driftWorkerLoop" };

	_CUDA_CALL(cudaSetDevice(m_deviceIndex));
	_CUDA_CALL(cudaStreamCreate(&m_driftStream));
	if (cublasCreate(&m_driftCublas) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "Failed to create drift cuBLAS handle");
	cublasSetStream(m_driftCublas, m_driftStream);

	{
		std::vector<float> host_static((size_t)C * C), host_test((size_t)C * C);
		_CUDA_CALL(cudaMemcpy(host_static.data(), d_driftMatrix, (size_t)C * C * sizeof(float), cudaMemcpyDeviceToHost));

		auto maxAbsDiff = [&](bool doTranspose) -> double {
			ComputeDriftMat(m_driftCublas, m_driftStream, d_xc, d_yc, d_iKxx, d_Kyx,
			                m_sigInterp, m_dshiftLast, C, d_driftMatrixB, doTranspose);
			_CUDA_CALL(cudaStreamSynchronize(m_driftStream));
			_CUDA_CALL(cudaMemcpy(host_test.data(), d_driftMatrixB, (size_t)C * C * sizeof(float), cudaMemcpyDeviceToHost));
			double m = 0.0;
			for (size_t i = 0; i < host_test.size(); ++i)
				m = std::max(m, (double)fabs(host_test[i] - host_static[i]));
			return m;
		};

		double diffNo  = maxAbsDiff(false);
		double diffYes = maxAbsDiff(true);
		m_driftTranspose = (diffYes < diffNo);
		std::cout << "[Drift] Matrix orientation gate: maxdiff(M)=" << diffNo
		          << " maxdiff(M^T)=" << diffYes << " -> using "
		          << (m_driftTranspose ? "M^T" : "M") << std::endl;
		if (std::min(diffNo, diffYes) > 1e-2)
			std::cout << "[Drift] WARNING: neither orientation matches drift_matrix.npy well; "
			             "check sig_interp/iKxx/dshift_last exports." << std::endl;
	}

	std::vector<float> depths, amps;
	long endCt = 0;
	for (;;) {
		{
			std::unique_lock<std::mutex> lk(m_driftMutex);
			m_driftCV.wait(lk, [&] { return m_driftJobReady || m_driftStop; });
			if (m_driftStop) break;
			depths = std::move(m_snapDepths);
			amps   = std::move(m_snapAmps);
			endCt  = m_snapWindowEndCt;
			m_snapDepths.clear();
			m_snapAmps.clear();
			m_driftJobReady = false;
		}
		estimateDrift(depths, amps, endCt);
	}

	cublasDestroy(m_driftCublas);
	cudaStreamDestroy(m_driftStream);
}

// Register a window, integrate the estimate, rebuild + publish the drift matrix.
void OnlineSpikesV2::estimateDrift(const std::vector<float>& depths,
                                   const std::vector<float>& amps, long windowEndCt)
{
	static const char *ptLabel = { "OnlineSpikesV2::estimateDrift" };
	const long minSpikes = 300;
	static long s_diagWindowIdx = 0; // TODO: can remove, diagnostic for drift investigation
	if ((long)depths.size() < minSpikes) {
		// Too few spikes to trust: keep the current matrix and estimate.
		std::cout << "[Drift][diag] window " << s_diagWindowIdx++
				  << " SKIPPED (nSpikes=" << depths.size()
				  << " < minSpikes=" << minSpikes << ") endCt=" << windowEndCt << std::endl;
		return;
	}

	std::vector<float> F((size_t)m_fpDmax * m_nAmpBins, 0.0f);
	buildFingerprint(depths, amps, F);

	float residualUm = registerFingerprint(F);

	float totalBeforeUm = m_totalShiftUm; // TEMP DIAGNOSTIC
	m_totalShiftUm += DRIFT_SIGN * residualUm;
	float rel = m_totalShiftUm - m_dshiftLast;
	bool clamped = false; // TEMP DIAGNOSTIC
	if (rel >  m_driftMaxShiftUm) { rel =  m_driftMaxShiftUm; m_totalShiftUm = m_dshiftLast + rel; clamped = true; }
	if (rel < -m_driftMaxShiftUm) { rel = -m_driftMaxShiftUm; m_totalShiftUm = m_dshiftLast + rel; clamped = true; }

	// TEMP DIAGNOSTIC: raw per-window residual vs. accumulated total - remove after drift discrepancy investigation
	std::cout << "[Drift][diag] window " << s_diagWindowIdx++
	          << " nSpikes=" << depths.size()
	          << " residualUm=" << residualUm
	          << " totalShiftUm=" << totalBeforeUm << "->" << m_totalShiftUm
	          << " rel=" << rel
	          << (clamped ? " [CLAMPED]" : "")
	          << " endCt=" << windowEndCt << std::endl;

	// Build the new matrix into the inactive buffer, then publish via pointer swap.
	float* active = m_activeDriftMatrix.load(std::memory_order_acquire);
	float* target = (active == d_driftMatrix) ? d_driftMatrixB : d_driftMatrix;
	ComputeDriftMat(m_driftCublas, m_driftStream, d_xc, d_yc, d_iKxx, d_Kyx,
	                m_sigInterp, m_totalShiftUm, C, target, m_driftTranspose);
	_CUDA_CALL(cudaStreamSynchronize(m_driftStream));

	m_activeDriftMatrix.store(target, std::memory_order_release);
	m_estDriftUm.store(rel, std::memory_order_relaxed);
	m_atomicDriftUpdateCt.store(windowEndCt, std::memory_order_relaxed);
}

void OnlineSpikesV2::runSpikeSorting()
{
	static const char *ptLabel = { "OnlineSpikesV2::runSpikeSorting" };

	long 	processedCt, // Most recent stream sample count that has been processed
			allowedCt, // Samples we are behind
			skipCounter = 0, // Number of times we skipped
			currBatchNumSamples; // Number of samples in current batch

	float	p2p; // peak-to-peak data to be sent to decoder
	std::vector<float> p2ps(C, 0); // per-channel peak-to-peak data to be sent to decoder

	// whether we performed any skipping during the current batch
	bool skip = false;

	// timespec's to keep track of processing time to be sent to the Decoder
	struct timespec batchBefore, batchAfter;

	// Vectors to store the spike times, templates, and amplitudes to be sent to the Decoder
	std::vector<long> times;
	std::vector<long> templates;
	std::vector<float> amplitudes;
	memset(lastSpikeTime.data(), 0, sizeof(long) * T);

	// Parameters specific to each individual OSS during parallelization
	OSSSpecificParams osParams = {
		C,
		channelMap,
		substream
	};

	// Get the latest sample count from Spike GLX
	latestCt = sglxSock->getStreamSampleCt(IMEC, osParams);
	processedCt = latestCt;
	m_windowStartCt = latestCt; // start of the first drift-estimation window

	// Main spike sorting loop
	while (true) {
		// Wait until the minimum time window has passed before processing
		sglxSock->waitUntil(latestCt + minWindow, osParams);
		
		// Time for tracking processing time in OutputGUI
		clock_gettime(batchBefore);

		// Update latest count to determine window size
		latestCt = sglxSock->getStreamSampleCt(IMEC, osParams);
		allowedCt = latestCt - timeBehind * samplingRate / 1'000;

		if ((latestCt - processedCt) <= maxWindow) {
			latestCt = sglxSock->fetchLatest(fetchBuf, osParams, processedCt);

			/* timing issues, such as if command line takes over main thread of execution
			   causes lLatestCt to be very large, making lLatestCt - lProcessedCt exceed
			   m_lMaxWindow despite the conditional, thus the need for the min */
			currBatchNumSamples = min((latestCt - processedCt) * (latestCt - processedCt >= 0), maxWindow) + minWindow;
			skip = false;
		} 

		// We're behind, but being behind is not tolerated (m_lTimeBehind == 0) so skip enough data to fetch most recent m_lMaxWindow batch
		else if (timeBehind == 0) {
			latestCt = sglxSock->fetchLatest(fetchBuf + minWindow * C, osParams, processedCt);
			currBatchNumSamples = maxWindow;
			skip = true;
		}
		
		// We are behind, but we think we can catch up, thus take from the place we are at now	
		else if (processedCt >= allowedCt) { // We are too behind, 2 options, small or big skip. 
			latestCt = sglxSock->fetchFromPlace(fetchBuf + minWindow * C, osParams, processedCt);
			currBatchNumSamples = maxWindow + minWindow;
			skip = false;
		} 

		// Smallskip: we fetch from the sample count we are allowed to be
		else if (smallSkip) {
			latestCt = sglxSock->fetchFromPlace(fetchBuf + minWindow * C, osParams, allowedCt);
			currBatchNumSamples = maxWindow;
			skip = true;
		} 
		
		// Bigskip: we fetch the most recent m_lMaxWindow batch.
		else {  // m_bSmallskip == false 
			latestCt = sglxSock->fetchLatest(fetchBuf + minWindow * C, osParams, processedCt);
			currBatchNumSamples = maxWindow;
			skip = true;
		}

		// Copy Data from cpu --> gpu
		{
			Timer timer("cpu to gpu");
			if (skip) {
				// Skip the last minScanWindow of previous batch (the first m_lMinWindow * m_lC bits)
				_CUDA_CALL(cudaMemcpy(d_fetchBuf, fetchBuf + minWindow * C, C * currBatchNumSamples * sizeof(float), cudaMemcpyHostToDevice));

				// Increment skip counter
				skipCounter++;
			}
			else {
				_CUDA_CALL(cudaMemcpy(d_fetchBuf, fetchBuf, C * currBatchNumSamples * sizeof(float), cudaMemcpyHostToDevice));
			}
			_CUDA_CALL(cudaDeviceSynchronize());
		}

		/* 
			The data received from SpikeGLX in d_fetchBuf is currently arranged such that SAMPLES are contiguous, i.e. 
			d_fetchBuf[t + chan * currBatchNumSamples] corresponds to sample t, channel chan
		*/

		// Calculate peak-to-peak for OutputGUI
		// TODO: Try and make it compute P2P per channel, send it to the GUI, write GUI to display per-channel P2P, and then 
		//			make sure the computations for P2P per channel is fast by writing custom kernel
		p2p = P2P_calc(d_fetchBuf, C * currBatchNumSamples);

		// Remove means
		_CUDA_CALL(cudaMemset(d_means, 0, C * sizeof(float)));
		{
			Timer timer("meanRemove()");
			meanRemove(d_fetchBuf, d_means, currBatchNumSamples, C);
		}
		_CUDA_CALL(cudaDeviceSynchronize());

		// Median removal
		{
			Timer timer("medianRemove()");
			medianRemove(d_fetchBuf, C, currBatchNumSamples);
		}
		_CUDA_CALL(cudaDeviceSynchronize());

		// Perform a high-pass filter at 300 hz assuming the signal is at 30000 hz
		{
			Timer timer("highpassFilter()");
			transpose(d_fetchBuf, d_fetchBuf2, currBatchNumSamples, C);
			highpassFilter(d_fetchBuf2, C, currBatchNumSamples, 30000, 300);
		}
		_CUDA_CALL(cudaDeviceSynchronize());

		// Whiten the batch on device (THIS WORKS FOR SURE, DO NOT TOUCH OR WORRY ABOUT IT)
		{
			Timer timer("whitening()");
			matMul(cublasHandle, d_whitening, d_fetchBuf2, d_fetchBuf, C, C, currBatchNumSamples);
		}
		_CUDA_CALL(cudaDeviceSynchronize());

		// Drift correct. The matrix pointer is hot-swapped by the drift worker
		// (double-buffered); read it once per batch via an acquire load.
		{
			Timer timer("driftCorrection()");
			float* driftMat = m_activeDriftMatrix.load(std::memory_order_acquire);
			matMul(cublasHandle, driftMat, d_fetchBuf, d_fetchBuf2, C, C, currBatchNumSamples);
		}
		_CUDA_CALL(cudaDeviceSynchronize());
		
		// Perform OMP
		numSpikes = kilosortMatchingPursuit(d_fetchBuf2, currBatchNumSamples);

		// Use results of OMP to assign unmapped spike templates to the closest clusters
		// - inputs: d_spikeTemplates, d_spikeTimes, d_residual
		// - outputs: closest_x, closest_y
		{
			Timer timer("closestCluster()");
			computeClosestClusters(currBatchNumSamples, numSpikes);
		}

		// Save the spikes into times, templates, amplitudes
		saveSpikes(numSpikes, latestCt - currBatchNumSamples + 1, currBatchNumSamples - minWindow, times, templates, amplitudes);

		// Drift estimation: accumulate this batch's spikes into the current
		// window and, when the window is full, hand a snapshot to the worker.
		if (m_driftEnabled) {
			accumulateFingerprint(numSpikes);
			if (latestCt - m_windowStartCt >= m_estWindowSamples) {
				{
					std::lock_guard<std::mutex> lk(m_driftMutex);
					if (!m_driftJobReady) {          // worker idle: hand off
						m_snapDepths.swap(m_fpDepths);
						m_snapAmps.swap(m_fpAmps);
						m_snapWindowEndCt = latestCt;
						m_driftJobReady = true;
						m_driftCV.notify_one();
					}
				}
				m_fpDepths.clear();
				m_fpAmps.clear();
				m_windowStartCt = latestCt;
			}
		}

		clock_gettime(batchAfter);
		long processTime = GetTimeDiff(batchAfter, batchBefore);

		// Send relevant data to decoder
		OnlineSpikesPayload payload = { recordingOffset,
								latestCt,
								times,
								templates,
								amplitudes,
								rootMeanSquared,
								p2p,
								processTime
		};
		payload.driftShiftUm  = m_estDriftUm.load(std::memory_order_relaxed);
		payload.driftUpdateCt = m_atomicDriftUpdateCt.load(std::memory_order_relaxed);

		sendPayload(&imecFm, payload, decoderImecAddr);
		//duplicate check in save spikes
		// Debug
		writeSpikesToFile(times, templates, amplitudes);

		// Update stream sample count
		processedCt = latestCt - lookback; //positive lookback means we are overlapping our batches to ensure we capture all of the data with the zeroing

		times.clear();
		templates.clear();
		amplitudes.clear();
	}
}


void OnlineSpikesV2::computeClosestClusters(long currBatchNumSamples, long numSpikes)
{
	static const char* ptLabel = { "OnlineSpikesV2::computeClosestClusters" };
	static const dim3 blockDim(16, 16);

	_CUDA_CALL(cudaMemcpy(d_spikeTemplates, spikeTemplates, numSpikes * sizeof(long), cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaMemcpy(d_spikeTimes, spikeTimes, numSpikes * sizeof(long), cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaDeviceSynchronize());
	
	dim3 gridDim((numNearestChans + blockDim.x - 1) / blockDim.x,
		(numSpikes + blockDim.y - 1) / blockDim.y);

	compute_xfeat_kernel <<<gridDim, blockDim >>> (
		numSpikes, numNearestChans, M, K, unclu_T, C,
		d_iCC, d_iU, d_Ucc, d_residual, currBatchNumSamples,
		d_wPCA_permuted, d_spikeTimes, d_spikeTemplates, d_amps,
		d_xfeat
	);

	int blocksPerGrid = (numSpikes * K * numNearestChans + DEFAULT_TPB - 1) / DEFAULT_TPB;
	transpose_xfeat <<<blocksPerGrid, DEFAULT_TPB >>> (d_xfeat, d_tF, numNearestChans, numSpikes, K);
	_CUDA_CALL(cudaDeviceSynchronize());





	blocksPerGrid = (numSpikes + DEFAULT_TPB - 1) / DEFAULT_TPB;
	compute_spike_positions_kernel <<<blocksPerGrid, DEFAULT_TPB >>> (
		numSpikes,
		numNearestChans,
		K,
		d_tF,          // shape: [numSpikes, numNearestChans, K]
		d_spikeTemplates, // shape: [numSpikes]
		d_iU,             // shape: [numTemplates]
		d_iCC,            // shape: [numNearestChans, numTemplates] in row-major order
		d_xc,            // channel x positions, length: numChannels
		d_yc,            // channel y positions, length: numChannels
		d_xs,                  // output: spike x positions [numSpikes]
		d_ys,                  // output: spike y positions [numSpikes]
		unclu_T,
		C
	);

	_CUDA_CALL(cudaMemcpy(closest_x, d_xs, numSpikes * sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpy(closest_y, d_ys, numSpikes * sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaDeviceSynchronize());
}

// Expects d_batch to be arranged such that samples are contiguous (this is not what SpikeGLX provides)
long OnlineSpikesV2::kilosortMatchingPursuit(float* d_batch, long currBatchNumSamples)
{
	static const char* ptLabel = { "OnlineSpikesV2::kilosortMatchingPursuit" };

	const float isSpikeThreshold = Th_learned * Th_learned;

	// --- Copy everything to residual, which will be updated per iteration of OMP
	_CUDA_CALL(cudaMemcpy(d_residual, d_batch, C * currBatchNumSamples * sizeof(float), cudaMemcpyDeviceToDevice));

	// --- Project batch into PCA space
	{
		Timer timer("batchToPCA");
		projectToPCA(d_batch, d_wPCA, d_batchPCA, K, M, C, currBatchNumSamples);
		_CUDA_CALL(cudaDeviceSynchronize());
	}

	// --- Compute convolution between batch and templates in PCA space
	{
		Timer timer("crossCorrelation()");
		crossCorrelation(d_Wall3, d_batchPCA, d_convResult, unclu_T, K, C, currBatchNumSamples);
		_CUDA_CALL(cudaDeviceSynchronize());
	}

	// --- Perform OMP on the batch with neuron templates as the OMP templates
	long numSpikes = 0;
	memset(spikeTemplates, 0, unclu_T * currBatchNumSamples * sizeof(long));
	memset(spikeTimes, 0, unclu_T * currBatchNumSamples * sizeof(long));
	memset(spikeAmplitudes, 0, unclu_T * currBatchNumSamples * sizeof(long));
	d_spikeIndices.clear();

	for (int j = 0; j < 50; j++) {
		// Normalize matrix into target units by applying ReLU, squaring values, and zeroing the border
		{
			Timer timer("normalizeConvolution");
			normalizeConv(d_convResult, d_nm, d_convNormalized, unclu_T, currBatchNumSamples, M);
			_CUDA_CALL(cudaDeviceSynchronize());
		}

		// Collapse channel dimension, i.e. result will be array of length = # samples.
		// The result will be such that result[s] = max_{c, t} (similarity between batch and template t at channel c at sample s}
		{
			Timer timer("collapseWithMax");
			reduceToTimeDimByMax(d_convNormalized, d_maxAtTime, d_imax, unclu_T, currBatchNumSamples);
			_CUDA_CALL(cudaDeviceSynchronize());
		}

		// Pool together the result of above to find local maxima (across time, of window size M)
		{
			Timer timer("Max pool");
			fwdMaxPool1d(d_maxAtTime, d_Cfmaxpool, currBatchNumSamples, M);
			_CUDA_CALL(cudaDeviceSynchronize());
		}

		// Compute the indices that are local maxima AND above a certain threshold
		{
			Timer timer("Finding matching indices");
			findMatchingIndices(d_Cfmaxpool, d_convNormalized, isSpikeThreshold, unclu_T, currBatchNumSamples, M, d_spikeIndices, d_count);
		}
	
		// No spikes detected this iteration, meaning no more spikes in rest of batch
		if (d_spikeIndices.size() == 0) break;

		// Add spikes to total spike indices, also, need to reformat index ordering
		for (long idx : d_spikeIndices) {
			int temp = idx / currBatchNumSamples;
			int sample = idx % currBatchNumSamples;

			// note this index ordering is different from the one used above, need to return wrt legacy index ordering
			spikeTemplates[numSpikes] = temp;
			spikeTimes[numSpikes] = sample;
			numSpikes++;
		}

		// Copy spike templates to device for later use in extracting ctc matrix
		_CUDA_CALL(cudaMemcpy(d_spikeTemplates, spikeTemplates, d_spikeIndices.size() * sizeof(long), cudaMemcpyHostToDevice));

		// Compute the amplitudes of each spike
		{
			Timer timer("Compute Amplitudes");
			_CUDA_CALL(cudaMemset(d_amps, 0, unclu_T * currBatchNumSamples * sizeof(float)));
			auto blocksPerGrid = (d_spikeIndices.size() + DEFAULT_TPB - 1) / DEFAULT_TPB;
			compute_amps_kernel <<<blocksPerGrid, DEFAULT_TPB>>> (d_convResult, d_nm, thrust::raw_pointer_cast(d_spikeIndices.data()), d_amps, d_spikeIndices.size(), unclu_T, currBatchNumSamples);
			_CUDA_CALL(cudaDeviceSynchronize());
		}
	
		_CUDA_CALL(cudaMemcpy(spikeAmplitudes + numSpikes - d_spikeIndices.size(), d_amps, d_spikeIndices.size() * sizeof(float), cudaMemcpyDeviceToHost));
		_CUDA_CALL(cudaDeviceSynchronize());

		// Update the residual by removing the contribution of the spikes from both the raw batch and the convolution result.
		// The only reason we must remove the signal from the raw batch is because it helps us compute the location of spikes later
		{
			Timer timer("Update Residual");
			updateResidual(d_spikeIndices, d_amps, d_templateWaveforms, M, C, currBatchNumSamples, d_ctc, unclu_T, d_residual, d_convResult);
			_CUDA_CALL(cudaDeviceSynchronize());
		}
	}

	return numSpikes;
}

// TODO: Cache cufftHandle's: since batch sizes are variable and since cufftHandles are dependent on the batch size,
//			we can avoid having to reallocate memory for cufft every single time we get a new batch by caching commonly used handles
//			(such as for max window size).
void OnlineSpikesV2::highpassFilter(float* d_batch, int C, int currBatchNumSamples, float sampling_freq, float frequency_low)
{
	static const char* ptLabel = { "OnlineSpikesV2::highpassFilter" };

	if (filterLen < currBatchNumSamples) {
		std::cout << "Please decrease window size to below " << filterLen << " samples" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Crop the filter to match the batch length
	const int cropLen = (filterLen - currBatchNumSamples) / 2;
	_CUDA_CALL(cudaMemcpy(d_hpFilterSub, d_hpFilterFull + cropLen, currBatchNumSamples * sizeof(float), cudaMemcpyDeviceToDevice));
	_CUDA_CALL(cudaDeviceSynchronize());

	// Grab plan for forward-FFT
	int dims[1] = { currBatchNumSamples };
	cufftHandle batchPlanForward;
	cufftPlanMany(&batchPlanForward, 1, dims, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, CUFFT_C2C, C);

	cufftHandle filterPlanForward;
	cufftPlanMany(&filterPlanForward, 1, dims, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, CUFFT_C2C, 1);

	// Lift from real to complex, d_hpworkspace will contain the frequency data at the end
	float_to_cufftComplex(d_batch, d_hpworkspace, C * currBatchNumSamples);
	float_to_cufftComplex(d_hpFilterSub, d_hpworkspace2, currBatchNumSamples);

	// Perform FFT for filter + batch
	cufftExecC2C(batchPlanForward,  d_hpworkspace,  d_batchFreq,    CUFFT_FORWARD);
	cufftExecC2C(filterPlanForward, d_hpworkspace2, d_hpFilterFreq, CUFFT_FORWARD);

	// Apply filter
	applyFilter(d_batchFreq, d_hpFilterFreq, currBatchNumSamples, C);

	// Perform inverse FFT
	cufftHandle batchPlanInverse;
	cufftPlanMany(&batchPlanInverse, 1, dims, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, CUFFT_C2C, C);
	cufftExecC2C(batchPlanInverse, d_batchFreq, d_hpworkspace, CUFFT_INVERSE);

	// Extract real part
	cufftComplex_to_float(d_hpworkspace, d_batch, C * currBatchNumSamples);

	// Normalize because cufft doesn't do that
	scale(d_batch, 1.0 / (static_cast<float>(currBatchNumSamples)), C * currBatchNumSamples);
	shift(d_batch, d_shifted, currBatchNumSamples, C);

	_CUDA_CALL(cudaMemcpy(d_batch, d_shifted, C * currBatchNumSamples * sizeof(float), cudaMemcpyDeviceToDevice));
	_CUDA_CALL(cudaDeviceSynchronize());
	cufftDestroy(batchPlanForward);
	cufftDestroy(batchPlanInverse);
	cufftDestroy(filterPlanForward);
}

int OnlineSpikesV2::closestCluster(const float x, const float y)
{
	static const char* ptLabel = { "OnlineSpikesV2::closestCluster" };

	float min = FLT_MAX;
	int minInd = -1;

	// Scan over the FINAL-cluster centroids, xs/ys come from cluster_centroids.npy
	for (int t = 0; t < (long)xs.size(); t++) {
		float dist = sqrt((x - xs[t]) * (x - xs[t]) + (y - ys[t]) * (y - ys[t]));
		if (dist < min) {
			min = dist;
			minInd = t;
		}
	}
	return minInd;
}

SorterParameters OnlineSpikesV2::getSorterParams() {
	SorterParameters params = { T,
								C,
								M,
								W,
								samplingRate,
								activeChannels,
								templateMap
	};
	return params;
}

void OnlineSpikesV2::findMaxAbs(float *input, long length, int *ind, float *val) {
	static const char *ptLabel = { "OnlineSpikesV2::findMaxAbs" };

	// Allocate temporary buffer for absolute values
	float *d_absBuf;
	_CUDA_CALL(cudaMalloc((void**)&d_absBuf, length * sizeof(float)));
	
	// Compute absolute values
	_NPP_CALL(nppsAbs_32f(input, d_absBuf, length), "Error in computing absolute values.");

	// Find max value and index
	_NPP_CALL(nppsMaxIndx_32f(d_absBuf, length, d_nppValBuf, d_nppIndBuf, d_nppMaxBuf), "Error in finding the maximum absolute value.");

	// Copy results back to host
	_CUDA_CALL(cudaMemcpy(val, d_nppValBuf, sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpy(ind, d_nppIndBuf, sizeof(int), cudaMemcpyDeviceToHost));
}


void OnlineSpikesV2::fwdMaxPool1d(float* d_matrix, float* d_result, int len, int width) {
	static const char *ptLabel = { "OnlineSpikesV2::fwdMaxPool" };
	int blocksPerGrid = (len + DEFAULT_TPB - 1) / DEFAULT_TPB;
	fwd_max_pool_1d_kernel << <DEFAULT_TPB, blocksPerGrid >> > (d_matrix, d_result, len, width);
}

void OnlineSpikesV2::findMax(float *input, long length, int *ind, float *val) {
	static const char *ptLabel = { "OnlineSpikesV2::findMax" };

	// Find max value and index
	_NPP_CALL(nppsMaxIndx_32f(input, length, d_nppValBuf, d_nppIndBuf, d_nppMaxBuf), "Error in finding the maximum value.");

	// Copy results back to host
	_CUDA_CALL(cudaMemcpy(val, d_nppValBuf, sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpyAsync(ind, d_nppIndBuf, sizeof(int), cudaMemcpyDeviceToHost));
}


void OnlineSpikesV2::findMin(float *input, long length, int *ind, float *val) {
	static const char *ptLabel = { "OnlineSpikesV2::findMin" };
	NppStatus err = nppsMinIndx_32f(input, length, d_nppValBuf, d_nppIndBuf, d_nppMinBuf);

	if (err != NPP_SUCCESS) _RUN_ERROR(ptLabel, "Error in finding the minimum value.");

	_CUDA_CALL(cudaMemcpyAsync(ind, d_nppValBuf, sizeof(int), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpy(val, d_nppIndBuf, sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaDeviceSynchronize());

}

float OnlineSpikesV2::P2P_calc(float *input, long length) {

	float sfMaxVal{}, sfMinVal{};
	int sfMaxInd{}, sfMinInd{};

	findMax(input, length, &sfMaxInd, &sfMaxVal);
	findMin(input, length, &sfMinInd, &sfMinVal);

	return sfMaxVal - sfMinVal;
}

void OnlineSpikesV2::writeSpikesToFile(std::vector<long> times, std::vector<long> templates, std::vector<float> amplitudes) {
	static const char *ptLabel = { "OnlineSpikesV2::writeSpikesToFile" };

	if (!spikesFileOut.is_open() || spikesFileOut.fail()) {
		std::cerr << "WARNING: Spike output file is not writable, attempting to reopen..." << std::endl;
		spikesFileOut.clear();
		spikesFileOut.open(ossOutputDir + "spikeOutput.txt", std::ios::app); // append to avoid clobbering current run's data
		if (!spikesFileOut.is_open()) {
			std::cerr << "ERROR: Could not reopen spike output file!" << std::endl;
			return;
		}
	}

	for (int i = 0; i < times.size() && i < templates.size(); i++) {
		spikesFileOut << recordingOffset + times[i] << "," << templates[i]
			<< "," << amplitudes[i] << "," << closest_y[i] << "\n";
	}
	spikesFileOut.flush();
}

template <class T>
typename std::vector<T>::iterator insert_sorted(std::vector<T> &vec, T const& item) {
	return vec.insert(std::upper_bound(vec.begin(), vec.end(), item), item);
}

void OnlineSpikesV2::saveSpikes(
	long numSpikes,
	long startSampleOffset, long endSampleOffset, 
	std::vector<long>& times, std::vector<long>& templates, std::vector<float>& amplitudes
) 
{
	static const char *ptLabel = { "OnlineSpikes::saveSpikes" }; _UNUSED(*ptLabel);
	long  sampleInd;
	long  templateInd;
	float amplitude;

	//Loop over found spikes
	for (long i = 0; i < numSpikes; i++) {
		sampleInd = spikeTimes[i] - M / 2 - M + nt0min; // just copying kilosort here
		sampleInd -= lookback;							// to account for our overlapping batches
		amplitude = spikeAmplitudes[i];	
		templateInd = closestCluster(closest_x[i], closest_y[i]);

		// if template is spiking too soon, skip
		bool isDuplicate = startSampleOffset + sampleInd - lastSpikeTime[templateInd] < dt;
		if (isDuplicate) continue;
		lastSpikeTime[templateInd] = startSampleOffset + sampleInd;

		// This means that if ln => lEndValid, skip inserting into vectors
		if (sampleInd >= endSampleOffset)
			continue;

		// Put time, template, and amplitudes into respective vectors, sorted by time
		auto itr = insert_sorted(times, startSampleOffset + sampleInd);
		size_t pos = itr - times.begin(); // Get position of where new time was inserted
		templates.insert(templates.begin() + pos, templateInd);
		amplitudes.insert(amplitudes.begin() + pos, amplitude);
	}
}