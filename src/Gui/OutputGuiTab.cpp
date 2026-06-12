#include <numeric>

#include <ImGUI/implot.h>
#include <ImGUI/implot_internal.h>
extern ImGuiID g_spikeStatsDockNode;
extern ImGuiID g_rasterDockNode;
#include "../Helpers/GuiHelpers.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/onlineSpikesPayload.h"
#include "../Networking/NetworkHelpers.h"

#include "OutputGuiTab.h"

#define BIN_SIZE 1500
static std::atomic<long long> maxSpikeTimesSize = 0;

OutputGuiTab::OutputGuiTab(std::string tabName) :
	m_lT(0),
	m_lC(0),
	m_lM(0),
	m_lN(0),
	m_fSampRate(0),
	m_dChanpos(),
	m_tCalcing(),
	predictLabel(-1),
	sock(Sock::UDP),
	streamSampleCt(-1),
	m_bNeurons(),
	plotWindowClass(ImGuiWindowClass()),
	fm(&sock),
	tabName(tabName)
{
	plotWindowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoDockingOverMe;
	std::thread fmThread = fm.assemblerThread();
	fmThread.detach();
};


OutputGuiTab::~OutputGuiTab()
{
	m_tCalcing.join();

	//Delete neurons from the heap
	for (int ii = 0; ii < m_lT; ii++) {
		delete m_bNeurons[ii];

	}

	delete[] m_bNeurons;
}


void OutputGuiTab::setupOutput(sockaddr_in mainAddr, long m_lMaxScanWind, long m_lSpikeRateWindow, bool isDecoding)
{
	this->isDecoding = isDecoding; 
	// Connect to the Main Server
	std::cout << tabName << ": Sending GUI connect message..." << std::endl;
	sendConnectMsg(&sock, mainAddr, _GUI);
	std::cout << tabName << ": Sent GUI connect message!" << std::endl;


#ifdef _DEBUG
	std::vector<double> m_dChanActivated;
	for (int i = 0; i < 100; i++) {
		m_dChanActivated.push_back(i);
	}
	SorterParameters params = { 100, 100, 100, 1500, 30000, m_dChanActivated };
#else
	// Receive SorterParams from the decoder
	std::cout << tabName << ": Waiting for connection message from decoder..." << std::endl;
	SorterParameters params = recvPayload<SorterParameters>(&fm);
	std::cout << tabName << ": Received connection message from decoder!" << std::endl;
#endif

	m_lT = params.m_lT; // TODO: potentially integrate SorterParameters struct further into GUIfuncs 
	m_lC = params.m_lC;
	m_lM = params.m_lM;
	m_lN = params.m_lN;
	m_fSampRate = params.m_fSampRate;
	m_dChanpos = params.m_dChanpos;
	m_vNeuronIndices = params.m_vNeuronIndices;

	int min = INT_MAX;
	for (auto &index : m_vNeuronIndices) if (index < min) min = index;
	m_iMinNeuronIndex = min;

	m_lAvgWindowTime = m_lSpikeRateWindow; // TODO rename these variables to something standard
	setMaxScanWindow(m_lMaxScanWind, params.m_fSampRate);

	// initialize the neurons on the heap
	m_bNeurons = new Neuron*[m_lT];
	for (int i = 0; i < m_lT; i++) {
		m_bNeurons[i] = new Neuron(m_vNeuronIndices[i]);
	}

	// Start calculations on seperate thread
	m_tCalcing = std::thread(&OutputGuiTab::UpdateEvents, this);

	// Set channel number
	for (int ii = 0; ii < m_lT; ii++)
		m_bNeurons[ii]->SetChanNum((int)m_dChanpos[ii]);
}

void OutputGuiTab::UpdateEvents() {
	OnlineSpikesPayload payload;

	static int binsize = BIN_SIZE;

	while (true) { // TODO implement some exit protocol
	#ifdef _DEBUG
		std::vector<long> vTimes;
		std::vector<long> vTemplates;
		std::vector<float> vAmplitudes;
		for (int i = 0; i < 50; i++) {
			vTimes.push_back(streamSampleCt + i);
			vTemplates.push_back(i / m_lC);
			vAmplitudes.push_back(4);
		}
		payload = {};
		payload.streamSampleCt = streamSampleCt + 1500;
		payload.Times = vTimes;
		payload.Templates = vTemplates;
		payload.Amplitudes = vAmplitudes;
	#else
		payload = recvPayload<OnlineSpikesPayload>(&fm);
	#endif

		{
			std::lock_guard<std::mutex> lk(trialMutex);
			predictLabel = payload.predictLabel;
			label = payload.label;
			nTrials = payload.nTrials;
			nCorrect = payload.nCorrect;
			confidence = payload.confidence;
		}

		streamSampleCt = payload.streamSampleCt;
		long eventTime = payload.eventStreamSampleCt;
		
		{
			std::lock_guard<std::mutex> lk(eventMutex);
			if (eventTime != -1 && (eventTimes.empty() || eventTimes.back() != eventTime)) {
				eventTimes.push_back(eventTime);
				eventLabels.push_back(label);
				
			}
		}

		{
			std::lock_guard<std::mutex> lk(vrmsMutex);
			m_vdVRMS.push_back(payload.VRMS);
		}
		{
			std::lock_guard<std::mutex> lk(p2pMutex);
			m_vfP2P.push_back(payload.P2P);
		}

		processingTimeMutex.lock();
		m_vProcessTimes.push_back(payload.processTime);
		processingTimeMutex.unlock();

		/*
		for (int ii = 0; ii < payload.Times.size(); ii++) {
			int neur = payload.Templates[ii];
			long time = payload.Times[ii];

			if (neur >= m_lT) continue;

			m_bNeurons[neur]->AddSpike(payload.Times[ii]);
			m_bNeurons[neur]->AddSpikeAmplitude(payload.Amplitudes[ii]);

			m_bNeurons[neur]->CalcSpikeRate(&streamSampleCt, m_lAvgWindowTime, m_fSampRate);
			m_bNeurons[neur]->AddSpikeRate();

			// For now hardcoded, maybe change later
			m_bNeurons[neur]->CheckAndAddBin(binsize, streamSampleCt);
		}*/

		for (int i = 0; i < payload.Times.size(); ++i) {
			int neur = payload.Templates[i];
			if (neur < m_lT) {
				m_bNeurons[neur]->AddSpike(payload.Times[i]);
				m_bNeurons[neur]->AddSpikeAmplitude(payload.Amplitudes[i]);
			}
		}

		for (int n = 0; n < m_lT; ++n) {
			m_bNeurons[n]->CalcSpikeRate(&streamSampleCt,
				m_lAvgWindowTime,
				m_fSampRate);
			m_bNeurons[n]->AddSpikeRate();
			m_bNeurons[n]->CheckAndAddBin(binsize,
				streamSampleCt);
		}
	}
}


void OutputGuiTab::Render(const ImVec2 windowCenter)
{
	//Draw the ImGUI components
	DrawImGUI(windowCenter);

	//This updates and draws the circles
	for (int ii = 0; ii < m_lT; ii++) {
		m_bNeurons[ii]->Update(this);
	}
}


void OutputGuiTab::DrawImGUI(const ImVec2 windowCenter) {
	// Gui state
	static bool showRaster = true;
	static bool showNeuronInfo = true;
	static bool showProcessTimes = true;
	static bool showVRMS = false;
	static bool showP2P = false;
	static bool showTrialInfo = true;

	ImGui::Begin("Plot Menu");
	ImGui::Checkbox("Spike Raster", &showRaster);
	ImGui::Checkbox("Neuron Info", &showNeuronInfo);
	ImGui::Checkbox("Processing Times", &showProcessTimes);
	//ImGui::Checkbox("VRMS", &showVRMS);	
	ImGui::Checkbox("P2P", &showP2P);
	//ImGui::Checkbox("Trial Info", &showTrialInfo);
	ImGui::End();

	std::string DisplayText = "Amount of time data was skipped: ";
	//ImGui::Text(DisplayText.append(std::to_string(Sorter->GetSkipCounter())).c_str());  //TODO This needs some fixing

	if (showRaster)
		plotRaster(windowCenter, showRaster);

	if (showNeuronInfo)
		displayNeuronInfo(windowCenter, showNeuronInfo);

	if (showProcessTimes)
		plotProcessTimes(windowCenter, showProcessTimes);

	//if (showVRMS)
	//	plotVRMS(windowCenter, showVRMS);

	if (showP2P)
		plotP2P(windowCenter, showP2P);

	//if (showTrialInfo)
	//	displayTrialInfo(windowCenter, showTrialInfo);
}

static void RasterTimeAxisFormatter(double value, char* buff, int size, void* /*user_data*/) {
	if (value < 0.0) value = 0.0;
	long totalSec = (long)(value + 0.5);
	long h = totalSec / 3600;
	long m = (totalSec % 3600) / 60;
	long s = totalSec % 60;
	if (h > 0)
		snprintf(buff, size, "%ld:%02ld:%02ld", h, m, s);
	else
		snprintf(buff, size, "%ld:%02ld", m, s);
}


void OutputGuiTab::plotRaster(const ImVec2 windowCenter, bool &showRaster) {
	if (g_rasterDockNode != 0)
		ImGui::SetNextWindowDockID(g_rasterDockNode, ImGuiCond_Always);

	ImGui::SetNextWindowClass(&plotWindowClass);

	// Check Begin() return value: returns false when the window is collapsed.
	// All ImPlot calls must be skipped in that case — calling EndPlot() without
	// a successful BeginPlot() corrupts ImPlot's context and triggers a GPU TDR crash.
	if (!ImGui::Begin("Raster Plot", &showRaster)) {
		ImGui::End();
		return;
	}

	static int	  history = 5; // Seconds
	ImGui::InputInt("History (s):", &history, 1, 10);
	history = max(1, history); // TODO clampedInput


	
	ImGui::SameLine();
	ImGui::Checkbox("Fit to active neurons", &m_bFittoActive);

	double yLo = m_iMinNeuronIndex - 0.5;
	double yHi = m_lT - 0.5;
	if (m_bFittoActive) {
		int lo = INT_MAX, hi = INT_MIN;
		for (int i = 0; i < m_lT; i++) {
			std::lock_guard<std::mutex> lock(m_bNeurons[i]->spikeTimeMutex);
			if (m_bNeurons[i]->m_vSpikeTime.empty())
			continue;
			if (i < lo) lo = i;
			if (i > hi) hi = i;
		}
		if (lo <= hi) { // check if at least one neuron has fired
			yLo = lo - 0.5;
			yHi = hi + 0.5;
		}
	}



	static std::vector<double> yVal;
	static std::vector<double> xSec;
	int size;

	// Compute the history window size (in streamSampleCounts)
	int window = m_fSampRate * history; // history window

	// startingStreamSampleCt is the left most stream sample count that will be plotted in the raster
	long startingStreamSampleCt = streamSampleCt - window;

	if (streamSampleCt < window) { // Start of run edge case (No scrolling because looking at a time window < window)
		ImPlot::SetNextAxisToFit(ImAxis_X1);
		ImPlot::SetNextAxisLimits(ImAxis_Y1, yLo, yHi, ImPlotCond_Always);

		if (ImPlot::BeginPlot("Raster plot:", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoTitle)) {
			ImPlot::SetupAxes("Time (hh:mm:ss)", "Neuron");
			ImPlot::SetupAxisFormat(ImAxis_X1, RasterTimeAxisFormatter);
			
			const double secPerSample = 1.0 / (double)m_fSampRate;
			for (int i = 0; i < m_lT; i++) {
				std::lock_guard<std::mutex> lock(m_bNeurons[i]->spikeTimeMutex);
				if (m_bNeurons[i]->m_vSpikeTime.empty())
					continue;

				size = m_bNeurons[i]->m_vSpikeTime.size();
				xSec.resize(size);
				for (int k = 0; k < size; k++) {
					xSec[k] = m_bNeurons[i]->m_vSpikeTime[k] * secPerSample;
				}
				yVal.assign(size, (double)m_bNeurons[i]->m_inumber);
				ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0.8F, ImVec4(0.26f, 0.59f, 0.98f, 0.78f), IMPLOT_AUTO, ImVec4(0.26f, 0.59f, 0.98f, 0.78f));
				ImPlot::PlotScatter(("N" + std::to_string(i)).c_str(), xSec.data(), yVal.data(), size);
			}

			ImPlot::EndPlot();
		}
	}
	else { // Non-edge case behavior (Scrolling raster of size window)

		const double secPerSample = 1.0 / (double)m_fSampRate;
		ImPlot::SetNextAxisLimits(ImAxis_X1, startingStreamSampleCt * secPerSample, (startingStreamSampleCt + window) * secPerSample, ImPlotCond_Always);
		ImPlot::SetNextAxisLimits(ImAxis_Y1, yLo, yHi, ImPlotCond_Always);  // TODO, allow to look at more specific neurons

		if (ImPlot::BeginPlot("Raster plot:", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoTitle)) {
			ImPlot::SetupAxes("Time (hh:mm:ss)", "Neuron");
			ImPlot::SetupAxisFormat(ImAxis_X1, RasterTimeAxisFormatter);

			const double startSec = startingStreamSampleCt * secPerSample;
			const double endSec = (startingStreamSampleCt + window) * secPerSample;
			static const double kStepLadder[] = { 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600 };
			double rawStep = (endSec - startSec) / 6.0; // aim for ~6 ticks
			double step = kStepLadder[0];
			for (double cand : kStepLadder) { step = cand; if (cand >= rawStep) break; }
			std::vector<double> tickSec;
			for (double t = std::ceil(startSec / step) * step; t <= endSec; t += step)
				tickSec.push_back(t);
			if (!tickSec.empty())
				ImPlot::SetupAxisTicks(ImAxis_X1, tickSec.data(), (int)tickSec.size());

			for (int i = 0; i < m_lT; i++) {
				std::lock_guard<std::mutex> lock(m_bNeurons[i]->spikeTimeMutex);
				if (m_bNeurons[i]->m_vSpikeTime.empty())
					continue;

				// Find index of first element in m_vSpikeTimes that occurs after startingStreamSampleCt
				int startingIndex = -1;
				if (m_bNeurons[i]->m_vSpikeTime[0] > startingStreamSampleCt) { // If first spike in bounds, all in bounds
					startingIndex = 0;
				}
				else {
					for (int j = m_bNeurons[i]->m_vSpikeTime.size() - 1; j >= 0; j--) { // Iterate backwards for efficiency
						if (m_bNeurons[i]->m_vSpikeTime[j] < startingStreamSampleCt) {
							startingIndex = j + 1;
							break;
						}
					}
				}

				if (startingIndex == -1)
					continue;
				
					
				size = m_bNeurons[i]->m_vSpikeTime.size() - startingIndex;
				xSec.resize(size);
				for (int k = 0; k < size; k++)
				{
					xSec[k] = m_bNeurons[i]->m_vSpikeTime[startingIndex + k] * secPerSample;
				}
				yVal.assign(size, (double)i);

				ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0.8F, ImVec4(0.26f, 0.59f, 0.98f, 0.78f), IMPLOT_AUTO, ImVec4(0.26f, 0.59f, 0.98f, 0.78f));
				ImPlot::PlotScatter(("N" + std::to_string(i)).c_str(), xSec.data(), yVal.data(), size);
			}

			ImPlot::EndPlot();
		}
	}

	ImGui::End();
}

void OutputGuiTab::displayNeuronInfo(const ImVec2 windowCenter, bool &showNeuronInfo) {
	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

	ImGui::Begin("Neurons", &showNeuronInfo);

	std::string txt;
	for (int ij = 0; ij < m_lT; ij++) {
		if (m_bFittoActive && m_bNeurons[ij]->GetTotSpikeCount() == 0)
			continue;
		txt = "N" + std::to_string(ij);
		if (ImGui::Button(txt.c_str())) {
			m_bNeurons[ij]->SelectNeuron();
		}
		ImGui::SameLine();
		txt = " FR: " + std::_Floating_to_string("%.2f", m_bNeurons[ij]->GetSpikeRate()) + "Hz, nSpikes: " + std::to_string(m_bNeurons[ij]->GetTotSpikeCount());
		ImGui::Text(txt.c_str());
	}
	ImGui::End();
}



void OutputGuiTab::setMaxScanWindow(long m_lMaxScanWindow, float m_fSampRate) {
	// Convert m_lMaxScanWindow (stream sample count) to ms
	maxScanWindow = m_lMaxScanWindow / (m_fSampRate / 1000);
}


void OutputGuiTab::plotVRMS(const ImVec2 windowCenter, bool &showVRMS) {
	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
	ImGui::SetNextWindowClass(&plotWindowClass);
	if (!ImGui::Begin("VRMS over time", &showVRMS)) { ImGui::End(); return; }

	static int history = 1000;
	ImGui::InputInt("History (s):", &history, 50, 500);

	double *PlotArray = m_vdVRMS.data();
	int ToShow = (history > m_vdVRMS.size()) ? m_vdVRMS.size() : history;
	PlotArray += m_vdVRMS.size() - ToShow;

	if (ImPlot::BeginPlot("VRMS vs. Time", ImVec2(-1, -1))) {
		ImPlot::PlotLine("VRMS", PlotArray, ToShow);
		ImPlot::EndPlot();
	}
	ImGui::End();
}

void OutputGuiTab::plotP2P(const ImVec2 windowCenter, bool &showP2P) {
	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
	ImGui::SetNextWindowClass(&plotWindowClass);
	if (!ImGui::Begin("P2P over time", &showP2P)) { ImGui::End(); return; }

	static int history = 1000;
	ImGui::InputInt("History (s):", &history, 2, 60);

	float *PlotArray = m_vfP2P.data();
	int ToShow = (history > m_vfP2P.size()) ? m_vfP2P.size() : history;
	PlotArray += m_vfP2P.size() - ToShow;

	if (ImPlot::BeginPlot("P2P vs. Time", ImVec2(-1, -1))) {
		ImPlot::PlotLine("P2P", PlotArray, ToShow);
		ImPlot::EndPlot();
	}
	ImGui::End();
}

void OutputGuiTab::plotProcessTimes(const ImVec2 windowCenter, bool &showProcessTimes) {
	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowClass(&plotWindowClass);
	if (!ImGui::Begin("Processing time distribution", &showProcessTimes)) { ImGui::End(); return; }

	static bool	  useFullHistory = true;
	static int	  history = 30; // Seconds

	ImGui::Checkbox("Use entire history", &useFullHistory);
	if (!useFullHistory) {
		ImGui::SetNextItemWidth(75); ImGui::SameLine();
		ImGui::InputInt("History (s):", &history, 5, 20);
		history = max(5, history); // TODO make clamped input accessible
	}

	// history is in seconds but each element in m_vProcessTimes represents about a maxScanWindow ms 
	int historyNBatches = history * 1000 / maxScanWindow;

	processingTimeMutex.lock();
	int len = m_vProcessTimes.size();
	int toShow = useFullHistory || (historyNBatches > len) ? len : historyNBatches;
	int toSkip = len - toShow;

	double mean = 0.0, stdDev = 0.0, inTimePerc = 0.0;
	if (toShow > 0) {
		// Calculate mean
		mean = (double)std::reduce(m_vProcessTimes.begin() + toSkip, m_vProcessTimes.end()) / toShow;
		// Calculate std
    	std::vector<double> diff(toShow);
    	std::transform(m_vProcessTimes.begin() + toSkip, m_vProcessTimes.end(), diff.begin(), [mean](double x) { return x - mean; });
    	double sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    	stdDev = std::sqrt(sqSum / toShow);


		double inTimeCount = std::count_if(m_vProcessTimes.begin() + toSkip, m_vProcessTimes.end(), [&](long const val) { return val < maxScanWindow; });
    	inTimePerc = inTimeCount / toShow;
	}

	processingTimeMutex.unlock();

	static int bins = 50;
	setHistogramBins(bins);

	static int range = 100;
	ImGui::SameLine();
	setRange(range, 10, 500);

	ImGui::Text("Mean: %.2f,  STD: %.2f, Percentage on time: %.2f", mean, stdDev, inTimePerc);

	static const double kYTiers[]    = { 0.15, 0.20, 0.30, 0.50, 0.80, 1.0, 1.5, 2.0 };
	static const int    kNumYTiers   = (int)(sizeof(kYTiers) / sizeof(kYTiers[0]));
	static int          yTier        = 0;    // index into kYTiers; 0 is the floor
	static double       lastPeak     = 0.0;  // density peak returned by PlotHistogram last frame
	static int          framesBelow  = 0;    // consecutive frames the peak fit the lower tier
	const double        kFillUp      = 0.85; // step up when the peak exceeds 85% of the current tier
	const int           kDownHold    = 120;  // ~2 s at 60 fps the peak must stay low before stepping down

	// Step up immediately -- clipping is the worst outcome, so bump until the peak fits.
	while (yTier < kNumYTiers - 1 && lastPeak > kFillUp * kYTiers[yTier]) {
		yTier++;
		framesBelow = 0;
	}
	// Step down only after the peak stays comfortably inside the next-lower tier.
	if (yTier > 0 && lastPeak < kFillUp * kYTiers[yTier - 1]) {
		if (++framesBelow >= kDownHold) {
			yTier--;
			framesBelow = 0;
		}
	} else {
		framesBelow = 0;
	}
	const double yMax = kYTiers[yTier];


	if (ImPlot::BeginPlot("Batch Processing Time Distribution", ImVec2(-1, -1))) {
		ImPlot::SetupAxes("Time (ms)", "Density", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
		ImPlot::SetupAxisLimits(ImAxis_X1, 0, range, ImPlotCond_Always);
		ImPlot::SetupAxisLimits(ImAxis_Y1, 0, yMax, ImPlotCond_Always);
		ImPlot::SetupLegend(ImPlotLocation_NorthEast);
		ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
		processingTimeMutex.lock();
		lastPeak = ImPlot::PlotHistogram("ProcessTimes", m_vProcessTimes.data() + toSkip, toShow, bins, false, true, ImPlotRange(0, range));
		processingTimeMutex.unlock();
		ImPlot::PlotVLines("Batch Size", &maxScanWindow, 1);
		ImPlot::EndPlot();
	}
	ImGui::End();
}


void OutputGuiTab::displayTrialInfo(const ImVec2 windowCenter, bool &showTrialInfo) {
	ImGui::SetNextWindowPos(windowCenter + ImVec2(0, 300), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(340, 90), ImGuiCond_FirstUseEver);

	ImGui::SetNextWindowClass(&plotWindowClass);
	ImGui::Begin("Trial Info", &showTrialInfo);
	if (!isDecoding) {
		std::string txt = "# Trials: " + std::to_string(nTrials);
		ImGui::Text(txt.c_str());
		txt = "Label: " + std::to_string(label);
		ImGui::Text(txt.c_str());
	}
	else {
		std::string txt = "# Trials: " + std::to_string(nTrials) + ", # Correct: " + std::to_string(nCorrect);
		ImGui::Text(txt.c_str());
		txt = "Accuracy: " + std::to_string((float)nCorrect / (float)nTrials);
		ImGui::Text(txt.c_str());
		txt = "Label: " + std::to_string(label) + " ";
		ImGui::Text(txt.c_str()); ImGui::SameLine();
		if (predictLabel != -1) {
			txt = "Predicted: " + std::to_string(predictLabel) + " ";
			ImGui::Text(txt.c_str()); ImGui::SameLine();
			if (predictLabel == label)
				ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.2f, 0.8f), "Correct!");
			else
				ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Incorrect!");
			double conf = round(confidence * 10000) / 100;
			txt = "The model had " + std::to_string(conf).substr(0, 5) + "%% confidence.";
			ImGui::Text(txt.c_str());
		}
	}
	ImGui::End();
}


Neuron::Neuron(int number)
	: m_iYchanPos(0)
	, m_iChannumber(0)
	, m_inumber(number)
	, m_bSelected(false)
	, m_vSpikeTime()
	, m_vfSpikeRate()
	, m_SpikeRate(0)
	, spikeTimeMutex()
{
};


Neuron::~Neuron() {

}

// OutputGUI section
void Neuron::Update(OutputGuiTab* outputGUI) {
	static int	  iCrossCorHistory = 100;

	static std::vector<int> NeuronNums;
	static std::vector<std::vector<float>> CrossCorr;

	//flags needed so that correlation is not done every loop
	static bool	  bCorrCalculating = false;
	static bool	  bCorrCalcDone = false;
	static bool   bCorrCalcPressed = false;

	static std::thread CrossCorrThread;

	if (IsSelected() == true) {
		ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiCond_FirstUseEver);

		if (g_spikeStatsDockNode != 0)
			ImGui::SetNextWindowDockID(g_spikeStatsDockNode, ImGuiCond_Always);

		std::string txt = "Neuron " + std::to_string(m_inumber) + " Stats";
		if (!ImGui::Begin(txt.c_str(), &m_bSelected)) { ImGui::End(); return; }

		txt = " FR: " + std::_Floating_to_string("%.2f", GetSpikeRate()) + "Hz, nSpikes: " + std::to_string(GetTotSpikeCount());
		ImGui::Text(txt.c_str());

		if (ImGui::CollapsingHeader("Firing rate Plot"))
			plotFR(30000, BIN_SIZE);

		if (ImGui::CollapsingHeader("Amplitude Plot"))
			plotAmplitude();

		if (ImGui::CollapsingHeader("ISI Histogram"))
			plotISI();

		//if (ImGui::CollapsingHeader("PST Histogram"))
		//	plotPSTHs(outputGUI->eventTimes, outputGUI->eventLabels, outputGUI->m_fSampRate);

		//if (ImGui::CollapsingHeader("Autocorrelation"))
		//	plotAutoCorr();

		if (ImGui::CollapsingHeader("Cross-corellation")) {

			ImGui::InputInt("History (s):", &iCrossCorHistory, 1, 5);

			// If it is not currently calculating show a button to request if that is needed
			if (!bCorrCalcPressed) {

				// If pressed calculate the cross correlation
				if (ImGui::Button("Calculate") && !bCorrCalculating) {
					ImGui::SameLine(); HelpMarker("Calculating the (cross)correlation between multiple different neurons takes a lot of time, so instantiate it with this button.");

					bCorrCalcPressed = true;
					bCorrCalculating = true;

					NeuronNums = CalcNeuronNums(outputGUI->m_dChanpos, outputGUI->m_lT);


					// TODO: Implement the start variable so that we do not do the whole correlation while long in the experiment
					CrossCorrThread = std::thread(&Neuron::CorrelateWithNeighborsMult, this, outputGUI->m_bNeurons, NeuronNums, &bCorrCalcDone, &CrossCorr, 0);

				}
			}
			else {

				if (bCorrCalcDone) {
					plotCrossCorr(iCrossCorHistory, CrossCorr, NeuronNums);

					if (CrossCorrThread.joinable())
						CrossCorrThread.join();

				}
				else {
					ImGui::Text("Calculating Cross-Correlation... ");
				}


				if (ImGui::Button("Recalculate")) {
					bCorrCalcPressed = false;
					bCorrCalcDone = false;
					bCorrCalculating = false;


				}
			}
		}
		ImGui::End();
	}
}


// GPT-04-mini-high wrote this. TODO: fully verify correctness
void Neuron::plotFR(float sampRate, int binSize) {
	// 1) User controls: total history & smoothing window (both in seconds)
	static int history_s = 5;
	static int smooth_s = 1;
	ImGui::InputInt("History (s):", &history_s, 1, 5);
	ImGui::InputInt("Window (s):", &smooth_s, 1, 5);
	history_s = max(1, history_s);
	smooth_s = max(1, smooth_s);

	// 2) Snapshot per-bin spike counts under lock
	std::vector<float> spikes;
	{
		std::lock_guard<std::mutex> lk(binMutex);
		spikes = m_vfSpikesPerBin;
	}

	size_t nbins = spikes.size();
	if (nbins == 0) {
		ImGui::Text("No data yet");
		return;
	}

	// 3) Bin duration in seconds
	const float binDur_s = float(binSize) / sampRate;

	// 4) Number of whole bins covering the smoothing window
	size_t smoothBins = size_t(std::floor(smooth_s / binDur_s));
	smoothBins = std::clamp(smoothBins, size_t(1), nbins);

	// 5) Compute smoothed firing rate: sum spikes over smoothBins, divide by actual window duration
	size_t npts = nbins - smoothBins + 1;
	std::vector<float> y_smooth(npts);
	float sum = std::accumulate(spikes.begin(), spikes.begin() + smoothBins, 0.0f);
	const float winDur_s = smoothBins * binDur_s;
	y_smooth[0] = sum / winDur_s;
	for (size_t i = smoothBins; i < nbins; ++i) {
		sum += spikes[i] - spikes[i - smoothBins];
		y_smooth[i - smoothBins + 1] = sum / winDur_s;
	}

	// 6) Show only last history_s seconds worth of points
	size_t ptsToShow = min(
		npts,
		size_t(std::ceil(history_s / binDur_s))
	);
	size_t startIdx = npts - ptsToShow;

	// 7) Build time axis from -history_s to 0
	std::vector<float> x(ptsToShow), y(ptsToShow);
	for (size_t i = 0; i < ptsToShow; ++i) {
		x[i] = (float(i) - float(ptsToShow - 1)) * binDur_s;
		y[i] = y_smooth[startIdx + i];
	}

	// 8) Plot with fixed X limits
	ImPlot::SetNextAxisLimits(ImAxis_X1, -float(history_s), 0.0f, ImPlotCond_Always);
	ImPlot::BeginPlot("Firing Rate (sliding)", ImVec2(-1, 0));
	ImPlot::SetupAxes("Time (s)", "Rate (Hz)");
	ImPlot::PlotLine("FR", x.data(), y.data(), int(ptsToShow));
	ImPlot::EndPlot();
}


void Neuron::plotAutoCorr()
{
	static int history = 100;
	ImGui::InputInt("History (s):", &history, 1, 5);

	//Point to start of data
	std::vector<float> autocorr = AutoCorrelation();

	//Calculate how much to show
	size_t ToShow = (history > autocorr.size()) ? autocorr.size() : history;

	ImPlot::BeginPlot("Autocorrelation", ImVec2(-1, 0));
	ImPlot::PlotLine("Autocorrelation", autocorr.data() + autocorr.size() - ToShow, ToShow);
	ImPlot::EndPlot();
}

// calculate the Neurons within a specific range (default is 10 channels)
std::vector<int> Neuron::CalcNeuronNums(std::vector<double> Channums, long lT, int ChanRange) {

	std::vector<int> NeuronNums;

	for (int i = 0; i < lT; i++) {
		//Check if they fall within range
		if (std::abs(m_iChannumber - Channums[i]) <= ChanRange)
			NeuronNums.push_back(i);
	}

	return NeuronNums;
}

void Neuron::plotCrossCorr(int History, OutputGuiTab* outputGUI)
{
	// static variables as these don't change
	static std::vector<double> Channums = outputGUI->m_dChanpos;
	static long lT = outputGUI->m_lT;

	// amount of channels that the neuron looks up or down
	// TODO: make this a (member) variable
	int iChanRange = 10;

	std::vector<int> NeuronNums = CalcNeuronNums(Channums, lT, iChanRange);

	std::vector<std::vector<float> > crosscorr = CorrelateWithNeighbors(outputGUI->m_bNeurons, NeuronNums);
	//point to star

	std::string num;

	for (int j = 0; j < NeuronNums.size(); j++) {

		float *plotArray;

		num = std::to_string(NeuronNums[j]);

		if (ImGui::CollapsingHeader(std::string("Correlation with Neuron ").append(num).c_str())) {
			plotArray = crosscorr[j].data();

			//calculate how much to show
			size_t toshow = (History > crosscorr[j].size()) ? crosscorr[j].size() : History;

			//go to end of data and decrement
			plotArray += crosscorr[j].size() - toshow;

			ImPlot::BeginPlot(std::string("Cross Correlation with Neuron ").append(num).c_str(), ImVec2(-1, 0));
			ImPlot::PlotLine("Cross Correlation", plotArray, toshow);
			ImPlot::EndPlot();
		}

	}

}

void Neuron::plotCrossCorr(int History, std::vector<std::vector<float> >& crosscorr, std::vector<int>& NeuronNums)
{
	std::string num;

	for (int j = 0; j < NeuronNums.size(); j++) {
		float *plotArray;

		num = std::to_string(NeuronNums[j]);

		if (ImGui::CollapsingHeader(std::string("Correlation with Neuron ").append(num).c_str())) {

			plotArray = crosscorr[j].data();

			//calculate how much to show
			size_t toshow = (History > crosscorr[j].size()) ? crosscorr[j].size() : History;

			//go to end of data and decrement
			plotArray += crosscorr[j].size() - toshow;

			ImPlot::BeginPlot(std::string("Cross Correlation with Neuron ").append(num).c_str());
			ImPlot::PlotLine("Cross Correlation", plotArray, toshow);
			ImPlot::EndPlot();
		}
	}
}

void Neuron::plotAmplitude() {
	static int history = 1000;
	ImGui::InputInt("Spikes to look back:", &history, 50, 500);

	//Point to start of data
	std::vector<float> amplitudes;
	{
		std::lock_guard<std::mutex>lk(ampMutex);
		amplitudes = m_vfSpikeAmplitude;
	}

	//Calculate how much to show
	size_t ToShow = min(history, amplitudes.size());

	ImPlot::BeginPlot("Amplitude Plot", ImVec2(-1, 0));
	ImPlot::SetupAxes("Spike occurrences", "Amplitude", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
	ImPlot::PlotLine("Measured Amplitudes", amplitudes.data() + amplitudes.size() - ToShow, ToShow);
	ImPlot::EndPlot();
}

void Neuron::plotISI() {
	static int bins = 100; // Default values
	static int range = 100;

	// ISI calculation
	std::vector<long> ISIs;
	{
		std::lock_guard<std::mutex> lk(spikeTimeMutex);
		std::adjacent_difference(m_vSpikeTime.begin(), m_vSpikeTime.end(), std::back_inserter(ISIs));
	}

	size_t SampsToShow = ISIs.size() - 1;
	if (!ISIs.empty()) ISIs.erase(ISIs.begin()); // drop first element because adjacent_difference keeps first element
	float samp2ms = 1000.0f / 30000;
	for (auto &d : ISIs) {
		d = long(d * samp2ms);  // now each ISI is in ms
	}
	setHistogramBins(bins);
	ImGui::SameLine();
	setRange(range, 20, 500);

	ImPlot::BeginPlot("ISI", ImVec2(-1, 0));
	ImPlot::SetupAxes("Inter-Spike Interval (ms)", "Density", ImPlotAxisFlags_AutoFit);
	ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
	ImPlot::PlotHistogram("ISI", ISIs.data(), SampsToShow, bins, false, true, ImPlotRange(0, range));
	ImPlot::EndPlot();
}

void Neuron::plotPSTH(std::vector<long> eventTimes, std::vector<int16_t> eventLabels, float sampRate, int bins, int binSize, int binSizeCts, int rangeCts, int negRange, int negRangeCts, int16_t label) { // TODO change function so PST only calculated when new event is added
	// Indexes to keep track
	int binIdx = 0;
	int eventIdx = 0;

	std::vector<float> PSTs(bins);
	{
		std::lock_guard<std::mutex> lk(spikeTimeMutex);
		for (size_t i = 0; i < m_vSpikeTime.size(); i++) {
			// While spike is after current event's range, move onto next event
			while (eventIdx < eventTimes.size() && m_vSpikeTime[i] > eventTimes[eventIdx] + rangeCts) {
				eventIdx++;
				binIdx = 0;
			}

			// Exit out if passed last event
			if (eventIdx == eventTimes.size())
				break;

			// If label is -1, we use all events for plot. Else we only use events where the eventLabel == label
			if (label != -1 && label != eventLabels[eventIdx])
				continue;

			// If spike in event's range
			if (eventTimes[eventIdx] - negRangeCts <= m_vSpikeTime[i] && m_vSpikeTime[i] <= eventTimes[eventIdx] + rangeCts) {
				// While spike is after current bin's range, move onto next bin
				while (binIdx < bins && m_vSpikeTime[i] > eventTimes[eventIdx] - negRangeCts + binSizeCts * (binIdx + 1))
					binIdx++;

				// Continue if passed last bin of event
				if (binIdx == bins)
					continue;

				PSTs[binIdx]++;
			}
		}
	}

	// Determine the number of events for this label.
	int nEvents = (label == -1) ? eventLabels.size() : std::count(eventLabels.begin(), eventLabels.end(), label);

	// Normalize bins (bin starts as # of spikes in nEvents durations of binSizeCts. Divide by binSizeCts and eventTimes.size() -> spikes/streamSampleCts. Multiply by sampRate -> spikes/sec)
	if (nEvents != 0) {
		for (float &bin : PSTs)
			bin *= sampRate / (nEvents * binSizeCts);
	}

	// Create the x data for the bar plot
	std::vector<float> x(bins);
	for (int b = 0; b < bins; b++) {
		x[b] = b * binSize + binSize * 0.5 - negRange;
	}

	std::string plotTitle = (label == -1) ? "PSTH across all events" : "PSTH for label " + std::to_string(label);

	ImPlot::BeginPlot(plotTitle.c_str(), ImVec2(-1, 0));
	if (label == -1 || label == 0) // Currently all plots fit to the y axis of psth of label 0, but should be to whichever is max
		ImPlot::SetupAxes("Peristimulus time (ms)", "Firing Rate", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
	else
		ImPlot::SetupAxis(ImAxis_X1, "Peristimulus time (ms)", ImPlotAxisFlags_AutoFit);

	ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
	ImPlot::PlotBars("PSTH", x.data(), PSTs.data(), bins, binSize);
	ImPlot::EndPlot();
}

void Neuron::plotPSTHs(std::vector<long> eventTimes, std::vector<int16_t> eventLabels, float sampRate) {
	static bool plotIndividual = false;
	static int bins = 50;
	static int range = 500; // ms after stimulus onset the plots show
	static int negRange = 0; // ms before stimulus onset the plots show

	// If no events yet, break out
	if (!eventTimes.size()) {
		ImGui::Text("No event yet!");
		return;
	}

	ImGui::Checkbox("Plot PSTHs by label", &plotIndividual);

	setHistogramBins(bins);
	setRange(range, 50, 3000);
	ImGui::SameLine(); ImGui::SetNextItemWidth(100);
	ImGui::SliderInt("##NegRange", &negRange, 0, 1000);
	ImGui::SameLine(); ImGui::Text("Neg. Range");

	// binSize in ms
	int binSize = (range + negRange) / bins;

	// Convert ms to stream sample counts
	int binSizeCts = binSize * (sampRate / 1000);
	int rangeCts = range * (sampRate / 1000);
	int negRangeCts = negRange * (sampRate / 1000);

	if (!plotIndividual) // Plot PSTH for all label types
		plotPSTH(eventTimes, eventLabels, sampRate, bins, binSize, binSizeCts, rangeCts, negRange, negRangeCts, -1);
	else { // Plot PSTHs by label
		int16_t nLabels = *std::max_element(eventLabels.begin(), eventLabels.end()) + 1;

		ImPlot::BeginSubplots("PSTHs", 1, nLabels, ImVec2(-1, 0), ImPlotSubplotFlags_LinkAllY);

		for (int16_t label = 0; label < nLabels; label++) {
			plotPSTH(eventTimes, eventLabels, sampRate, bins, binSize, binSizeCts, rangeCts, negRange, negRangeCts, label);
		}

		ImPlot::EndSubplots();
	}
}

void Neuron::AddSpike(long time) {

	std::lock_guard<std::mutex> lk(spikeTimeMutex);
	m_vSpikeTime.push_back(time);
}

std::vector<float> Neuron::AutoCorrelation(int start, int end) {

	// start and end are currently not used, do later if everything is working.

	std::lock_guard<std::mutex> lk(binMutex);

	int sum = 0;
	for (int i = 0; i < m_vfSpikesPerBin.size(); i++) {
		sum += m_vfSpikesPerBin[i];
	}

	float mean = float(sum) / m_vfSpikesPerBin.size();

	std::vector<float> autocorr(m_vfSpikesPerBin.size() / 2, 0);

	for (int lag = 0; lag < autocorr.size(); lag++) {

		float numerator = 0;
		float denominator = 0;

		for (int i = 0; i < m_vfSpikesPerBin.size() - lag; i++) {
			float dev = m_vfSpikesPerBin[i] - mean;
			numerator += dev * (m_vfSpikesPerBin[i + lag] - mean);
			denominator += dev * dev;
		}

		if (denominator != 0)
			autocorr[lag] = numerator / denominator;

	}

	return autocorr;
}


std::vector<std::vector<float> > Neuron::CorrelateWithNeighbors(Neuron **Neurons, std::vector<int>& NeuronNums) {

	//std::vector<std::vector<float>> output(NeuronNums.size());
	std::vector<std::vector<float>> output;

	output.reserve(NeuronNums.size());

	for (int j = 0; j < NeuronNums.size(); j++) {
	    // snapshot both sides under lock
		std::vector<float> fcopy, gcopy;
		{
			std::lock_guard<std::mutex> lk(binMutex);
			std::lock_guard<std::mutex> lk2(Neurons[NeuronNums[j]]->binMutex);
			fcopy = m_vfSpikesPerBin;
			gcopy = Neurons[NeuronNums[j]]->m_vfSpikesPerBin;
		}
		output.emplace_back(Correlate(fcopy, gcopy));
	}

	return output;
}

void Neuron::CorrelateWithNeighborsMult(Neuron **Neurons, std::vector<int>& NeuronNums, bool *done, std::vector<std::vector<float>> *output, int start) {

	// clear the vector and fill resize it agian to the necessary size
	output->clear();
	output->reserve(NeuronNums.size());

	std::lock_guard<std::mutex> lk(binMutex);
	for (int j = 0; j < NeuronNums.size(); j++) {
		std::lock_guard<std::mutex> lk2(Neurons[NeuronNums[j]]->binMutex);
		output->emplace_back(Correlate(m_vfSpikesPerBin, Neurons[NeuronNums[j]]->m_vfSpikesPerBin));
	}

	*done = true;
}


std::vector<float> Neuron::Correlate(std::vector<float> const &f, std::vector<float> const &g, int start) {

	// set sizes
	int const nf = f.size() - start;
	int const ng = g.size() - start;

	// size of the new signal
	int const n = nf + ng - 1;

	if (nf < 0 || ng < 0 || n <= 0)
		return std::vector<float>();

	// Initialize variables
	std::vector<float> out(n);

	// loop over elements
	for (int i = 0; i < n; ++i) {

		//Initialize variables for size 
		int const jmn = (i >= ng - 1) ? i - (ng - 1) : start;
		int const jmx = (i < nf - 1) ? i : nf - 1;

		for (int j = jmn; j <= jmx; ++j) {
			//out[i] += (f[j] * g[i - j])
			out[i] += (f[j] * g[(ng - 1) + j - i]);

		}
	}

	return out;
}

void Neuron::CheckAndAddBin(const int BinSize, const int streamSampleCount) {

	//Check amount of Bins already processed
	binMutex.lock();
	int added = m_vfSpikesPerBin.size() * BinSize;
	binMutex.unlock();
	int ToAdd = (streamSampleCount - added) / BinSize;
	int sum = 0;
	int left = 0;
	int right = 0;

	std::lock_guard<std::mutex> lk(spikeTimeMutex);

	// How many to add
	for (int i = 0; i < ToAdd; i++) {
		left = added + i * BinSize;
		right = added + (i + 1) * BinSize;
		sum = 0;

		// Loop from end to beginning over the vector and add all spikes that are within the bin. Spiketime vector is still sorted right?
		for (auto rit = m_vSpikeTime.rbegin(); rit != m_vSpikeTime.rend(); ++rit) {
			if (*rit >= right)
				continue;
			else if (*rit < right && *rit >= left)
				sum++;
			else if (*rit < left)
				break;
			else
				std::cout << "Error while filling the bins" << std::endl;
		}
		{
			std::lock_guard<std::mutex> lk(binMutex);
			m_vfSpikesPerBin.push_back(sum);
		}
	}
}


void Neuron::AddSpikeAmplitude(float amp) {
	std::lock_guard<std::mutex> lk(ampMutex);
	m_vfSpikeAmplitude.push_back(amp);
}

void Neuron::SetChanNum(int Channum) {
	m_iChannumber = Channum;
}

long Neuron::GetTotSpikeCount() {
	std::lock_guard<std::mutex> lk(spikeTimeMutex);
	return m_vSpikeTime.size();
}

float Neuron::GetSpikeRate() {
	return m_SpikeRate;
}

void Neuron::AddSpikeRate() {
	std::lock_guard<std::mutex> lk(spikeRateMutex);
	m_vfSpikeRate.push_back(m_SpikeRate);
}

void Neuron::AddBin(float BinVal) {
	std::lock_guard<std::mutex> lk(binMutex);
	m_vfSpikesPerBin.push_back(BinVal);
}

void Neuron::SelectNeuron() {
	m_bSelected = true;
}

void Neuron::DeselectNeuron() {
	m_bSelected = false;
}

bool Neuron::IsSelected() {
	return m_bSelected;
}

void Neuron::CalcSpikeRate(long *streamSampleCount, long TimeWindow, float SamplingRate) {
	//Initialize variables
	long count = 0;
	long streamSampleCountFrame;
	float TimeWindowSecs = TimeWindow * 1.f / 1'000;
	long AllowedDiff = TimeWindowSecs * SamplingRate;

	while (*streamSampleCount == 0) {
		std::this_thread::sleep_for(std::chrono::microseconds(1'000)); // TODO switch to Sleep()
	}
	streamSampleCountFrame = *streamSampleCount; // Once streamSampleCount != 0, take its value. Check here if problems

	std::lock_guard<std::mutex> lk(spikeTimeMutex);
	if (streamSampleCountFrame < AllowedDiff) {
		m_SpikeRate = m_vSpikeTime.size() * SamplingRate / streamSampleCountFrame;
	}
	else {
		for (auto it = m_vSpikeTime.rbegin(); it != m_vSpikeTime.rend(); it++) {
			if ((streamSampleCountFrame - *it) >= AllowedDiff) {
				break;
			}
			count++;
		}

		m_SpikeRate = count / TimeWindowSecs;
	}

}
