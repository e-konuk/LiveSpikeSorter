#ifndef SPIKESPAYLOAD_H
#define SPIKESPAYLOAD_H
#include <cereal/archives/binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include "SerializationHelpers.h"

struct SpikesPayload {
	long        recordingOffset;
	long		streamSampleCt;
	
	std::vector<long> Times;
	std::vector<long> Templates;
	std::vector<float> Amplitudes;

	double		VRMS;
	float		P2P;
	long		processTime;	

	// These used for Decoder->GUI but not LiveSorter->Decoder
	long eventStreamSampleCt;
	int16_t		label;
	int16_t		nTrials;

	long		driftUpdateCt = 0;    
	float		driftShiftUm = 0.0f;  

	// OMP confidence
	std::vector<float> MatchScores;

	// Using the Cereal serialization library
	template <class Archive>
	void serialize(Archive & ar)
	{
		ar(recordingOffset,
			streamSampleCt,
			Times,
			Templates,
			Amplitudes,
			VRMS,
			P2P,
			processTime,
			eventStreamSampleCt,
			label,
			nTrials,
			driftUpdateCt, 
			driftShiftUm,
			MatchScores);  
	}
};
#endif
