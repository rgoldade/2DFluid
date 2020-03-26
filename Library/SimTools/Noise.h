#ifndef LIBRARY_NOISE_H
#define LIBRARY_NOISE_H

#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// Borrowed and modified from Robert Bridson's
// public code.
//
///////////////////////////////////

namespace FluidSim2D::SimTools
{

using namespace Utilities;

struct Noise3
{
	Noise3(unsigned seed = 171717);
	virtual ~Noise3() {};

	void reinitialize(unsigned seed);
	float operator()(float x, float y, float z) const;
	float operator()(const Vec3f& x) const { return (*this)(x[0], x[1], x[2]); }

protected:

	static constexpr unsigned n = 128;
	Vec3f myBasis[n];
	int myPerm[n];

	unsigned hashIndex(int i, int j, int k) const
	{
		return myPerm[(myPerm[(myPerm[i%n] + j) % n] + k) % n];
	}
};

// FlowNoise classes - time varying versions of some of the above
struct FlowNoise3 : public Noise3
{
	FlowNoise3(unsigned seed = 171717, float spinVariation = 0.2);
	void setTime(float time); // period of repetition is approximately 1

protected:

	Vec3f myOriginalBasis[n];
	float mySpinRate[n];
	Vec3f mySpinAxis[n];
};

}
#endif