#ifndef LIBRARY_NOISE_H
#define LIBRARY_NOISE_H

#include "Common.h"

///////////////////////////////////
//
// Borrowed and modified from Robert Bridson's
// public code.
//
///////////////////////////////////

struct Noise3
{
	Noise3(unsigned seed = 171717);
	virtual ~Noise3() {};

	void reinitialize(unsigned seed);
	Real operator()(Real x, Real y, Real z) const;
	Real operator()(const Vec3R &x) const { return (*this)(x[0], x[1], x[2]); }

protected:

	static constexpr unsigned n = 128;
	Vec3R myBasis[n];
	int myPerm[n];

	unsigned hashIndex(int i, int j, int k) const
	{
		return myPerm[(myPerm[(myPerm[i%n] + j) % n] + k) % n];
	}
};

// FlowNoise classes - time varying versions of some of the above
struct FlowNoise3 : public Noise3
{
	FlowNoise3(unsigned seed = 171717, Real spinVariation = 0.2);
	void setTime(Real time); // period of repetition is approximately 1

protected:
	
	Vec3R myOriginalBasis[n];
	Real mySpinRate[n];
	Vec3R mySpinAxis[n];
};

#endif