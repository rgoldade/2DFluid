#ifndef FLUIDSIM2D_NOISE_H
#define FLUIDSIM2D_NOISE_H

#include "Utilities.h"

///////////////////////////////////
//
// Borrowed and modified from Robert Bridson's
// public code.
//
///////////////////////////////////

namespace FluidSim2D
{

struct Noise3
{
	Noise3(unsigned seed = 171717);
	virtual ~Noise3() {};

	void reinitialize(unsigned seed);
	double operator()(double x, double y, double z) const;
	double operator()(const Vec3d& x) const { return (*this)(x[0], x[1], x[2]); }

protected:

	static constexpr unsigned n = 128;
	Vec3d myBasis[n];
	int myPerm[n];

	unsigned hashIndex(int i, int j, int k) const
	{
		return myPerm[(myPerm[(myPerm[i%n] + j) % n] + k) % n];
	}
};

// FlowNoise classes - time varying versions of some of the above
struct FlowNoise3 : public Noise3
{
	FlowNoise3(unsigned seed = 171717, double spinVariation = 0.2);
	void setTime(double time); // period of repetition is approximately 1

protected:

	Vec3d myOriginalBasis[n];
	double mySpinRate[n];
	Vec3d mySpinAxis[n];
};

}
#endif