#pragma once

#include "Common.h"

struct Noise3
{
	Noise3(unsigned seed = 171717);
	virtual ~Noise3() {};

	void reinitialize(unsigned seed);
	Real operator()(Real x, Real y, Real z) const;
	Real operator()(const Vec3R &x) const { return (*this)(x[0], x[1], x[2]); }

protected:

	static constexpr unsigned n = 128;
	Vec3R basis[n];
	int perm[n];

	unsigned hash_index(int i, int j, int k) const
	{
		return perm[(perm[(perm[i%n] + j) % n] + k) % n];
	}
};

// FlowNoise classes - time varying versions of some of the above
struct FlowNoise3 : public Noise3
{
	FlowNoise3(unsigned seed = 171717, Real spin_variation = 0.2);
	void set_time(Real time); // period of repetition is approximately 1

protected:
	
	Vec3R original_basis[n];
	Real spin_rate[n];
	Vec3R spin_axis[n];
};
