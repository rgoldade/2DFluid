#pragma once

#include "core.h"

struct Noise3
{
	Noise3(unsigned int seed = 171717);
	virtual ~Noise3() {};

	void reinitialize(unsigned int seed);
	double operator()(double x, double y, double z) const;
	double operator()(const Vec3d &x) const { return (*this)(x[0], x[1], x[2]); }

protected:
	static const unsigned int n = 128;
	Vec3d basis[n];
	int perm[n];

	unsigned int hash_index(int i, int j, int k) const
	{
		return perm[(perm[(perm[i%n] + j) % n] + k) % n];
	}
};

// FlowNoise classes - time varying versions of some of the above
struct FlowNoise3 : public Noise3
{
	FlowNoise3(unsigned int seed = 171717, double spin_variation = 0.2);
	void set_time(double t); // period of repetition is approximately 1

protected:
	Vec3d original_basis[n];
	double spin_rate[n];
	Vec3d spin_axis[n];
};