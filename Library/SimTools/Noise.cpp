#include "Noise.h"

template<unsigned N>
Vec<Real, N> sampleSphere(unsigned &seed)
{
	Vec<Real, N> v;
	Real m2;
	do {
		for (unsigned i = 0; i < N; ++i) {
			v[i] = Util::randhashf(seed++, -1, 1);
		}
		m2 = mag2(v);
	} while (m2 > 1 || m2 == 0);
	return v / std::sqrt(m2);
}

//============================================================================

Noise3::Noise3(unsigned seed)
{
	for (unsigned i = 0; i < n; ++i)
	{
		myBasis[i] = sampleSphere<3>(seed);
		myPerm[i] = i;
	}
	reinitialize(seed);
}

void Noise3::reinitialize(unsigned seed)
{
	for (unsigned i = 1; i < n; ++i)
	{
		int j = Util::randhash(seed++) % (i + 1);
		std::swap(myPerm[i], myPerm[j]);
	}
}

Real Noise3::operator()(Real x, Real y, Real z) const
{
	Real floorx = std::floor(x), floory = std::floor(y), floorz = std::floor(z);

	int i = (int)floorx, j = (int)floory, k = (int)floorz;
	const Vec3R &n000 = myBasis[hashIndex(i, j, k)];
	const Vec3R &n100 = myBasis[hashIndex(i + 1, j, k)];
	const Vec3R &n010 = myBasis[hashIndex(i, j + 1, k)];
	const Vec3R &n110 = myBasis[hashIndex(i + 1, j + 1, k)];
	const Vec3R &n001 = myBasis[hashIndex(i, j, k + 1)];
	const Vec3R &n101 = myBasis[hashIndex(i + 1, j, k + 1)];
	const Vec3R &n011 = myBasis[hashIndex(i, j + 1, k + 1)];
	const Vec3R &n111 = myBasis[hashIndex(i + 1, j + 1, k + 1)];

	Real fx = x - floorx, fy = y - floory, fz = z - floorz;
	Real sx = fx * fx * fx * (10 - fx * (15 - fx * 6)),
			sy = fy * fy * fy * (10 - fy * (15 - fy * 6)),
			sz = fz * fz * fz * (10 - fz * (15 - fz * 6));

	return Util::trilerp(fx * n000[0] + fy * n000[1] + fz * n000[2],
						(fx - 1) * n100[0] + fy * n100[1] + fz * n100[2],
						fx * n010[0] + (fy - 1) * n010[1] + fz * n010[2],
						(fx - 1) * n110[0] + (fy - 1) * n110[1] + fz * n110[2],
						fx * n001[0] + fy * n001[1] + (fz - 1) * n001[2],
						(fx - 1) * n101[0] + fy * n101[1] + (fz - 1) * n101[2],
						fx * n011[0] + (fy - 1) * n011[1] + (fz - 1) * n011[2],
						(fx - 1) * n111[0] + (fy - 1) * n111[1] + (fz - 1) * n111[2],
						sx, sy, sz);
}

//============================================================================

FlowNoise3::FlowNoise3(unsigned seed, Real spin_variation)
	: Noise3(seed)
{
	seed += 8 * n; // probably avoids overlap with sequence used in initializing superclass Noise3
	for (unsigned i = 0; i < n; ++i)
	{
		myOriginalBasis[i] = myBasis[i];
		mySpinAxis[i] = sampleSphere<3>(seed);
		mySpinRate[i] = 2.0 * Util::PI * Util::randhashd(seed++, 0.1 - 0.5 * spin_variation, 0.1 + 0.5 * spin_variation);
	}
}

void FlowNoise3::setTime(Real t)
{
	for (unsigned i = 0; i < n; ++i) {
		Real theta = mySpinRate[i] * t;
		Real c = std::cos(theta), s = std::sin(theta);
		// form rotation matrix
		Real R00 = c + (1 - c) * Util::sqr(mySpinAxis[i][0]),
				R01 = (1 - c) * mySpinAxis[i][0] * mySpinAxis[i][1] - s * mySpinAxis[i][2],
				R02 = (1 - c) * mySpinAxis[i][0] * mySpinAxis[i][2] + s * mySpinAxis[i][1];
		Real R10 = (1 - c) * mySpinAxis[i][0] * mySpinAxis[i][1] + s * mySpinAxis[i][2],
				R11 = c + (1 - c) * Util::sqr(mySpinAxis[i][1]),
				R12 = (1 - c) * mySpinAxis[i][1] * mySpinAxis[i][2] - s * mySpinAxis[i][0];
		Real R20 = (1 - c) * mySpinAxis[i][0] * mySpinAxis[i][2] - s * mySpinAxis[i][1],
				R21 = (1 - c) * mySpinAxis[i][1] * mySpinAxis[i][2] + s * mySpinAxis[i][0],
				R22 = c + (1 - c) * Util::sqr(mySpinAxis[i][2]);
		myBasis[i][0] = R00 * myOriginalBasis[i][0] + R01 * myOriginalBasis[i][1] + R02 * myOriginalBasis[i][2];
		myBasis[i][1] = R10 * myOriginalBasis[i][0] + R11 * myOriginalBasis[i][1] + R12 * myOriginalBasis[i][2];
		myBasis[i][2] = R20 * myOriginalBasis[i][0] + R21 * myOriginalBasis[i][1] + R22 * myOriginalBasis[i][2];
	}
}
