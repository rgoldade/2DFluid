#include "Noise.h"

template<unsigned N>
Vec<Real, N> sample_sphere(unsigned &seed)
{
	Vec<Real, N> v;
	Real m2;
	do {
		for (unsigned i = 0; i < N; ++i) {
			v[i] = randhashf(seed++, -1, 1);
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
		basis[i] = sample_sphere<3>(seed);
		perm[i] = i;
	}
	reinitialize(seed);
}

void Noise3::reinitialize(unsigned seed)
{
	for (unsigned i = 1; i < n; ++i)
	{
		int j = randhash(seed++) % (i + 1);
		std::swap(perm[i], perm[j]);
	}
}

Real Noise3::operator()(Real x, Real y, Real z) const
{
	Real floorx = std::floor(x), floory = std::floor(y), floorz = std::floor(z);

	int i = (int)floorx, j = (int)floory, k = (int)floorz;
	const Vec3R &n000 = basis[hash_index(i, j, k)];
	const Vec3R &n100 = basis[hash_index(i + 1, j, k)];
	const Vec3R &n010 = basis[hash_index(i, j + 1, k)];
	const Vec3R &n110 = basis[hash_index(i + 1, j + 1, k)];
	const Vec3R &n001 = basis[hash_index(i, j, k + 1)];
	const Vec3R &n101 = basis[hash_index(i + 1, j, k + 1)];
	const Vec3R &n011 = basis[hash_index(i, j + 1, k + 1)];
	const Vec3R &n111 = basis[hash_index(i + 1, j + 1, k + 1)];

	Real fx = x - floorx, fy = y - floory, fz = z - floorz;
	Real sx = fx * fx * fx * (10 - fx * (15 - fx * 6)),
			sy = fy * fy * fy * (10 - fy * (15 - fy * 6)),
			sz = fz * fz * fz * (10 - fz * (15 - fz * 6));

	return trilerp(fx * n000[0] + fy * n000[1] + fz * n000[2],
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
		original_basis[i] = basis[i];
		spin_axis[i] = sample_sphere<3>(seed);
		spin_rate[i] = 2.0 * M_PI * randhashd(seed++, 0.1 - 0.5 * spin_variation, 0.1 + 0.5 * spin_variation);
	}
}

void FlowNoise3::set_time(Real t)
{
	for (unsigned i = 0; i < n; ++i) {
		Real theta = spin_rate[i] * t;
		Real c = std::cos(theta), s = std::sin(theta);
		// form rotation matrix
		Real R00 = c + (1 - c) * sqr(spin_axis[i][0]),
				R01 = (1 - c) * spin_axis[i][0] * spin_axis[i][1] - s * spin_axis[i][2],
				R02 = (1 - c) * spin_axis[i][0] * spin_axis[i][2] + s * spin_axis[i][1];
		Real R10 = (1 - c) * spin_axis[i][0] * spin_axis[i][1] + s * spin_axis[i][2],
				R11 = c + (1 - c) * sqr(spin_axis[i][1]),
				R12 = (1 - c) * spin_axis[i][1] * spin_axis[i][2] - s * spin_axis[i][0];
		Real R20 = (1 - c) * spin_axis[i][0] * spin_axis[i][2] - s * spin_axis[i][1],
				R21 = (1 - c) * spin_axis[i][1] * spin_axis[i][2] + s * spin_axis[i][0],
				R22 = c + (1 - c) * sqr(spin_axis[i][2]);
		basis[i][0] = R00 * original_basis[i][0] + R01 * original_basis[i][1] + R02 * original_basis[i][2];
		basis[i][1] = R10 * original_basis[i][0] + R11 * original_basis[i][1] + R12 * original_basis[i][2];
		basis[i][2] = R20 * original_basis[i][0] + R21 * original_basis[i][1] + R22 * original_basis[i][2];
	}
}
