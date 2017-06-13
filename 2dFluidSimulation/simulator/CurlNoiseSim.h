#pragma once

#include "core.h"
#include "Renderer.h"
#include "Noise.h"

///////////////////////////////////
//
// CurlNoise.h
// Ryan Goldade 2017
//
// Curl noise velocity field generator.
// Adapted from El Topo
//
////////////////////////////////////

class CurlNoise2D
{
public:
	CurlNoise2D()
		: noise_lengthscale(1),
		noise_gain(1),
		noise(),
		m_dx(1E-4)
	{
		noise_lengthscale[0] = 1.5;
		noise_gain[0] = 1.3;
	}

	Vec2R get_velocity(const Vec2R &x) const
	{
		Vec2R v;
		v[0] = ((potential(x[0], x[1] + m_dx)[2] - potential(x[0], x[1] - m_dx)[2])
			- (potential(x[0], x[1], m_dx)[1] - potential(x[0], x[1], -m_dx)[1])) / (2 * m_dx);
		v[1] = ((potential(x[0], x[1], m_dx)[0] - potential(x[0], x[1], -m_dx)[0])
			- (potential(x[0] + m_dx, x[1])[2] - potential(x[0] - m_dx, x[1])[2])) / (2 * m_dx);

		return v;
	}

	Vec3R potential(Real x, Real y, Real z = 0.) const
	{
		Vec3R psi(0, 0, 0);
		Real height_factor = 0.5;

		static const Vec3R centre(0.0, 1.0, 0.0);
		static Real radius = 4.0;

		for (unsigned int i = 0; i<noise_lengthscale.size(); ++i)
		{
			Real sx = x / noise_lengthscale[i];
			Real sy = y / noise_lengthscale[i];
			Real sz = z / noise_lengthscale[i];

			Vec3R psi_i(0.f, 0.f, noise2(sx, sy, sz));

			Real dist = mag(Vec3R(x, y, z) - centre);
			Real scale = max((radius - dist) / radius, 0.0);
			psi_i *= scale;

			psi += height_factor * noise_gain[i] * psi_i;
		}

		return psi;
	}

	Real noise2(Real x, Real y, Real z) const { return noise(z - 203.994, x + 169.47, y - 205.31); }

	inline Vec2R operator()(Real, const Vec2R& pos) const
	{
		return get_velocity(pos);
	}

private:

	std::vector<Real> noise_lengthscale, noise_gain;
	Real m_dx;

	FlowNoise3 noise;
};