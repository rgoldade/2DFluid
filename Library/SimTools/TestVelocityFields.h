#pragma once

#include "Core.h"
#include "Util.h"
#include "Renderer.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

////////////////////////////////////
//
// 2-D single vortex velocity field simulation
// from the standard surface tracker tests
//
////////////////////////////////////

class SingleVortexSim2D
{
public:
	SingleVortexSim2D() : m_time(0), m_totaltime(6)
	{}

	SingleVortexSim2D(Real t0, Real T) : m_time(t0), m_totaltime(T)
	{}

	void draw_sim_vectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> start_points;
		std::vector<Vec2R> end_points;
		for (Real r = 0.0; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.0; theta < 2 * M_PI; theta += M_PI / 16.0)
			{
				Vec2d p0 = Vec2d(r * cos(theta), r * sin(theta));
				Vec2d p1 = p0 + dt * (*this)(dt, p0);

				start_points.push_back(p0);
				end_points.push_back(p1);
			}
		}
		renderer.add_lines(start_points, end_points, colour);
	}

	void advance(double dt)
	{
		m_time += dt;
	}

	// Sample positions given in world space
	Vec2R operator()(Real dt, const Vec2R& pos) const
	{
		double sin_x = sin(M_PI * pos[0]);
		double sin_y = sin(M_PI * pos[1]);

		double cos_x = cos(M_PI * pos[0]);
		double cos_y = cos(M_PI * pos[1]);

		double u = 2.0 * sin_y * cos_y * sin_x * sin_x;
		double v = -2.0 * sin_x * cos_x * sin_y * sin_y;

		return Vec2R(u, v) * cos(M_PI * (m_time + dt) / m_totaltime);
	}

private:

	Real m_time, m_totaltime;
};

#include "Noise.h"

///////////////////////////////////
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

///////////////////////////////////
//
// Child class fluid sim that just creates a velocity field that
// rotates CCW. Useful sanity check for surface trackers.
//
////////////////////////////////////

class CircularSim2D
{
public:
	CircularSim2D() : m_center(Vec2d(0)), m_scale(1.) {}
	CircularSim2D(const Vec2R& c, Real scale = 1.) : m_center(c), m_scale(scale) {}

	void draw_sim_vectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> start_points;
		std::vector<Vec2R> end_points;
		for (Real r = 0.0; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.0; theta < 2 * M_PI; theta += M_PI / 16.0)
			{
				Vec2d p0 = Vec2d(r * cos(theta), r * sin(theta)) + m_center;
				Vec2d p1 = p0 + dt * (*this)(dt, p0);

				start_points.push_back(p0);
				end_points.push_back(p1);
			}
		}
		renderer.add_lines(start_points, end_points, colour);
	};

	inline Vec2R operator()(Real, const Vec2R& pos) const
	{
		return Vec2R(pos[1] - m_center[1], -(pos[0] - m_center[0])) * m_scale;
	}

private:
	Vec2d m_center;
	Real m_scale;
};

///////////////////////////////////
//
//
// Zalesak disk velocity field simulation
// Assumes you're running the notched disk IC
//
////////////////////////////////////

class NotchedDiskSim2D
{
public:

	void draw_sim_vectors(Renderer& renderer, Real dt = 1., Real radius = 10., const Vec3f& colour = Vec3f(0))
	{
		std::vector<Vec2R> start_points;
		std::vector<Vec2R> end_points;
		for (Real r = 0.0; r <= radius; r += radius / 10.)
		{
			for (Real theta = 0.0; theta < 2 * M_PI; theta += M_PI / 16.0)
			{
				Vec2d p0 = Vec2d(r * cos(theta), r * sin(theta));
				Vec2d p1 = p0 + dt * (*this)(dt, p0);

				start_points.push_back(p0);
				end_points.push_back(p1);
			}
		}
		renderer.add_lines(start_points, end_points, colour);
	};

	// Procedural velocity field
	inline Vec2R operator()(Real, const Vec2d& pos) const
	{
		return Vec2d((M_PI / 314.) * (50.0 - pos[1]), (M_PI / 314.) * (pos[0] - 50.0));
	}
};
