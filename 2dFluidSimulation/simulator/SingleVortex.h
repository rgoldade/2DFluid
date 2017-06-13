#pragma once

#include "core.h"
#include "util.h"
#include "Renderer.h"

///////////////////////////////////
//
// SingleVortex.h
// Ryan Goldade 2017
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