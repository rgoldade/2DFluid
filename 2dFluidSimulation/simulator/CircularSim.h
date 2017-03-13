#pragma once

#include "core.h"
#include "Renderer.h"

///////////////////////////////////
//
// CircularSim.h
// Ryan Goldade 2016
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
			for (Real theta = 0.0; theta < 2 * M_PI; theta += M_PI/16.0)
			{
				Vec2d p0 = Vec2d(r * cos(theta),r * sin(theta)) + m_center;
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