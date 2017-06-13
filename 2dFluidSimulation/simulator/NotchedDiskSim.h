#pragma once

#include "core.h"
#include "util.h"
#include "Renderer.h"

///////////////////////////////////
//
// NotchedDiskSim.h
// Ryan Goldade 2017
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
			for (Real theta = 0.0; theta < 2 * M_PI; theta += M_PI/16.0)
			{
				Vec2d p0 = Vec2d(r * cos(theta),r * sin(theta));
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