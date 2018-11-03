#pragma once

#include "VectorGrid.h"
#include "LevelSet2D.h"
#include "ScalarGrid.h"

#include "Renderer.h"

///////////////////////////////////
//
// ViscositySolver.h/cpp
// Ryan Goldade 2017
//
// Variational viscosity solver.
// Uses ghost fluid weights for moving
// collisions. Uses volume control
// weights for the various tensor and
// velocity sample positions.
//
////////////////////////////////////

class ViscositySolver
{
	static constexpr int UNSOLVED = -2;
	static constexpr int COLLISION = -1;

public:

	ViscositySolver(Real dt, const LevelSet2D& surface, VectorGrid<Real>& vel,
					const LevelSet2D& collision, const VectorGrid<Real>& collision_vel)
		: m_dt(dt)
		, m_vel(vel)
		, m_surface(surface)
		, m_collision_vel(collision_vel)
		, m_collision(collision)
		{
			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(m_surface.is_matched(m_collision));

			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(m_vel.size(0)[0] - 1 == m_surface.size()[0] &&
				m_vel.size(0)[1] == m_surface.size()[1] &&
				m_vel.size(1)[0] == m_surface.size()[0] &&
				m_vel.size(1)[1] - 1 == m_surface.size()[1]);

			assert(m_collision_vel.size(0)[0] - 1 == m_surface.size()[0] &&
				m_collision_vel.size(0)[1] == m_surface.size()[1] &&
				m_collision_vel.size(1)[0] == m_surface.size()[0] &&
				m_collision_vel.size(1)[1] - 1 == m_surface.size()[1]);
		}

	void set_viscosity(Real mu);
	void set_viscosity(const ScalarGrid<Real>& mu);

	void solve(const VectorGrid<Real>& face_volumes,
				ScalarGrid<Real>& center_volumes,
				ScalarGrid<Real>& node_volumes,
				const ScalarGrid<Real>& collision_center_volumes,
				const ScalarGrid<Real>& collision_node_volumes);
private:

	VectorGrid<Real>& m_vel;
	const VectorGrid<Real>& m_collision_vel;
	
	const LevelSet2D& m_surface;
	const LevelSet2D& m_collision;

	ScalarGrid<Real> m_viscosity;

	Real m_dt;
};