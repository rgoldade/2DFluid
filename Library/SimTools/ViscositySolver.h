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
public:

	ViscositySolver(Real dt, VectorGrid<Real>& vel, const LevelSet2D& surface, const LevelSet2D& collision)
		: m_dt(dt)
		, m_vel(vel)
		, m_surface(surface)
		, m_collision(collision)
		, m_colvel_set(false)
		, m_colweight_set(false)
		{
			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(vel.size(0)[0] - 1 == surface.size()[0] &&
					vel.size(0)[1] == surface.size()[1] &&
					vel.size(1)[0] == surface.size()[0] &&
					vel.size(1)[1] - 1 == surface.size()[1]);

			assert(vel.size(0)[0] - 1 == collision.size()[0] &&
					vel.size(0)[1] == collision.size()[1] &&
					vel.size(1)[0] == collision.size()[0] &&
					vel.size(1)[1] - 1 == collision.size()[1]);
		}

	void set_collision_velocity(const VectorGrid<Real>& collision_vel)
	{
		assert(collision_vel.size(0) == m_vel.size(0) &&
			collision_vel.size(1) == m_vel.size(1));
		m_collision_vel = collision_vel;
		m_colvel_set = true;
	}

	void set_collision_weights(const ScalarGrid<Real>& collision_center_volumes,
								const ScalarGrid<Real>& collision_node_volumes)
	{
		assert(collision_center_volumes.sample_type() == ScalarGridSettings::CENTER &&
				collision_center_volumes.size() == m_surface.size() &&
				collision_center_volumes.xform() == m_surface.xform() &&
				collision_node_volumes.sample_type() == ScalarGridSettings::NODE &&
				collision_node_volumes.size() - Vec2st(1) == m_surface.size() &&
				collision_node_volumes.xform() == m_surface.xform());

		m_colweight_set = true;

		m_col_center_vol = &collision_center_volumes;
		m_col_node_vol = &collision_node_volumes;
	}

	void set_viscosity(Real mu);
	void set_viscosity(const ScalarGrid<Real>& mu);

	// TODO: add weights for collision
	void solve(const VectorGrid<Real>& face_volumes,
				ScalarGrid<Real> center_volumes,
				ScalarGrid<Real> node_volumes,
				Renderer& renderer);
private:

	VectorGrid<Real>& m_vel;
	
	bool m_colvel_set, m_colweight_set;
	VectorGrid<Real> m_collision_vel;

	const LevelSet2D& m_surface;
	const LevelSet2D& m_collision;

	const ScalarGrid<Real> *m_col_node_vol, *m_col_center_vol;

	ScalarGrid<Real> m_viscosity;

	Real m_dt;
};