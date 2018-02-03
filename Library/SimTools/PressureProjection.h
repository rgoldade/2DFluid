#pragma once

#include "Core.h"

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "LevelSet2D.h"

#include "Renderer.h"

#include "FluidParticles.h"

///////////////////////////////////
//
// PressureProjection.h/cpp
// Ryan Goldade 2017
//
// Variational pressure solve. Allows
// for moving solids.
//
////////////////////////////////////

class PressureProjection
{
	// Store the i, j coordinates and staggered face axis
	typedef Vec3i BubbleFaceIndex;
	// Store just the i, j coordinates
	typedef Vec2i FluidCellIndex;

public:
	// For variational solve, surface should be extrapolated into the collision volume
	PressureProjection(Real dt, const VectorGrid<Real>& vel, const LevelSet2D& surface)
		: m_dt(dt)
		, m_vel(vel)
		, m_surface(surface)
		, m_collision(NULL)
		, m_sp_set(false)
		, m_colvel_set(false)
		, m_divergence(NULL)
		{
			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(vel.size(0)[0] - 1 == surface.size()[0] &&
				vel.size(0)[1] == surface.size()[1] &&
				vel.size(1)[0] == surface.size()[0] &&
				vel.size(1)[1] - 1 == surface.size()[1]);

			m_pressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
			m_valid = VectorGrid<Real>(surface.xform(), surface.size(), 0, VectorGridSettings::STAGGERED);
		}
	
	// TODO: add a check here that makes sure that it matches with samples, etc. too
	void set_collision_velocity(const VectorGrid<Real>& collision_vel)
	{
		assert(collision_vel.size(0) == m_vel.size(0) &&
				collision_vel.size(1) == m_vel.size(1));
		m_collision_vel = collision_vel;
		m_colvel_set = true;
	}
	// TODO: figure out a better way than copying stuff
	void set_surface_pressure(const ScalarGrid<Real>& surface_pressure, Real sp_scale = 1.)
	{
		assert(surface_pressure.size() == m_surface.size());
		m_surface_pressure = surface_pressure;
		m_sp_scale = sp_scale;
		m_sp_set = true;
	}
	void set_goal_divergence(const ScalarGrid<Real>& divergence)
	{
		assert(divergence.size() == m_surface.size());
		m_divergence = &divergence;
	}

	inline bool is_valid(size_t x, size_t y, size_t axis) const
	{
		assert(x < m_valid.size(axis)[0] && y < m_valid.size(axis)[1]);
		return m_valid(x, y, axis) > 0;
	}

	// The liquid weights refer to the volume of liquid in each cell. This is useful for ghost fluid.
	// Note that the surface should be extrapolated into the collision volume by 1 voxel before computing the
	// weights. 
	// The fluid weights refer to the cut-cell length of fluid (air and liquid) through a cell face.
	// In both cases, 0 means "empty" and 1 means "full".
	void project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights, const ScalarGrid<Real>& center_weights, Renderer& renderer);

	// Apply solution to a velocity field at solvable faces
	void apply_solution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights);
	void apply_valid(VectorGrid<Real> &valid);

	void draw_pressure(Renderer& renderer) const;
	void draw_divergence(Renderer& renderer) const;

private:

	Real m_prev_dt;

	const VectorGrid<Real>& m_vel;

	bool m_colvel_set;
	VectorGrid<Real> m_collision_vel;
	VectorGrid<Real> m_valid; // Store solved faces

	const LevelSet2D& m_surface;
	const LevelSet2D* m_collision;
	
	ScalarGrid<Real> m_pressure;
	
	ScalarGrid<Real>	m_surface_pressure;
	const ScalarGrid<Real> *m_divergence;

	Real m_dt;
	bool m_sp_set;
	Real m_sp_scale;
};
