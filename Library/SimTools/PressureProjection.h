#pragma once

#include "Common.h"

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "LevelSet2D.h"

#include "Renderer.h"

///////////////////////////////////
//
// PressureProjection.h/cpp
// Ryan Goldade 2017
//
// Variational pressure solve. Allows
// for moving solids.
//
////////////////////////////////////

static constexpr int UNSOLVED = -1;
static constexpr Real MINTHETA = 0.01;

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the collision volume
	PressureProjection(Real dt, const LevelSet2D& surface, const VectorGrid<Real>& vel,
							const LevelSet2D& collision, const VectorGrid<Real>& collision_vel)
		: m_dt(dt)
		, m_surface(surface)
		, m_vel(vel)
		, m_collision(collision)
		, m_collision_vel(collision_vel)
		{
			assert(surface.is_matched(collision));

			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(vel.size(0)[0] - 1 == surface.size()[0] &&
				vel.size(0)[1] == surface.size()[1] &&
				vel.size(1)[0] == surface.size()[0] &&
				vel.size(1)[1] - 1 == surface.size()[1]);

			assert(collision_vel.size(0)[0] - 1 == surface.size()[0] &&
				collision_vel.size(0)[1] == surface.size()[1] &&
				collision_vel.size(1)[0] == surface.size()[0] &&
				collision_vel.size(1)[1] - 1 == surface.size()[1]);

			m_pressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
			m_valid = VectorGrid<Real>(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
		}
	
	// The liquid weights refer to the volume of liquid in each cell. This is useful for ghost fluid.
	// Note that the surface should be extrapolated into the collision volume by 1 voxel before computing the
	// weights. 
	// The fluid weights refer to the cut-cell length of fluid (air and liquid) through a cell face.
	// In both cases, 0 means "empty" and 1 means "full".
	void project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights);

	// Apply solution to a velocity field at solvable faces
	void apply_solution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights);
	void apply_valid(VectorGrid<Real> &valid);

	void draw_pressure(Renderer& renderer) const;

private:

	const VectorGrid<Real>& m_vel, &m_collision_vel;

	VectorGrid<Real> m_valid; // Store solved faces

	const LevelSet2D& m_surface, &m_collision;
	
	ScalarGrid<Real> m_pressure;
	
	Real m_dt;
};
