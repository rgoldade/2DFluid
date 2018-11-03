#pragma once
#include <limits>
#include <vector>

#include "Common.h"

#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "LevelSet2D.h"
#include "Transform.h"

#include "Integrator.h"

#include "AdvectField.h"
#include "ExtrapolateField.h"

///////////////////////////////////
//
// EulerianSmoke.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid smoke simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

class EulerianSmoke
{
public:
	EulerianSmoke(const Transform& xform, Vec2ui size, Real ambienttemp = 300)
		: m_xform(xform), m_ambient_temperature(ambienttemp)
	{
		m_vel = VectorGrid<Real>(m_xform, size, VectorGridSettings::SampleType::STAGGERED);
		m_collision_vel = VectorGrid<Real>(m_xform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		m_collision = LevelSet2D(m_xform, size, 5);

		m_smoke_density = ScalarGrid<Real>(m_xform, size, 0);
		m_smoke_temperature = ScalarGrid<Real>(m_xform, size, m_ambient_temperature);
	}

	void set_collision_volume(const LevelSet2D& collision);
	void set_collision_velocity(const VectorGrid<Real>& collision_vel);

	void set_smoke_velocity(const VectorGrid<Real>& vel);

	void set_smoke_source(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature);

	void advect_smoke(Real dt, const InterpolationOrder& order);
	void advect_velocity(Real dt, const InterpolationOrder& order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void run_simulation(Real dt, Renderer& renderer);

	// Useful for CFL
	Real max_vel_mag() { return m_vel.max_magnitude(); }

	// Rendering tools
	void draw_grid(Renderer& renderer) const;
	void draw_smoke(Renderer& renderer, Real maxdensity);
	void draw_collision(Renderer& renderer);
	void draw_collision_vel(Renderer& renderer, Real length) const;
	void draw_velocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> m_vel, m_collision_vel;
	LevelSet2D m_collision;
	ScalarGrid<Real> m_smoke_density, m_smoke_temperature;

	Real m_ambient_temperature;

	Transform m_xform;
};