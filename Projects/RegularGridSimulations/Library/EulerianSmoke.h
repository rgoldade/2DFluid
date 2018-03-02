#pragma once
#include <limits>
#include <vector>

#include "Core.h"
#include "Vec.h"

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
	EulerianSmoke(const Transform& xform, Vec2st nx, Real ambienttemp = 300)
		: m_xform(xform), m_ambienttemp(ambienttemp)
	{
		m_vel = VectorGrid<Real>(m_xform, nx, VectorGridSettings::STAGGERED);
		m_collision_vel = VectorGrid<Real>(m_xform, nx, 0., VectorGridSettings::STAGGERED);

		m_collision = LevelSet2D(m_xform, nx, 5);

		m_smokedensity = ScalarGrid<Real>(m_xform, nx, 0);
		m_smoketemperature = ScalarGrid<Real>(m_xform, nx, m_ambienttemp);
	}

	void set_collision_volume(const LevelSet2D& collision);
	void set_collision_velocity(const VectorGrid<Real>& collision_vel);

	void set_smoke_velocity(const VectorGrid<Real>& vel);

	void set_smoke_source(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature);

	void advect_smoke(Real dt, IntegratorSettings::Integrator order);
	void advect_velocity(Real dt, IntegratorSettings::Integrator order);

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
	ScalarGrid<Real> m_smokedensity, m_smoketemperature;

	Real m_ambienttemp;

	Transform m_xform;
};