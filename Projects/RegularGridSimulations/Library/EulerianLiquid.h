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
// EulerianLiquid.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid liquid simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

class EulerianLiquid
{
public:
	EulerianLiquid(const Transform& xform, Vec2ui size, Real narrow_band = 5.)
		: m_xform(xform)
		, m_solve_viscosity(false)
		, m_surfacetension_scale(0.)
	{
		m_vel = VectorGrid<Real>(m_xform, size, VectorGridSettings::SampleType::STAGGERED);
		m_collision_vel = VectorGrid<Real>(m_xform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		m_surface = LevelSet2D(m_xform, size, narrow_band);
		m_collision = LevelSet2D(m_xform, size, narrow_band);
	}

	void set_collision_volume(const LevelSet2D& collision);
	void set_collision_velocity(const VectorGrid<Real>& collision_vel);
	void set_surface_volume(const LevelSet2D& surface);
	void set_surface_velocity(const VectorGrid<Real>& vel);
	void set_surface_tension(Real st_scale)
	{
		m_surfacetension_scale = st_scale;
	}

	void set_viscosity(const ScalarGrid<Real>& visc_coeff)
	{
		assert(m_surface.is_matched(visc_coeff));
		m_variable_viscosity = visc_coeff;
		m_solve_viscosity = true;
	}

	void set_viscosity(Real visc_coeff = 1.)
	{
		m_variable_viscosity = ScalarGrid<Real>(m_surface.xform(), m_surface.size(), visc_coeff);
		m_solve_viscosity = true;
	}

	void add_surface_volume(const LevelSet2D& surface);

	template<typename ForceSampler>
	void add_force(Real dt, const ForceSampler& force);
	
	void add_force(Real dt, const Vec2R& force);

	void advect_surface(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advect_viscosity(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER, InterpolationOrder interpolator = InterpolationOrder::LINEAR);
	void advect_velocity(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void run_simulation(Real dt, Renderer& renderer);

	// Useful for CFL
	Real max_vel_mag() { return m_vel.max_magnitude(); }
	
	// Rendering tools
	void draw_grid(Renderer& renderer) const;
	void draw_surface(Renderer& renderer);
	void draw_collision(Renderer& renderer);
	void draw_collision_vel(Renderer& renderer, Real length) const;
	void draw_velocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> m_vel, m_collision_vel;
	LevelSet2D m_surface, m_collision;
	ScalarGrid<Real> m_variable_viscosity;

	Transform m_xform;

	bool m_solve_viscosity;
	Real m_surfacetension_scale;
};