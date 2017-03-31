#pragma once

#include <limits>
#include <vector>

#include "core.h"
#include "vec.h"

#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "LevelSet2d.h"
#include "Transform.h"

#include "Integrator.h"

#include "AdvectField.h"
#include "ExtrapolateField.h"

#include "MarkerParticles.h"
#include "FlipParticles.h"

class FlipSimulation
{
public:
	FlipSimulation(const Transform& xform, Vec2st nx, size_t nb = 5)
		: m_xform(xform)
		, m_moving_solids(false)
		, m_solve_viscosity(false)
		, m_st_scale(0.)
	{
		m_vel = VectorGrid<Real>(m_xform, nx, VectorGridSettings::STAGGERED);
		m_collision_vel = VectorGrid<Real>(m_xform, nx, 0., VectorGridSettings::STAGGERED);

		m_surface = LevelSet2D(m_xform, nx, nb);
		m_collision = LevelSet2D(m_xform, nx, nb);
	}

	void set_collision_volume(const LevelSet2D& collision);
	void set_collision_velocity(const VectorGrid<Real>& collision_vel);
	void set_surface_volume(const LevelSet2D& surface);
	void set_surface_velocity(const VectorGrid<Real>& vel);

	void set_surface_tension(Real st_scale)
	{
		m_st_scale = st_scale;
	}

	void add_surface_volume(const LevelSet2D& surface);

	template<typename ForceSampler>
	void add_force(const ForceSampler& force, Real dt);

	void add_force(const Vec2R& force, Real dt);

	// Move fluid particles
	void advect_surface(Real dt, IntegratorSettings::Integrator order);
	// TODO: move viscosity with particles so this is not needed
	void advect_viscosity(Real dt, IntegratorSettings::Integrator order);
	// TODO: since particles carry velocity, this is not needed either
	void advect_velocity(Real dt, IntegratorSettings::Integrator order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void run_simulation(Real dt, Renderer& renderer);

	// Useful for CFL
	Real max_vel_mag() { return mag(m_particles.max_vel()); }

	// Rendering tools
	void draw_grid(Renderer& renderer) const;
	void draw_surface(Renderer& renderer);
	void draw_collision(Renderer& renderer);
	void draw_velocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> m_vel, m_collision_vel;
	LevelSet2D m_surface, m_collision;
	ScalarGrid<Real> m_variableviscosity;

	Transform m_xform;

	bool m_moving_solids, m_solve_viscosity;
	Real m_st_scale;

	FlipParticles m_particles;

};