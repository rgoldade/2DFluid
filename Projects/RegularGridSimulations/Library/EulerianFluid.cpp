#include "EulerianFluid.h"
#include "PressureProjection.h"
#include "ViscositySolver.h"

#include "ExtrapolateField.h"
#include "ComputeWeights.h"

void EulerianFluid::draw_grid(Renderer& renderer) const
{
	m_surface.draw_grid(renderer);
}

void EulerianFluid::draw_surface(Renderer& renderer)
{
	m_surface.draw_surface(renderer, Vec3f(0., 0., 1.0));
}

void EulerianFluid::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer, Vec3f(1.,0.,1.));
}

void EulerianFluid::draw_collision_vel(Renderer& renderer, Real length) const
{
	if (m_moving_solids)
		m_collision_vel.draw_sample_point_vectors(renderer, Vec3f(0,1,0), m_collision_vel.dx() * length);
}

void EulerianFluid::draw_velocity(Renderer& renderer, Real length) const
{
	m_vel.draw_sample_point_vectors(renderer, Vec3f(0), m_vel.dx() * length);
}

// Incoming collision volume must already be inverted
void EulerianFluid::set_collision_volume(const LevelSet2D& collision)
{
	assert(collision.inverted());

	Mesh2D temp_mesh;
	collision.extract_mesh(temp_mesh);
	
	// TODO : consider making collision node sampled
	m_collision.set_inverted();
	m_collision.init(temp_mesh, false);
}

void EulerianFluid::set_surface_volume(const LevelSet2D& fluid)
{
	Mesh2D temp_mesh;
	fluid.extract_mesh(temp_mesh);
	
	m_surface.init(temp_mesh, false);
}

void EulerianFluid::set_surface_velocity(const VectorGrid<Real>& vel)
{
	for (size_t axis = 0; axis < 2; ++axis)
	{
		size_t x_size = m_vel.size(axis)[0];
		size_t y_size = m_vel.size(axis)[1];

		for (size_t x = 0; x < x_size; ++x)
			for (size_t y = 0; y < y_size; ++y)
			{
				Vec2R pos = m_vel.idx_to_ws(Vec2R(x, y), axis);
				m_vel.set(x, y, axis, vel.interp(pos, axis));
			}
	}
}

void EulerianFluid::set_collision_velocity(const VectorGrid<Real>& collision_vel)
{
	for (size_t axis = 0; axis < 2; ++axis)
	{
		size_t x_size = m_collision_vel.size(axis)[0];
		size_t y_size = m_collision_vel.size(axis)[1];

		for (size_t x = 0; x < x_size; ++x)
			for (size_t y = 0; y < y_size; ++y)
			{
				Vec2R wpos = m_collision_vel.idx_to_ws(Vec2R(x, y), axis);
				m_collision_vel.set(x, y, axis, collision_vel.interp(wpos, axis));
			}
	}

	m_moving_solids = true;
}

void EulerianFluid::add_surface_volume(const LevelSet2D& surface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (size_t axis = 0; axis < 2; ++axis)
	{
		size_t x_size = m_vel.size(axis)[0];
		size_t y_size = m_vel.size(axis)[1];

		for (size_t x = 0; x < x_size; ++x)
			for (size_t y = 0; y < y_size; ++y)
			{
				Vec2R pos = m_vel.idx_to_ws(Vec2R(x, y), axis);
				if (surface.phi(pos) <= 0. && m_surface.phi(pos) > 0.)
				{
					m_vel(x, y, axis) = 0;
				}
			}
	}

	// Combine surfaces
	m_surface.surface_union(surface);
}

template<typename ForceSampler>
void EulerianFluid::add_force(const ForceSampler& force, Real dt)
{
	for (size_t axis = 0; axis < 2; ++axis)
	{
		size_t x_size = m_vel.size(axis)[0];
		size_t y_size = m_vel.size(axis)[1];

		for (size_t x = 0; x < x_size; ++x)
			for (size_t y = 0; y < y_size; ++y)
			{
				Vec2R pos = m_vel.idx_to_ws(Vec2R(x, y), axis);
				m_vel.set(x, y, axis, m_vel(x, y, axis) + force(pos, axis) * dt);
			}
	}
}

void EulerianFluid::add_force(const Vec2R& force, Real dt)
{
	add_force([&](Vec2R, size_t axis) {return force[axis]; }, dt);
}

void EulerianFluid::advect_surface(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<LevelSet2D> advector(m_vel, m_surface);
	advector.set_collision_volumes(m_collision);
	switch (order)
	{
	case IntegratorSettings::FE:
		m_surface.backtrace_advect(dt, advector, Integrator::forward_euler<Vec2R, AdvectField<LevelSet2D>>());
		break;
	case IntegratorSettings::RK3:
		m_surface.backtrace_advect(dt, advector, Integrator::rk3<Vec2R, AdvectField<LevelSet2D>>());
		break;
	default:
		assert(false);
	}

	m_surface.reinit();
}

void EulerianFluid::advect_viscosity(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<ScalarGrid<Real>> advector(m_vel, m_variableviscosity);
	advector.set_collision_volumes(m_collision);
	ScalarGrid<Real> temp_visc = m_variableviscosity;
	advector.advect_field(dt, temp_visc, order);
	m_variableviscosity = temp_visc;
}

void EulerianFluid::advect_velocity(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<VectorGrid<Real>> advector(m_vel, m_vel);
	advector.set_collision_volumes(m_collision);
	VectorGrid<Real> temp_vel = m_vel;
	advector.advect_field(dt, temp_vel, order);
	m_vel = temp_vel;
}

void EulerianFluid::run_simulation(Real dt, Renderer& renderer)
{
	ComputeWeights pressureweightcomputer(m_surface, m_collision);

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> liquid_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);
	pressureweightcomputer.compute_gf_weights(liquid_weights);

	VectorGrid<Real> cc_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);
	pressureweightcomputer.compute_cutcell_weights(cc_weights);

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, m_vel, m_surface);

	if (m_moving_solids)
	{
		projectdivergence.set_collision_velocity(m_collision_vel);
	}

	if (m_st_scale != 0.)
	{
		const ScalarGrid<Real> surface_tension = m_surface.get_curvature();
		projectdivergence.set_surface_pressure(surface_tension, m_st_scale);
	}

	VectorGrid<Real> valid(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);

	ComputeWeights volumecomputer(m_surface, m_collision);
	ScalarGrid<Real> center_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::CENTER);
	volumecomputer.compute_supersampled_volumes(center_weights, 3);

	projectdivergence.project(liquid_weights, cc_weights, center_weights, renderer);
	
	// Update velocity field
	projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

	projectdivergence.apply_valid(valid);

	// Solve for viscosity if set
	if (m_solve_viscosity)
	{
		size_t samples = 2;
		ComputeWeights viscosityvolumecomputer(m_surface, m_collision);
		// Compute supersampled volumes
		VectorGrid<Real> face_vol_weights(m_surface.xform(), m_surface.size(), 0, VectorGridSettings::STAGGERED);
		viscosityvolumecomputer.compute_supersampled_volumes(face_vol_weights.grid(0), samples);
		viscosityvolumecomputer.compute_supersampled_volumes(face_vol_weights.grid(1), samples);

		ScalarGrid<Real> center_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::CENTER);
		viscosityvolumecomputer.compute_supersampled_volumes(center_vol_weights, samples);

		ScalarGrid<Real> node_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::NODE);
		viscosityvolumecomputer.compute_supersampled_volumes(node_vol_weights, samples);

		ScalarGrid<Real> collision_center_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::CENTER);
		viscosityvolumecomputer.compute_supersampled_volumes(collision_center_vol_weights, samples, true);

		ScalarGrid<Real> collision_node_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::NODE);
		viscosityvolumecomputer.compute_supersampled_volumes(collision_node_vol_weights, samples, true);

		ViscositySolver viscosity(dt, m_vel, m_surface, m_collision);

		viscosity.set_viscosity(m_variableviscosity);
		viscosity.set_collision_weights(collision_center_vol_weights, collision_node_vol_weights);
		
		if (m_moving_solids)
		{
			viscosity.set_collision_velocity(m_collision_vel);
		}

		viscosity.solve(face_vol_weights, center_vol_weights, node_vol_weights, renderer);

		// Initialize and call pressure projection		
		PressureProjection projectdivergence2(dt, m_vel, m_surface);

		if (m_moving_solids)
		{
			projectdivergence2.set_collision_velocity(m_collision_vel);
		}

		projectdivergence.project(liquid_weights, cc_weights, center_weights, renderer);

		// Update velocity field
		projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

		projectdivergence.apply_valid(valid);

	}
	else
	{
		// Label solved faces
		projectdivergence.apply_valid(valid);
	}

	// Extrapolate velocity
	ExtrapolateField<VectorGrid<Real>> extrapolator(m_vel);
	extrapolator.extrapolate(valid, 10);

	// Enforce collision velocity
	for (size_t axis = 0; axis < 2; ++axis)
	{
		for (size_t x = 0; x < m_vel.size(axis)[0]; ++x)
			for (size_t y = 0; y < m_vel.size(axis)[1]; ++y)
			{
				if (cc_weights(x, y, axis) == 0.)
				{
					m_vel(x, y, axis) = m_collision_vel(x, y, axis);
				}
			}
	}

	advect_surface(dt, IntegratorSettings::RK3);
	advect_velocity(dt, IntegratorSettings::RK3);
	
	if(m_solve_viscosity)
		advect_viscosity(dt, IntegratorSettings::RK3);
}
