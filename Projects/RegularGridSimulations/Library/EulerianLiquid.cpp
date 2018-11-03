#include <iostream>

#include "EulerianLiquid.h"
#include "PressureProjection.h"
#include "ViscositySolver.h"

#include "ExtrapolateField.h"
#include "ComputeWeights.h"

#include "Timer.h"

void EulerianLiquid::draw_grid(Renderer& renderer) const
{
	m_surface.draw_grid(renderer);
}

void EulerianLiquid::draw_surface(Renderer& renderer)
{
	m_surface.draw_surface(renderer, Vec3f(0., 0., 1.0));
}

void EulerianLiquid::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer, Vec3f(1.,0.,1.));
}

void EulerianLiquid::draw_collision_vel(Renderer& renderer, Real length) const
{
	m_collision_vel.draw_sample_point_vectors(renderer, Vec3f(0,1,0), m_collision_vel.dx() * length);
}

void EulerianLiquid::draw_velocity(Renderer& renderer, Real length) const
{
	m_vel.draw_sample_point_vectors(renderer, Vec3f(0), m_vel.dx() * length);
}

// Incoming collision volume must already be inverted
void EulerianLiquid::set_collision_volume(const LevelSet2D& collision)
{
	assert(collision.inverted());

	Mesh2D temp_mesh;
	collision.extract_mesh(temp_mesh);
	
	// TODO : consider making collision node sampled
	m_collision.set_inverted();
	m_collision.init(temp_mesh, false);
}

void EulerianLiquid::set_surface_volume(const LevelSet2D& fluid)
{
	Mesh2D temp_mesh;
	fluid.extract_mesh(temp_mesh);
	
	m_surface.init(temp_mesh, false);
}

void EulerianLiquid::set_surface_velocity(const VectorGrid<Real>& vel)
{
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R pos = m_vel.idx_to_ws(Vec2R(face), axis);
			m_vel(face, axis) = vel.interp(pos, axis);
		});
	}
}

void EulerianLiquid::set_collision_velocity(const VectorGrid<Real>& collision_vel)
{
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_collision_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R wpos = m_collision_vel.idx_to_ws(Vec2R(face), axis);
			m_collision_vel(face, axis) = collision_vel.interp(wpos, axis);
		});
	}
}

void EulerianLiquid::add_surface_volume(const LevelSet2D& surface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R pos = m_vel.idx_to_ws(Vec2R(face), axis);
			if (surface.interp(pos) <= 0. && m_surface.interp(pos) > 0.)
				m_vel(face, axis) = 0;
		});
	}

	// Combine surfaces
	m_surface.surface_union(surface);
	m_surface.reinitFIM();
}

template<typename ForceSampler>
void EulerianLiquid::add_force(Real dt, const ForceSampler& force)
{
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R pos = m_vel.idx_to_ws(Vec2R(face), axis);
			m_vel(face, axis) = m_vel(face, axis) + dt * force(pos, axis);
		});
	}
}

void EulerianLiquid::add_force(Real dt, const Vec2R& force)
{
	add_force(dt, [&](Vec2R, unsigned axis) {return force[axis]; });
}

void EulerianLiquid::advect_surface(Real dt, IntegrationOrder integrator)
{
	auto vel_func = [&](Real, const Vec2R& pos) { return m_vel.interp(pos);  };
	m_surface.advect(dt, vel_func, integrator);
	m_surface.reinitFIM();
}

void EulerianLiquid::advect_viscosity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto vel_func = [&](Real, const Vec2R& pos) { return m_vel.interp(pos); };

	AdvectField<ScalarGrid<Real>> advector(m_variable_viscosity);
	ScalarGrid<Real> temp_viscosity(m_variable_viscosity.xform(), m_variable_viscosity.size());
	
	advector.advect_field(dt, temp_viscosity, vel_func, integrator, interpolator);
	std::swap(temp_viscosity, m_variable_viscosity);
}

void EulerianLiquid::advect_velocity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto vel_func = [&](Real, const Vec2R& pos) { return m_vel.interp(pos); };

	VectorGrid<Real> temp_vel(m_vel.xform(), m_vel.grid_size(), VectorGridSettings::SampleType::STAGGERED);

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		AdvectField<ScalarGrid<Real>> advector(m_vel.grid(axis));

		advector.advect_field(dt, temp_vel.grid(axis), vel_func, integrator, interpolator);
	}

	std::swap(m_vel, temp_vel);
}

void EulerianLiquid::run_simulation(Real dt, Renderer& renderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer sim_timer;

	ComputeWeights pressureweightcomputer(m_surface, m_collision);

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> liquid_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::SampleType::STAGGERED);
	pressureweightcomputer.compute_gf_weights(liquid_weights);

	VectorGrid<Real> cc_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::SampleType::STAGGERED);
	pressureweightcomputer.compute_cutcell_weights(cc_weights);

	std::cout << "  Compute weights: " << sim_timer.stop() << "s" << std::endl;
	
	sim_timer.reset();

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, m_surface, m_vel, m_collision, m_collision_vel);

	//if (m_surfacetension_scale != 0.)
	//{
	//	const ScalarGrid<Real> surface_tension = m_surface.get_curvature();
	//	projectdivergence.set_surface_pressure(surface_tension, m_st_scale);
	//}

	VectorGrid<Real> valid(m_surface.xform(), m_surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	projectdivergence.project(liquid_weights, cc_weights);
	
	// Update velocity field
	projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

	// Solve for viscosity if set
	if (m_solve_viscosity)
	{
		std::cout << "  Solve for pressure: " << sim_timer.stop() << "s" << std::endl;
		sim_timer.reset();


		unsigned samples = 3;
		ComputeWeights viscosityvolumecomputer(m_surface, m_collision);
		// Compute supersampled volumes
		VectorGrid<Real> face_vol_weights(m_surface.xform(), m_surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
		viscosityvolumecomputer.compute_supersampled_volumes(face_vol_weights.grid(0), samples);
		viscosityvolumecomputer.compute_supersampled_volumes(face_vol_weights.grid(1), samples);

		ScalarGrid<Real> center_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::SampleType::CENTER);
		viscosityvolumecomputer.compute_supersampled_volumes(center_vol_weights, samples);

		ScalarGrid<Real> node_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::SampleType::NODE);
		viscosityvolumecomputer.compute_supersampled_volumes(node_vol_weights, samples);

		ScalarGrid<Real> collision_center_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::SampleType::CENTER);
		viscosityvolumecomputer.compute_supersampled_volumes(collision_center_vol_weights, samples, true);

		ScalarGrid<Real> collision_node_vol_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::SampleType::NODE);
		viscosityvolumecomputer.compute_supersampled_volumes(collision_node_vol_weights, samples, true);

		std::cout << "  Compute viscosity weights: " << sim_timer.stop() << "s" << std::endl;
		sim_timer.reset();

		ViscositySolver viscosity(dt, m_surface, m_vel, m_collision, m_collision_vel);

		viscosity.set_viscosity(m_variable_viscosity);

		viscosity.solve(face_vol_weights, center_vol_weights, node_vol_weights, collision_center_vol_weights, collision_node_vol_weights);

		std::cout << "  Solve for viscosity: " << sim_timer.stop() << "s" << std::endl;
		sim_timer.reset();

		// Initialize and call pressure projection		
		PressureProjection projectdivergence2(dt, m_surface, m_vel, m_collision, m_collision_vel);

		projectdivergence.project(liquid_weights, cc_weights);

		// Update velocity field
		projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

		projectdivergence.apply_valid(valid);

		std::cout << "  Solve for pressure after viscosity: " << sim_timer.stop() << "s" << std::endl;
		
		sim_timer.reset();
	}
	else
	{
		// Label solved faces
		projectdivergence.apply_valid(valid);

		std::cout << "  Solve for pressure: " << sim_timer.stop() << "s" << std::endl;
		sim_timer.reset();
	}


	// Extrapolate velocity
	ExtrapolateField<VectorGrid<Real>> extrapolator(m_vel);
	extrapolator.extrapolate(valid, 10);

	// Enforce collision velocity
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			if (cc_weights(face, axis) == 0.)
			{
				m_vel(face, axis) = m_collision_vel(face, axis);
			}
		});
	}

	std::cout << "  Extrapolate velocity: " << sim_timer.stop() << "s" << std::endl;
	sim_timer.reset();

	advect_surface(dt, IntegrationOrder::FORWARDEULER);
	advect_velocity(dt, IntegrationOrder::RK3);

	if(m_solve_viscosity)
		advect_viscosity(dt, IntegrationOrder::FORWARDEULER);

	std::cout << "  Advect simulation: " << sim_timer.stop() << "s" << std::endl;
}
