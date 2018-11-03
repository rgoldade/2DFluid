#include <iostream>

#include "EulerianSmoke.h"
#include "PressureProjection.h"
#include "ViscositySolver.h"

#include "ExtrapolateField.h"
#include "ComputeWeights.h"

#include "Timer.h"

void EulerianSmoke::draw_grid(Renderer& renderer) const
{
	m_collision.draw_grid(renderer);
}

void EulerianSmoke::draw_smoke(Renderer& renderer, Real max_density)
{
	m_smoke_density.draw_volumetric(renderer, Vec3f(1), Vec3f(0), 0, max_density);
}

void EulerianSmoke::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer, Vec3f(1., 0., 1.));
}

void EulerianSmoke::draw_collision_vel(Renderer& renderer, Real length) const
{
	m_collision_vel.draw_sample_point_vectors(renderer, Vec3f(0, 1, 0), m_collision_vel.dx() * length);
}

void EulerianSmoke::draw_velocity(Renderer& renderer, Real length) const
{
	m_vel.draw_sample_point_vectors(renderer, Vec3f(0), m_vel.dx() * length);
}

// Incoming collision volume must already be inverted
void EulerianSmoke::set_collision_volume(const LevelSet2D& collision)
{
	assert(collision.is_matched(m_collision));

	assert(collision.inverted());

	Mesh2D temp_mesh;
	collision.extract_mesh(temp_mesh);

	// TODO : consider making collision node sampled
	m_collision.set_inverted();
	m_collision.init(temp_mesh, false);
}

void EulerianSmoke::set_collision_velocity(const VectorGrid<Real>& collision_vel)
{
	assert(collision_vel.is_matched(m_collision_vel));
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_collision_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R wpos = m_collision_vel.idx_to_ws(Vec2R(cell), axis);
			m_collision_vel(cell, axis) =  collision_vel.interp(wpos, axis);
		});
	}
}

void EulerianSmoke::set_smoke_velocity(const VectorGrid<Real>& vel)
{
	assert(vel.is_matched(m_vel));

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R pos = m_vel.idx_to_ws(Vec2R(cell), axis);
			m_vel(cell, axis) = vel.interp(pos, axis);
		});
	}
}

void EulerianSmoke::set_smoke_source(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature)
{
	assert(density.is_matched(m_smoke_density));
	assert(temperature.is_matched(m_smoke_temperature));

	Vec2ui size = m_smoke_density.size();

	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		if (density(cell) > 0)
		{
			Real temp = density(cell);
			temp = temperature(cell);
			m_smoke_density(cell) = density(cell);
			m_smoke_temperature(cell) = temperature(cell);
		}
	});
}

void EulerianSmoke::advect_smoke(Real dt, const InterpolationOrder& order)
{
	auto vel_func = [&](Real, const Vec2R& pos) { return m_vel.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> density_advector(m_smoke_density);

		ScalarGrid<Real> temp_density(m_smoke_density.xform(), m_smoke_density.size());
		// TODO: add some sort of collision management
		density_advector.advect_field(dt, temp_density, vel_func, IntegrationOrder::FORWARDEULER, order);
		
		std::swap(m_smoke_density, temp_density);
	}

	{
		AdvectField<ScalarGrid<Real>> temperature_advector(m_smoke_temperature);

		ScalarGrid<Real> temp_temperature(m_smoke_temperature.xform(), m_smoke_temperature.size());

		temperature_advector.advect_field(dt, temp_temperature, vel_func, IntegrationOrder::FORWARDEULER, order);

		std::swap(m_smoke_temperature, temp_temperature);
	}
}

void EulerianSmoke::advect_velocity(Real dt, const InterpolationOrder& order)
{
	auto vel_func = [&](Real, const Vec2R& pos) { return m_vel.interp(pos);  };
	
	VectorGrid<Real> temp_vel(m_vel.xform(), m_vel.grid_size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		AdvectField<ScalarGrid<Real>> advector(m_vel.grid(axis));

		advector.advect_field(dt, temp_vel.grid(axis), vel_func, IntegrationOrder::RK3, order);
	}

	std::swap(m_vel, temp_vel);
}

void EulerianSmoke::run_simulation(Real dt, Renderer& renderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer add_forces;

	// Add bouyancy forces
	Real alpha = 1;
	Real beta = 1;

	for_each_voxel_range(Vec2ui(0), m_vel.size(1), [&](const Vec2ui& face)
	{
		// Average density and temperature values at velocity face
		Vec2R pos = m_vel.idx_to_ws(Vec2R(face), 1);
		Real density = m_smoke_density.interp(pos);
		Real temperature = m_smoke_temperature.interp(pos);

		Real force = dt * (-alpha * density + beta * (temperature - m_ambient_temperature));
		m_vel(face, 1) += force;

	});
	
	std::cout << "Add forces: " << add_forces.stop() << "s" << std::endl;

	Timer compute_weights;

	LevelSet2D dummysurface(m_collision.xform(), m_collision.size(), 5, false);
		
	ComputeWeights pressureweightcomputer(dummysurface, m_collision);

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> liquid_weights(m_collision.xform(), m_collision.size(), 1., VectorGridSettings::SampleType::STAGGERED);

	VectorGrid<Real> cc_weights(m_collision.xform(), m_collision.size(), VectorGridSettings::SampleType::STAGGERED);
	pressureweightcomputer.compute_cutcell_weights(cc_weights);

	std::cout << "Compute weights: " << compute_weights.stop() << "s" << std::endl;

	Timer pressure_projection;

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, dummysurface, m_vel, m_collision, m_collision_vel);
	// TODO: handle moving boundaries.

	VectorGrid<Real> valid(m_collision.xform(), m_collision.size(), 0, VectorGridSettings::SampleType::STAGGERED);
	
	projectdivergence.project(liquid_weights, cc_weights);

	// Update velocity field
	projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

	projectdivergence.apply_valid(valid);

	std::cout << "Pressure projection: " << pressure_projection.stop() << "s" << std::endl;

	Timer fix_velocity;
	//// Extrapolate velocity
	//ExtrapolateField<VectorGrid<Real>> extrapolator(m_vel);
	//extrapolator.extrapolate(valid, 10);

	// Enforce collision velocity
	for (size_t axis = 0; axis < 2; ++axis)
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

	std::cout << "Clean up velocity: " << fix_velocity.stop() << "s" << std::endl;

	Timer advection;

	advect_smoke(dt, InterpolationOrder::CUBIC);
	advect_velocity(dt, InterpolationOrder::LINEAR);

	std::cout << "Advection: " << advection.stop() << "s" << std::endl;
}