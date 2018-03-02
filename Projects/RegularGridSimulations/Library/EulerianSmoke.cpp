#include "EulerianSmoke.h"
#include "PressureProjection.h"
#include "ViscositySolver.h"

#include "ExtrapolateField.h"
#include "ComputeWeights.h"

void EulerianSmoke::draw_grid(Renderer& renderer) const
{
	m_collision.draw_grid(renderer);
}

void EulerianSmoke::draw_smoke(Renderer& renderer, Real max_density)
{
	m_smokedensity.draw_volumetric(renderer, Vec3f(1), Vec3f(0), 0, max_density);
}

void EulerianSmoke::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer, Vec3f(1., 0., 1.));
}

void EulerianSmoke::draw_collision_vel(Renderer& renderer, Real length) const
{
	//if (m_moving_solids)
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

	//m_moving_solids = true;
}

void EulerianSmoke::set_smoke_velocity(const VectorGrid<Real>& vel)
{
	assert(vel.is_matched(m_vel));

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

void EulerianSmoke::set_smoke_source(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature)
{
	assert(density.is_matched(m_smokedensity));
	assert(temperature.is_matched(m_smoketemperature));

	Vec2st size = m_smokedensity.size();

	for (int i = 0; i < size[0]; ++i)
		for (int j = 0; j < size[1]; ++j)
		{
			if (density(i, j) > 0)
			{
				m_smokedensity(i, j) = density(i, j);
				m_smoketemperature(i, j) = temperature(i, j);
			}
		}
}

void EulerianSmoke::advect_smoke(Real dt, IntegratorSettings::Integrator order)
{
	ScalarGrid<Real> tempdensity(m_smokedensity.xform(), m_smokedensity.size());
	AdvectField<ScalarGrid<Real>> densityadvector(m_vel, m_smokedensity);
	densityadvector.set_collision_volumes(m_collision);
	
	densityadvector.advect_field(dt, tempdensity, order);
	m_smokedensity = tempdensity;

	ScalarGrid<Real> temptemp(m_smoketemperature.xform(), m_smoketemperature.size());
	AdvectField<ScalarGrid<Real>> temperatureadvector(m_vel, m_smoketemperature);
	temperatureadvector.set_collision_volumes(m_collision);

	temperatureadvector.advect_field(dt, temptemp, order);
	m_smoketemperature = temptemp;
}

void EulerianSmoke::advect_velocity(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<VectorGrid<Real>> advector(m_vel, m_vel);
	advector.set_collision_volumes(m_collision);
	VectorGrid<Real> temp_vel = m_vel;
	advector.advect_field(dt, temp_vel, order);
	m_vel = temp_vel;
}

void EulerianSmoke::run_simulation(Real dt, Renderer& renderer)
{
	// Add bouyancy forces
	Real alpha = 1;
	Real beta = 1;
	for (int i = 0; i < m_vel.size(1)[0]; ++i)
		for (int j = 0; j < m_vel.size(1)[1]; ++j)
		{
			// Average density and temperature values at velocity face
			Vec2R pos = m_vel.idx_to_ws(Vec2R(i, j), 1);
			Real density = m_smokedensity.interp(pos);
			Real temperature = m_smoketemperature.interp(pos);

			m_vel(i, j, 1) += dt * (-alpha * density + beta * (temperature - m_ambienttemp));
		}
	
	LevelSet2D dummysurface(m_collision.xform(), m_collision.size(), 5, false, -5);

	
	ComputeWeights pressureweightcomputer(dummysurface, m_collision);

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> liquid_weights(m_collision.xform(), m_collision.size(), 1., VectorGridSettings::STAGGERED);

	VectorGrid<Real> cc_weights(m_collision.xform(), m_collision.size(), VectorGridSettings::STAGGERED);
	pressureweightcomputer.compute_cutcell_weights(cc_weights);

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, m_vel, dummysurface);

	//if (m_moving_solids)
	//{
	//	projectdivergence.set_collision_velocity(m_collision_vel);
	//}

	VectorGrid<Real> valid(m_collision.xform(), m_collision.size(), VectorGridSettings::STAGGERED);
	
	projectdivergence.project(liquid_weights, cc_weights);

	// Update velocity field
	projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

	projectdivergence.apply_valid(valid);

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

	advect_smoke(dt, IntegratorSettings::RK3);
	advect_velocity(dt, IntegratorSettings::RK3);

}