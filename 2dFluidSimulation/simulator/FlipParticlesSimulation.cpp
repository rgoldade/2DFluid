#include "FlipParticlesSimulation.h"
#include "PressureProjection.h"
#include "ViscositySolver.h"
#include "ExtrapolateField.h"
#include "ComputeWeights.h"

void FlipParticlesSimulation::draw_grid(Renderer& renderer) const
{
	m_surface.draw_grid(renderer);
}

void FlipParticlesSimulation::draw_surface(Renderer& renderer)
{
	m_particles.draw_points(renderer, Vec3f(0., 0., 1.0));
	m_surface.draw_surface(renderer, Vec3f(0., 0., 1.0));
}

void FlipParticlesSimulation::draw_air(Renderer& renderer)
{
	if (m_air_volume)
	{
		m_air_particles.draw_points(renderer, Vec3f(1., 0., 0.));
		m_air_surface.draw_surface(renderer, Vec3f(1., 0., 0.));
	}
}

void FlipParticlesSimulation::draw_collision(Renderer& renderer)
{
	m_collision.draw_surface(renderer);
}

void FlipParticlesSimulation::draw_collision_vel(Renderer& renderer, Real length) const
{
	if (m_moving_solids)
		m_collision_vel.draw_sample_point_vectors(renderer, Vec3f(0, 1, 0), m_collision_vel.dx() * length);
}


void FlipParticlesSimulation::draw_velocity(Renderer& renderer, Real length, bool from_particles) const
{
	if (from_particles)
		m_particles.draw_velocity(renderer, Vec3f(0), m_vel.dx() * length);
	else
		m_vel.draw_sample_point_vectors(renderer, Vec3f(0), m_vel.dx() * length);
}

// Incoming collision volume must already be inverted
void FlipParticlesSimulation::set_collision_volume(const LevelSet2D& collision)
{
	assert(collision.inverted());

	Mesh2D temp_mesh;
	collision.extract_mesh(temp_mesh);

	// TODO : consider making collision node sampled
	m_collision.set_inverted();
	m_collision.init(temp_mesh, false);
	m_collision.build_gradient();
}

void FlipParticlesSimulation::set_surface_volume(const LevelSet2D& fluid)
{
	Mesh2D temp_mesh;
	fluid.extract_mesh(temp_mesh);

	m_surface.init(temp_mesh, false);
	m_particles.init(m_surface);
}

void FlipParticlesSimulation::set_surface_velocity(const VectorGrid<Real>& vel)
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

	m_particles.set_velocity(vel);
}

void FlipParticlesSimulation::set_collision_velocity(const VectorGrid<Real>& collision_vel)
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

void FlipParticlesSimulation::set_air_volume()
{
	assert(m_surface.size() == m_air_surface.size());

	for (int i = 0; i < m_surface.size()[0]; ++i)
		for (int j = 0; j < m_surface.size()[1]; ++j)
		{
			Real phi = -std::min(m_collision(i, j), m_surface(i, j));
			m_air_surface.set_phi(Vec2st(i, j), phi);
		}

	m_air_surface.reinit();
	m_air_particles.reseed(m_air_surface);

	m_air_volume = true;
}

void FlipParticlesSimulation::add_surface_volume(const LevelSet2D& surface)
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
	m_particles.reseed(surface);

	// If we've changed the liquid volume, we should really reset the air volume
	if (m_air_volume) set_air_volume();
}

template<typename ForceSampler>
void FlipParticlesSimulation::add_force(const ForceSampler& force, Real dt)
{
	VectorGrid<Real> force_vel(m_surface.xform(), m_surface.size(), 0, VectorGridSettings::STAGGERED);
	for (size_t axis = 0; axis < 2; ++axis)
	{
		size_t x_size = m_vel.size(axis)[0];
		size_t y_size = m_vel.size(axis)[1];

		for (size_t x = 0; x < x_size; ++x)
			for (size_t y = 0; y < y_size; ++y)
			{
				Vec2R pos = m_vel.idx_to_ws(Vec2R(x, y), axis);
				//m_vel.set(x, y, axis, m_vel(x, y, axis) + force(pos, axis) * dt);

				force_vel.set(x, y, axis, force(pos, axis) * dt);
			}
	}

	m_particles.increment_velocity(force_vel);
}

void FlipParticlesSimulation::add_force(const Vec2R& force, Real dt)
{
	add_force([&](Vec2R, size_t axis) {return force[axis]; }, dt);
}

void FlipParticlesSimulation::advect_surface(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<LevelSet2D> advector(m_vel, m_surface);
	advector.set_collision_volumes(m_collision);
	switch (order)
	{
	case IntegratorSettings::FE:
		m_particles.forward_advect(dt, advector, Integrator::forward_euler<Vec2R, AdvectField<LevelSet2D>>());
		break;
	case IntegratorSettings::RK3:
		m_particles.forward_advect(dt, advector, Integrator::rk3<Vec2R, AdvectField<LevelSet2D>>());
		break;
	default:
		assert(false);
	}

	if (m_air_volume)
	{
		AdvectField<LevelSet2D> air_advector(m_vel, m_air_surface);
		advector.set_collision_volumes(m_collision);
		switch (order)
		{
		case IntegratorSettings::FE:
			m_air_particles.forward_advect(dt, advector, Integrator::forward_euler<Vec2R, AdvectField<LevelSet2D>>());
			break;
		case IntegratorSettings::RK3:
			m_air_particles.forward_advect(dt, advector, Integrator::rk3<Vec2R, AdvectField<LevelSet2D>>());
			break;
		default:
			assert(false);
		}
	}
}

void FlipParticlesSimulation::advect_viscosity(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<ScalarGrid<Real>> advector(m_vel, m_variableviscosity);
	//advector.set_collision_volumes(m_collision);
	ScalarGrid<Real> temp_visc = m_variableviscosity;
	advector.advect_field(dt, temp_visc, order);
	m_variableviscosity = temp_visc;
}

void FlipParticlesSimulation::advect_velocity(Real dt, IntegratorSettings::Integrator order)
{
	AdvectField<VectorGrid<Real>> advector(m_vel, m_vel);
	//advector.set_collision_volumes(m_collision);
	VectorGrid<Real> temp_vel = m_vel;
	advector.advect_field(dt, temp_vel, order);
	m_vel = temp_vel;
}

Real FlipParticlesSimulation::compute_volume(bool liquid) const
{
	size_t samples = 2;
	Real sample_dx = 1. / (Real)samples;
	Real sign = liquid ? 1 : -1;

	Real volume = 0;
	// Loop over each cell in the grid
	for (int i = 0; i < m_surface.size()[0]; ++i)
		for (int j = 0; j < m_surface.size()[1]; ++j)
		{
			if (sign * m_surface(i,j) < m_surface.dx() * 2.)
			{
				int volcount = 0;
				for (Real x = ((Real)i - .5) + (.5 * sample_dx); x < (Real)i + .5; x += sample_dx)
					for (Real y = ((Real)j - .5) + (.5 * sample_dx); y < (Real)j + .5; y += sample_dx)
					{
						Vec2R wpos = m_surface.idx_to_ws(Vec2R(x, y)); // TODO: add an index space interpolant to level set code
						if (sign * m_surface.interp(wpos) <= 0. && m_collision.interp(wpos) > 0) ++volcount; // Liquid inside the collision doesn't count
					}

				volume += (Real)volcount;
			}
		}

	return volume * sqr(sample_dx) * sqr(m_surface.dx());
}

void FlipParticlesSimulation::run_simulation(Real dt, Renderer& renderer)
{
	// Extrapolate surface into collision by one cell
	LevelSet2D extrap_surface = m_surface;
	Real dx = m_surface.dx();
	for (size_t x = 0; x < extrap_surface.size()[0]; ++x)
		for (size_t y = 0; y < extrap_surface.size()[1]; ++y)
		{
			if (m_surface(x, y) > -.5 * dx && m_collision(x, y) <= 0)
			{
				extrap_surface.set_phi(Vec2st(x, y), m_surface(x, y) - dx);//extrap_surface.set_phi(Vec2st(x, y), -.5);//
			}
		}
	
	extrap_surface.reinit();
	//extrap_surface.draw_surface(renderer, Vec3f(1, 0, 1));

	ComputeWeights pressureweightcomputer(extrap_surface, m_collision);

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> liquid_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);
	pressureweightcomputer.compute_gf_weights(liquid_weights);

	VectorGrid<Real> cc_weights(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);
	pressureweightcomputer.compute_cutcell_weights(cc_weights);

	// Stamp particle velocity to velocity grid
	m_particles.apply_velocity(m_vel);
	VectorGrid<Real> old_vel = m_vel;

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, m_vel, extrap_surface);
	VectorGrid<Real> valid(m_surface.xform(), m_surface.size(), VectorGridSettings::STAGGERED);

	Real liquid_volume = compute_volume(true);
	std::cout << "Liquid volume: " << liquid_volume << std::endl;
	std::cout << "Bubble volume: " << compute_volume(false) << std::endl;
	//projectdivergence.batty_pressure_solve(liquid_weights, cc_weights, renderer, valid, m_vel);


	if (m_moving_solids)
	{
		projectdivergence.set_collision_velocity(m_collision_vel);
	}

	if (m_volume_correction)
	{
		Real xn = (liquid_volume - m_target_volume) / m_target_volume;
		m_accum_error += xn * dt;
		Real kp = 2.3 / (25. * dt);
		Real kl = sqr(kp / 4.);

		//projectdivergence.set_volume_correction(-kp*xn - kl * m_accum_error);
		projectdivergence.set_volume_correction(liquid_volume - m_target_volume);
	}
	if (m_st_scale != 0.)
	{
		const ScalarGrid<Real> surface_tension = m_surface.get_curvature();
		projectdivergence.set_surface_pressure(surface_tension, m_st_scale);
	}
	if (m_enforce_bubbles)
	{
		projectdivergence.enforce_bubbles();
		std::cout << "Liquid volume: " << compute_volume(true) << std::endl;
		std::cout << "Bubble volume: " << compute_volume(false) << std::endl;
	}

	ComputeWeights volumecomputer(m_surface, m_collision);
	ScalarGrid<Real> center_weights(m_surface.xform(), m_surface.size(), 0, ScalarGridSettings::CENTER);
	volumecomputer.compute_supersampled_volumes(center_weights, 3);

	projectdivergence.project(liquid_weights, cc_weights, center_weights, renderer);

	// Update velocity field
	projectdivergence.apply_solution(m_vel, liquid_weights, cc_weights);

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

		// Solve viscosity
		ViscositySolver viscosity(dt, m_vel, m_surface, m_collision);

		viscosity.set_viscosity(m_variableviscosity);
		viscosity.set_collision_weights(collision_center_vol_weights, collision_node_vol_weights);
		if (m_moving_solids)
		{
			viscosity.set_collision_velocity(m_collision_vel);
		}
		viscosity.solve(face_vol_weights, center_vol_weights, node_vol_weights, renderer);

		// Initialize and call pressure projection
		PressureProjection projectdivergence2(dt, m_vel, extrap_surface);

		if (m_moving_solids)
		{
			projectdivergence2.set_collision_velocity(m_collision_vel);
		}

		if (m_st_scale != 0.)
		{
			const ScalarGrid<Real> surface_tension = m_surface.get_curvature();
			projectdivergence2.set_surface_pressure(surface_tension, m_st_scale);
		}

		if (m_enforce_bubbles)
			projectdivergence2.enforce_bubbles();
//		projectdivergence2.project(liquid_weights, cc_weights, renderer);

		// Update velocity field
		projectdivergence2.apply_solution(m_vel, liquid_weights, cc_weights);
		projectdivergence2.apply_valid(valid);
	}
	else
	{
		// Label solved faces
		projectdivergence.apply_valid(valid);
	}

	// Extrapolate velocity
	ExtrapolateField<VectorGrid<Real>> extrapolator(m_vel);
	extrapolator.extrapolate(valid, 4);

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

	// Apply velocity back to particles
	m_particles.blend_velocity(old_vel, m_vel, 0.9);

	// Move particles through velocity field -- TODO: deal with errant loose particles
	m_particles.bump_particles(m_collision);
	m_particles.reseed(m_surface, .5, 1.5, &m_vel);

	// Manage particles on the air side
	if (m_air_volume)
	{
		m_air_particles.bump_particles(m_collision);
		m_air_particles.reseed(m_air_surface, .5, 1.5);
	}

	// Advection includes air-side particles so we must manage them first
	advect_surface(dt, IntegratorSettings::RK3);

	// Rebuild surface from particles
	m_particles.construct_surface(m_surface);
	if (m_air_volume)
	{
		m_air_particles.construct_surface(m_air_surface);
		//Reset overlapping volumes
		for (int i = 0; i < m_surface.size()[0]; ++i)
			for (int j = 0; j < m_surface.size()[1]; ++j)
			{
				std::vector<Real> min(3);
				min[0] = m_surface(i, j);
				min[1] = m_air_surface(i, j);
				min[2] = m_collision(i, j);
				std::sort(min.begin(), min.end());
				
				Real phiavg = .5*(min[0] + min[1]);
				m_air_surface.set_phi(Vec2st(i, j), m_air_surface(i, j) - phiavg);
				m_surface.set_phi(Vec2st(i, j), m_surface(i, j) - phiavg);
			}

		m_air_surface.reinit();
		m_surface.reinit();

		m_air_particles.reseed(m_air_surface, .5, 1.5);
		m_particles.reseed(m_surface, .5, 1.5, &m_vel);
	}

	//advect_velocity(dt, IntegratorSettings::RK3);

	if (m_solve_viscosity)
		advect_viscosity(dt, IntegratorSettings::RK3);
}
