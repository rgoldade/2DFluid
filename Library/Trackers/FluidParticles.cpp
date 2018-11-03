#include <random>

#include "FluidParticles.h"

static Vec2R randomizer(const Vec2ui& coord, unsigned count, Real seed)
{
	int pos0 = (5915587277 * coord[0]) ^ (3367900313 * count) ^ (int)(3267000013. * seed);
	int pos1 = (2860486313 * coord[1]) ^ (9576890767 * count) ^ (int)(5463458053. * seed);

	pos0 = abs(pos0 % 100);
	pos1 = abs(pos1 % 100);

	return Vec2R((Real)(pos0) / 100. - .5, (Real)(pos1) / 100. - .5);
};

void FluidParticles::draw_points(Renderer& renderer, const Vec3f& colour, unsigned size) const
{
	renderer.add_points(m_parts, colour, size);
}

void FluidParticles::draw_velocity(Renderer& renderer, const Vec3f& colour, Real length) const
{
	assert(m_track_vel);

	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (unsigned p = 0; p < m_parts.size(); ++p)
	{
		start_points.push_back(m_parts[p]);
		end_points.push_back(m_parts[p] + m_vel[p] * length);
	}
	
	renderer.add_lines(start_points, end_points, colour);
}

void FluidParticles::init(const LevelSet2D& surface)
{
	m_parts.clear();

	for_each_voxel_range(Vec2ui(0), surface.size(), [&](const Vec2ui& cell)
	{
		if (surface(cell) < 2. * surface.dx())
		{
			Real sample_count = (surface(cell) > -2. * surface.dx()) ? m_count * m_oversample : m_count;

			unsigned seed_count = 0;

			for (unsigned seed_count = 0; seed_count < sample_count; ++seed_count)
			{
				Vec2R pos = Vec2R(cell) + randomizer(Vec2ui(cell), seed_count, 0);
				Vec2R wpos = surface.idx_to_ws(pos);

				if (surface.interp(wpos) <= 0.) m_parts.push_back(wpos);
			}
		}
	});

	m_vel.resize(m_parts.size(), Vec2R(0));
}

void FluidParticles::set_velocity(const VectorGrid<Real>& vel)
{
	assert(m_track_vel);
	m_vel.resize(m_parts.size());

	for (unsigned p = 0; p < m_parts.size(); ++p)
		m_vel[p] = vel.interp(m_parts[p]);
}

void FluidParticles::apply_velocity(VectorGrid<Real>& vel)
{
	assert(m_track_vel);
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		UniformGrid<Real> denominator(vel.size(axis), 0);
		UniformGrid<Real> numerator(vel.size(axis), 0);

		for (unsigned p = 0; p < m_parts.size(); ++p)
		{
			// Iterate over nearby voxels
			Vec2R ibbmin = floor(vel.ws_to_idx(m_parts[p], axis));
			Vec2R ibbmax = ceil(vel.ws_to_idx(m_parts[p], axis));

			max_union(ibbmin, Vec2R(0));
			min_union(ibbmax, Vec2R(vel.size(axis)) - Vec2R(1));

			for (int i = ibbmin[0]; i <= ibbmax[0]; ++i)
				for (int j = ibbmin[1]; j <= ibbmax[1]; ++j)
				{
					if (i < 0 || j < 0 || i >= vel.size(axis)[0]
						|| j >= vel.size(axis)[1]) continue;

					Vec2R grid_pos = vel.idx_to_ws(Vec2R(i, j), axis);

					if (dist2(grid_pos, m_parts[p]) <= sqr(vel.dx()))
					{
						Real k = 1. - dist(m_parts[p], grid_pos)/ vel.dx();
						numerator(i, j) += k * m_vel[p][axis];
						denominator(i, j) += k;
					}
				}
		}

		for_each_voxel_range(Vec2ui(0), vel.size(axis), [&](const Vec2ui& cell)
		{
			if (denominator(cell) > 0.)
			{
				vel(cell, axis) = numerator(cell) / denominator(cell);
			}
		});
	}
}
void FluidParticles::increment_velocity(VectorGrid<Real>& vel)
{
	assert(m_vel.size() == m_parts.size() && m_track_vel);

	for (unsigned p = 0; p < m_parts.size(); ++p)
		m_vel[p] += vel.interp(m_parts[p]);
}

void FluidParticles::blend_velocity(const VectorGrid<Real>& vel_old,
										const VectorGrid<Real>& vel_new,
										Real blend)
{
	assert(m_vel.size() == m_parts.size() && m_track_vel);

	for (unsigned p = 0; p < m_parts.size(); ++p)
	{
		Vec2R vel_part = m_vel[p];
		Vec2R vel_pic = vel_new.interp(m_parts[p]);
		Vec2R vel_flip = vel_pic - vel_old.interp(m_parts[p]);

		m_vel[p] = (1. - blend) * vel_pic + (blend) * (vel_part + vel_flip);
	}
}

void FluidParticles::construct_surface(LevelSet2D& surface) const
{
	Real r2 = 2 * m_prad;
	Real sqrrad2 = sqr(r2);

	UniformGrid<Real> min_grid(surface.size(), sqrrad2);
	Real dx = surface.dx();

	for (auto p : m_parts)
	{
		// Iterate over nearby voxels
		Vec2R ibbmin = floor(surface.ws_to_idx(p - Vec2R(3. * dx)));
		Vec2R ibbmax = ceil(surface.ws_to_idx(p + Vec2R(3. * dx)));

		max_union(ibbmin, Vec2R(0));
		min_union(ibbmax, Vec2R(surface.size()) - Vec2R(1));

		for (int i = ibbmin[0]; i <= ibbmax[0]; ++i)
			for (int j = ibbmin[1]; j <= ibbmax[1]; ++j)
			{
				if (i < 0 || j < 0 || i >= surface.size()[0]
					|| j >= surface.size()[1]) continue;

				Vec2R grid_pos = surface.idx_to_ws(Vec2R(i, j));
				Real sqrdist = dist2(grid_pos, p);
				if (sqrdist <= sqrrad2)
				{
					if (sqrdist < min_grid(i, j))
						min_grid(i, j) = sqrdist;
				}
			}
	}

	for_each_voxel_range(Vec2ui(0), surface.size(), [&](const Vec2ui& cell)
	{
		if (min_grid(cell) < sqrrad2)
		{
			surface(cell) = sqrt(min_grid(cell)) - m_prad;
		}
		else
			surface(cell) = r2;
	});

	surface.reinit();
}

void FluidParticles::reseed(const LevelSet2D& surface, Real min, Real max, const VectorGrid<Real>* vel, Real seed)
{
	m_add_parts.clear();

	// Load up particles into grid cells
	UniformGrid<std::vector<size_t>> particle_grid(surface.size());

	for (unsigned p = 0; p < m_parts.size(); ++p)
	{
		Vec2R pos = surface.ws_to_idx(m_parts[p]);
		Vec2R rpos = round(pos);
		
		if (rpos[0] < 0 || rpos[1] < 0
			|| rpos[0] >= surface.size()[0]
			|| rpos[1] >= surface.size()[1]) continue;

		// Add index to particle grid to reference later 
		particle_grid(rpos[0], rpos[1]).push_back(p);
	}

	std::vector<Vec2R> add_parts;
	std::vector<unsigned> del_parts;


	for_each_voxel_range(Vec2ui(0), surface.size(), [&](const Vec2ui& cell)
	{
		// Only reseed near/in the surface
		if (surface(cell) < 2 * surface.dx())
		{
			unsigned count = (surface(cell) > -2 * surface.dx()) ? m_count * m_oversample : m_count;
			if (particle_grid(cell).size() > count * max)
			{
				// Delete particles until we're down to the right amount
				for (int del_count = particle_grid(cell).size(); del_count > count; --del_count)
					del_parts.push_back(particle_grid(cell)[del_count - 1]);
			}

			else if (particle_grid(cell).size() < count * min)
			{
				// Add particles until we're up to the right amount
				for (unsigned add_count = particle_grid(cell).size(); add_count < count; ++add_count)
				{
					// TODO: build a single random generator

					Vec2R posrand = Vec2R(cell) + randomizer(cell, add_count, seed);
					Vec2R wpos = surface.idx_to_ws(posrand);

					if (surface.interp(wpos) <= -m_prad) add_parts.push_back(wpos);
				}
			}
		}

		else
		{
			for (int del_count = particle_grid(cell).size(); del_count > 0; --del_count)
				del_parts.push_back(particle_grid(cell)[del_count - 1]);
		}
	});

	// Reverse sort the parts to be deleted so we don't accidentally swap and delete the wrong particles
	std::sort(del_parts.begin(), del_parts.end(), std::greater<unsigned>());
	for (auto d : del_parts)
	{
		unsigned psize = m_parts.size();
		std::swap(m_parts[d], m_parts[psize - 1]);
		
		if (m_track_vel) std::swap(m_vel[d], m_vel[psize - 1]);
			
		m_parts.resize(psize - 1);
	}

	if (m_track_vel) m_vel.resize(m_parts.size());

	m_parts.insert(m_parts.end(), add_parts.begin(), add_parts.end());

	// Sample velocity field if tracked
	if (m_track_vel)
	{
		if (vel)
		{
			for (unsigned p = 0; p < add_parts.size(); ++p)
			{
				Vec2R pvel = vel->interp(add_parts[p]);
				m_vel.push_back(pvel);
			}
		}
		else
		for (unsigned p = 0; p < add_parts.size(); ++p)
		{
			m_vel.push_back(Vec2R(0));
		}
		assert(m_vel.size() == m_parts.size());
	}

	m_add_parts = add_parts;
}

void FluidParticles::bump_particles(const LevelSet2D& collision)
{
	std::vector<unsigned> del_parts;
	for (unsigned p = 0; p < m_parts.size(); ++p)
	{
		if (collision.interp(m_parts[p]) <= 0.)
			m_parts[p] -= .9 *collision.interp(m_parts[p]) * collision.normal(m_parts[p]);

	}
}

void FluidParticles::advect(Real dt, const VectorGrid<Real>& vel, const IntegrationOrder order)
{
	auto vel_func = [vel](Real, const Vec2R& world_pos)
	{
		return vel.interp(world_pos);
	};

	assert(dt >= 0);
	for (auto& p : m_parts)
		p = Integrator(dt, p, vel_func, order);
}