#include <random>

#include "MarkerParticles.h"

void MarkerParticles::draw_points(Renderer& renderer, const Vec3f& colour, size_t size) const
{
	renderer.add_points(m_parts, colour, size);
	//renderer.add_points(m_add_parts, Vec3f(0, 1, 0), 3. *size);
}

void MarkerParticles::draw_velocity(Renderer& renderer, const Vec3f& colour, Real length) const
{
	assert(m_track_vel);

	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (size_t p = 0; p < m_parts.size(); ++p)
	{
		start_points.push_back(m_parts[p]);
		end_points.push_back(m_parts[p] + m_vel[p] * length);
	}
	
	renderer.add_lines(start_points, end_points, colour);
}

void MarkerParticles::init(const LevelSet2D& surface)
{
	// Set up random number generator to jitter particles
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-.5, .5);
	
	m_parts.clear();
	for (size_t i = 0; i < surface.size()[0]; ++i)
		for (size_t j = 0; j < surface.size()[1]; ++j)
		{
			if (surface(i, j) < 2. * surface.dx())
			{
				Real sample_count = (surface(i, j) > -2. * surface.dx()) ? m_count * m_oversample : m_count;
				
				// Dissect the cell into pieces similar to super sampling
				// but each sample is a range to seed a random particle into
				Real samples = floor(sqrt(sample_count));
				Real sample_dx = 1. / samples;
				size_t seed_count = 0;
				for (Real x = ((Real)i - .5) + (.5 * sample_dx); x < (Real)i + .5; x += sample_dx)
					for (Real y = ((Real)j - .5) + (.5 * sample_dx); y < (Real)j + .5; y += sample_dx)
					{
						Real x_rand = x + dis(gen) * sample_dx;
						Real y_rand = y + dis(gen) * sample_dx;
						Vec2R wpos = surface.idx_to_ws(Vec2R(x_rand, y_rand));
						if (surface.interp(wpos) <= 0.) m_parts.push_back(wpos);
						++seed_count;
					}
				// Fill in left overs
				for (; seed_count < sample_count; ++seed_count)
				{
					Real x_rand = (Real)i + dis(gen);
					Real y_rand = (Real)j + dis(gen);
					Vec2R wpos = surface.idx_to_ws(Vec2R(x_rand, y_rand));
					if (surface.interp(wpos) <= 0.) m_parts.push_back(wpos);
				}
			}
		}

	m_vel.resize(m_parts.size(), Vec2R(0));
}

void MarkerParticles::set_velocity(const VectorGrid<Real>& vel)
{
	assert(m_track_vel);
	m_vel.resize(m_parts.size());

	for (size_t p = 0; p < m_parts.size(); ++p)
		m_vel[p] = vel.interp(m_parts[p]);
}

void MarkerParticles::apply_velocity(VectorGrid<Real>& vel)
{
	assert(m_track_vel);
	for (size_t axis = 0; axis < 2; ++axis)
	{
		UniformGrid<Real> denominator(vel.size(axis), 0);
		UniformGrid<Real> numerator(vel.size(axis), 0);

		for (size_t p = 0; p < m_parts.size(); ++p)
		{
			// Iterate over nearby voxels
			Vec2i ibbmin = floor(vel.ws_to_idx(m_parts[p], axis));
			Vec2i ibbmax = ceil(vel.ws_to_idx(m_parts[p], axis));

			max_union(ibbmin, Vec2i(0));
			min_union(ibbmax, Vec2i(vel.size(axis)) - Vec2i(1));

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

		for (size_t i = 0; i < vel.size(axis)[0]; ++i)
			for (size_t j = 0; j < vel.size(axis)[1]; ++j)
			{
				if (denominator(i, j) > 0.)
				{
					vel(i, j, axis) = numerator(i, j) / denominator(i, j);
				}
			}
	}
}
void MarkerParticles::increment_velocity(VectorGrid<Real>& vel)
{
	assert(m_vel.size() == m_parts.size() && m_track_vel);

	for (size_t p = 0; p < m_parts.size(); ++p)
		m_vel[p] += vel.interp(m_parts[p]);
}

void MarkerParticles::blend_velocity(const VectorGrid<Real>& vel_old,
										const VectorGrid<Real>& vel_new,
										Real blend)
{
	assert(m_vel.size() == m_parts.size() && m_track_vel);

	for (size_t p = 0; p < m_parts.size(); ++p)
	{
		Vec2R vel_part = m_vel[p];
		Vec2R vel_pic = vel_new.interp(m_parts[p]);
		Vec2R vel_flip = vel_pic - vel_old.interp(m_parts[p]);

		m_vel[p] = (1. - blend) * vel_pic + (blend) * (vel_part + vel_flip);
	}
}

void MarkerParticles::construct_surface(LevelSet2D& surface) const
{
	UniformGrid<Real> denominator(surface.size(), 0);
	UniformGrid<Vec2R> numerator(surface.size(), Vec2R(0));

	auto kernel = [](const Vec2R& x, const Vec2R& xi, Real h) -> Real
	{
		Real m2 = dist2(x, xi)/sqr(h);
		if (m2 >= 1.) return 0.;
		else return pow((1 - m2), 3);
	};

	Real sqr_dist = sqr(3 * m_prad);
	for (auto p : m_parts)
	{
		// Iterate over nearby voxels
		Vec2i ibbmin = floor(surface.ws_to_idx(p - Vec2R(3. * m_prad)));
		Vec2i ibbmax = ceil(surface.ws_to_idx(p + Vec2R(3. * m_prad)));

		max_union(ibbmin, Vec2i(0));
		min_union(ibbmax, Vec2i(surface.size()) - Vec2i(1));

		for (int i = ibbmin[0]; i <= ibbmax[0]; ++i)
			for (int j = ibbmin[1]; j <= ibbmax[1]; ++j)
			{
				if (i < 0 || j < 0 || i >= surface.size()[0]
					|| j >= surface.size()[1]) continue; 

				Vec2R grid_pos = surface.idx_to_ws(Vec2R(i, j));
				if (dist2(grid_pos, p) <= sqr_dist)
				{
					Real kern = kernel(grid_pos, p, 3. * m_prad);
					numerator(i, j) += kern * p;
					denominator(i, j) += kern;
				}
			}
	}

	for (size_t i = 0; i < surface.size()[0]; ++i)
		for (size_t j = 0; j < surface.size()[1]; ++j)
		{
			if (denominator(i, j) > 0.)
			{
				Vec2R grid_pos = surface.idx_to_ws(Vec2R(i, j));
				surface.set_phi(Vec2st(i, j), dist(grid_pos, numerator(i, j) / denominator(i, j)) - m_prad);
			}
			else
				surface.set_phi(Vec2st(i, j), m_prad);
		}

	surface.reinit(2);
}

void MarkerParticles::reseed(const LevelSet2D& surface, Real min, Real max, const VectorGrid<Real>* vel)
{
	m_add_parts.clear();

	// Set up random number generator to jitter particles
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-.5, .5);

	// Load up particles into grid cells
	UniformGrid<std::vector<size_t>> particle_grid(surface.size());

	for (size_t p = 0; p < m_parts.size(); ++p)
	{
		Vec2R pos = surface.ws_to_idx(m_parts[p]);
		Vec2i rpos = round(pos);
		
		if (rpos[0] < 0 || rpos[1] < 0
			|| rpos[0] >= surface.size()[0]
			|| rpos[1] >= surface.size()[1]) continue;

		// Add index to particle grid to reference later 
		particle_grid(rpos[0], rpos[1]).push_back(p);
	}

	std::vector<Vec2R> add_parts;
	std::vector<size_t> del_parts;

	for (size_t i = 0; i < surface.size()[0]; ++i)
		for (size_t j = 0; j < surface.size()[1]; ++j)
		{
			// Only reseed near/in the surface
			if (surface(i, j) < 2 * surface.dx())
			{
				size_t count = (surface(i, j) > -2 * surface.dx()) ? m_count * m_oversample : m_count;
				if (particle_grid(i, j).size() > count * max)
				{
					// Delete particles until we're down to the right amount
					for (size_t del_count = particle_grid(i, j).size(); del_count > count; --del_count)
						del_parts.push_back(particle_grid(i, j)[del_count - 1]);
				}

				else if (particle_grid(i, j).size() < count * min)
				{
					// Add particles until we're up to the right amount
					for (size_t add_count = particle_grid(i, j).size(); add_count < count; ++add_count)
					{
						// TODO: build a single random generator
						Real x_rand = (Real)i + dis(gen);
						Real y_rand = (Real)j + dis(gen);
						Vec2R wpos = surface.idx_to_ws(Vec2R(x_rand, y_rand));

						if (surface.interp(wpos) <= -m_prad) add_parts.push_back(wpos);
					}
				}
			}

			else
			{
				for (size_t del_count = particle_grid(i, j).size(); del_count > 0; --del_count)
					del_parts.push_back(particle_grid(i, j)[del_count - 1]);
			}
		}

	// Reverse sort the parts to be deleted so we don't accidentally swap and delete the wrong particles
	std::sort(del_parts.begin(), del_parts.end(), std::greater<int>());
	for (auto d : del_parts)
	{
		size_t psize = m_parts.size();
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
			for (size_t p = 0; p < add_parts.size(); ++p)
			{
				Vec2R pvel = vel->interp(add_parts[p]);
				m_vel.push_back(pvel);
			}
		}
		else
		for (size_t p = 0; p < add_parts.size(); ++p)
		{
			m_vel.push_back(Vec2R(0));
		}
		assert(m_vel.size() == m_parts.size());
	}

	m_add_parts = add_parts;
}

void MarkerParticles::bump_particles(const LevelSet2D& collision)
{
	std::vector<size_t> del_parts;
	for (size_t p = 0; p < m_parts.size(); ++p)
	{
		if (collision.interp(m_parts[p]) <= 0.)
			m_parts[p] -= .9 *collision.interp(m_parts[p]) * collision.normal_const(m_parts[p]);

	}
}