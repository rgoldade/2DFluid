#pragma once

#include "core.h"

#include "LevelSet2d.h"
#include "Renderer.h"

#include "QuadTreeIndex.h"

///////////////////////////////////
//
// MarkerParticles.h/cpp
// Ryan Goldade 2016
//
// 2d particle surface tracker with
// foward advection, resampling,
// rendering.
//
////////////////////////////////////

class MarkerParticles
{
public:
	MarkerParticles() : m_prad(0) {}

	// The particle radius is used to construct a surface around the particles.
	// This is also used during reseeding so we don't put particles too close to the surface.
	// The count is the target number of particles to see into a particle grid cell. The\
	// oversample is how many more particles should be created at cells near the supplied
	// surface.
	MarkerParticles(Real particle_radius, size_t count, Real oversample = 1.)
		: m_prad(particle_radius)
		, m_count(count)
		, m_oversample(oversample)
	{}

	// The initialize step seeds particles inside of the given surface.
	// The particles are seeded in each associated voxel of the LevelSet
	// grid. If a voxel is completely contained inside the surface, "count"
	// many particles will be seeded. Otherwise some fraction of the amount
	// will be generated.
	void init(const LevelSet2D& surface);

	void construct_surface(LevelSet2D& surface) const;
	void draw_points(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), size_t size = 1) const;

	// Seed particles into areas of the surface that are under represented
	// and delete particles in areas that are over represented. If the particle
	// count is below the min multiplier, then the cell will be seeded up to the count.
	// If the particle count is above the max multiplier, then the cell will emptied down
	// to the count.
	void reseed(const LevelSet2D& surface, Real min = 1., Real max = 1.);

	// Push particles inside collision objects back to the surface
	void bump_particles(const LevelSet2D& collision);

	template<typename VelField, typename Integrator>
	void forward_advect(Real dt, const VelField& vel, const Integrator& f);
protected:
	std::vector<Vec2R> m_parts, m_add_parts;
	Real m_prad;

	size_t m_count;
	Real m_oversample;
};

template<typename VelField, typename Integrator>
void MarkerParticles::forward_advect(Real dt, const VelField& vel, const Integrator& f)
{
	assert(dt >= 0);
	for (auto& p : m_parts)
		p = f(p, dt, vel);
}