#pragma once

#include "Core.h"

#include "LevelSet2D.h"
#include "Renderer.h"

///////////////////////////////////
//
// FluidParticles.h/cpp
// Ryan Goldade 2016
//
// 2d particle surface tracker with
// foward advection, resampling,
// rendering.
//
////////////////////////////////////

class FluidParticles
{
public:
	FluidParticles() : m_prad(0) {}

	// The particle radius is used to construct a surface around the particles.
	// This is also used during reseeding so we don't put particles too close to the surface.
	// The count is the target number of particles to see into a particle grid cell. The\
	// oversample is how many more particles should be created at cells near the supplied
	// surface.
	FluidParticles(Real particle_radius, size_t count, Real oversample = 1., bool track_vel = false)
		: m_prad(particle_radius)
		, m_count(count)
		, m_oversample(oversample)
		, m_track_vel(track_vel)
	{}

	// The initialize step seeds particles inside of the given surface.
	// The particles are seeded in each associated voxel of the LevelSet
	// grid. If a voxel is completely contained inside the surface, "count"
	// many particles will be seeded. Otherwise some fraction of the amount
	// will be generated.
	void init(const LevelSet2D& surface);
	
	void set_velocity(const VectorGrid<Real>& vel);
	void apply_velocity(VectorGrid<Real>& vel);
	void increment_velocity(VectorGrid<Real>& vel);
	void blend_velocity(const VectorGrid<Real>& vel_old,
						const VectorGrid<Real>& vel_new,
						Real blend);

	void construct_surface(LevelSet2D& surface) const;
	void draw_points(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), size_t size = 1) const;
	void draw_velocity(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;

	// Seed particles into areas of the surface that are under represented
	// and delete particles in areas that are over represented. If the particle
	// count is below the min multiplier, then the cell will be seeded up to the count.
	// If the particle count is above the max multiplier, then the cell will emptied down
	// to the count.
	void reseed(const LevelSet2D& surface, Real min = 1., Real max = 1., const VectorGrid<Real>* vel = NULL, Real seed = 0);

	// Push particles inside collision objects back to the surface
	void bump_particles(const LevelSet2D& collision);

	size_t size() const { return m_parts.size(); }
	Vec2R get_position(size_t part) const { assert(part < size());  return m_parts[part]; }

	const std::vector<Vec2R>& get_positions() const { return m_parts; }

	template<typename VelField, typename Integrator>
	void forward_advect(Real dt, const VelField& vel, const Integrator& f);

protected:
	std::vector<Vec2R> m_parts, m_add_parts;
	std::vector<Vec2R> m_vel;
	bool m_track_vel;
	Real m_prad;

	size_t m_count;
	Real m_oversample;
};

template<typename VelField, typename Integrator>
void FluidParticles::forward_advect(Real dt, const VelField& vel, const Integrator& f)
{
	assert(dt >= 0);
	for (auto& p : m_parts)
		p = f(p, dt, vel);
}
