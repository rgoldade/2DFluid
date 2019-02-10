#ifndef LIBRARY_FLUIDPARTICLES_H
#define LIBRARY_FLUIDPARTICLES_H

#include "Common.h"

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
	FluidParticles() : myParticleRadius(0) {}

	// The particle radius is used to construct a surface around the particles.
	// This is also used during reseeding so we don't put particles too close to the surface.
	// The count is the target number of particles to see into a particle grid cell. The\
	// oversample is how many more particles should be created at cells near the supplied
	// surface.
	FluidParticles(Real particleRadius, unsigned countPerCell, Real oversample = 1., bool trackVelocity = false)
		: myParticleRadius(particleRadius)
		, myParticleDensity(countPerCell)
		, myOversampleRate(oversample)
		, myTrackVelocity(trackVelocity)
	{}

	// The initialize step seeds particles inside of the given surface.
	// The particles are seeded in each associated voxel of the LevelSet
	// grid. If a voxel is completely contained inside the surface, "count"
	// many particles will be seeded. Otherwise some fraction of the amount
	// will be generated.
	void init(const LevelSet2D& surface);
	
	void setVelocity(const VectorGrid<Real>& vel);
	void applyVelocity(VectorGrid<Real>& vel);
	void incrementVelocity(VectorGrid<Real>& vel);
	void blendVelocity(const VectorGrid<Real>& vel_old,
						const VectorGrid<Real>& vel_new,
						Real blend);

	LevelSet2D surfaceParticles(const Transform& xform, const Vec2ui& size, const unsigned narrowBand) const;
	void drawPoints(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), unsigned size = 1) const;
	void drawVelocity(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;

	// Seed particles into areas of the surface that are under represented
	// and delete particles in areas that are over represented. If the particle
	// count is below the min multiplier, then the cell will be seeded up to the count.
	// If the particle count is above the max multiplier, then the cell will emptied down
	// to the count.
	void reseed(const LevelSet2D& surface, Real minDensity = 1., Real maxDensity = 1., const VectorGrid<Real>* velocity = nullptr, Real seed = 0);

	// Push particles inside collision objects back to the surface
	void bumpParticles(const LevelSet2D& collision);

	unsigned particleCount() const { return myParticles.size(); }
	Vec2R getPosition(unsigned part) const { assert(part < myParticles.size());  return myParticles[part]; }

	const std::vector<Vec2R>& getPositions() const { return myParticles; }

	void advect(Real dt, const VectorGrid<Real>& velocity, const IntegrationOrder order);

protected:
	std::vector<Vec2R> myParticles, myNewParticles;
	std::vector<Vec2R> myVelocity;
	bool myTrackVelocity;
	Real myParticleRadius;

	unsigned myParticleDensity;
	Real myOversampleRate;
};

#endif