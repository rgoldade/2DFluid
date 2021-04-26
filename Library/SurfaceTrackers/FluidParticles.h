#ifndef FLUIDSIM2D_FLUID_PARTICLES_H
#define FLUIDSIM2D_FLUID_PARTICLES_H

#include <vector>

#include "LevelSet.h"
#include "Renderer.h"
#include "Utilities.h"

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

namespace FluidSim2D
{

class FluidParticles
{
public:
	FluidParticles() : myParticleRadius(0) {}

	// The particle radius is used to construct a surface around the particles.
	// This is also used during reseeding so we don't put particles too close to the surface.
	// The count is the target number of particles to see into a particle grid cell. The\
	// oversample is how many more particles should be created at cells near the supplied
	// surface.
	FluidParticles(double particleRadius, int countPerCell, double oversample = 1., bool trackVelocity = false)
		: myParticleRadius(particleRadius)
		, myParticleDensity(countPerCell)
		, myOversampleRate(oversample)
		, myTrackVelocity(trackVelocity)
	{
		assert(countPerCell >= 0);
	}

	// The initialize step seeds particles inside of the given surface.
	// The particles are seeded in each associated voxel of the LevelSet
	// grid. If a voxel is completely contained inside the surface, "count"
	// many particles will be seeded. Otherwise some fraction of the amount
	// will be generated.
	void init(const LevelSet& surface);

	void setVelocity(const VectorGrid<double>& vel);
	void applyVelocity(VectorGrid<double>& vel);
	void incrementVelocity(VectorGrid<double>& vel);
	void blendVelocity(const VectorGrid<double>& vel_old,
		const VectorGrid<double>& vel_new,
		double blend);

	LevelSet surfaceParticles(const Transform& xform, const Vec2i& size, int narrowBand) const;
	void drawPoints(Renderer& renderer, const Vec3d& colour = Vec3d(1, 0, 0), double pointSize = 1) const;
	void drawVelocity(Renderer& renderer, const Vec3d& colour = Vec3d(0, 0, 1), double length = .25) const;

	// Seed particles into areas of the surface that are under represented
	// and delete particles in areas that are over represented. If the particle
	// count is below the min multiplier, then the cell will be seeded up to the count.
	// If the particle count is above the max multiplier, then the cell will emptied down
	// to the count.
	void reseed(const LevelSet& surface, double minDensity = 1., double maxDensity = 1., const VectorGrid<double>* velocity = nullptr, double seed = 0);

	// Push particles inside collision objects back to the surface
	void bumpParticles(const LevelSet& solidSurface);

	int particleCount() const { return int(myParticles.size()); }

	Vec2d getPosition(int particleIndex) const
	{
		assert(particleIndex >= 0 && particleIndex < myParticles.size());
		return myParticles[particleIndex];
	}

	const VecVec2d& getPositions() const { return myParticles; }

	void advect(double dt, const VectorGrid<double>& velocity, const IntegrationOrder order);

protected:
	VecVec2d myParticles, myNewParticles;
	VecVec2d myVelocity;
	bool myTrackVelocity;
	double myParticleRadius;

	int myParticleDensity;
	double myOversampleRate;
};

}

#endif