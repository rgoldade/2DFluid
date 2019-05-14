#ifndef SIMULATIONS_MULTIMATERIALLIQUID_H
#define SIMULATIONS_MULTIMATERIALLIQUID_H

#include "AdvectField.h"
#include "Common.h"
#include "ExtrapolateField.h"
#include "Integrator.h"
#include "LevelSet2D.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// MultiMaterialLiquid.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid liquid simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

class MultiMaterialLiquid
{
public:
	MultiMaterialLiquid(const Transform& xform, Vec2ui size, unsigned materials, Real narrowBand = 5.)
		: myXform(xform)
		, myGridSize(size)
		, myMaterialCount(materials)
		, myInitializedMaterialsCount(0)
	{
		assert(myMaterialCount > 1);
		myVelocities.resize(myMaterialCount);
		mySurfaces.resize(myMaterialCount);
		myDensities.resize(myMaterialCount);

		for (unsigned i = 0; i < myMaterialCount; ++i)
			myVelocities[i] = VectorGrid<Real>(xform, size, 0, VectorGridSettings::SampleType::STAGGERED);
	
		for (unsigned i = 0; i < myMaterialCount; ++i)
			mySurfaces[i] = LevelSet2D(xform, size, narrowBand);

		myCollisionSurface = LevelSet2D(myXform, size, narrowBand);
	}

	template<typename ForceSampler>
	void addForce(const Real dt, const unsigned material, const ForceSampler& force);

	void addForce(const Real dt, const unsigned material, const Vec2R& force);

	void advectSurface(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectVelocity(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& renderer);

	// Useful for CFL
	Real maxVelocityMagnitude() const
	{
		Real maxVelocity = 0;
		for (int i = 0; i < myMaterialCount; ++i)
			maxVelocity = std::max(maxVelocity, myVelocities[i].maxMagnitude());
		return maxVelocity;
	}

	void setCollisionVolume(const LevelSet2D& collision);

	void setMaterial(const LevelSet2D &surface, const Real density, const unsigned material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isMatched(mySurfaces[material]));
	    mySurfaces[material] = surface;
	    myDensities[material] = density;
	}

	void setMaterial(const LevelSet2D &surface, const VectorGrid<Real> &velocity,
			    const Real density, const unsigned material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isMatched(mySurfaces[material]));
	    mySurfaces[material] = surface;

	    assert(velocity.isMatched(myVelocities[material]));
	    myVelocities[material] = velocity;

	    myDensities[material] = density;
	}

	void drawMaterialSurface(Renderer &renderer, unsigned material);
	void drawMaterialVelocity(Renderer &renderer, Real length, unsigned material) const;
	void drawCollisionSurface(Renderer &renderer);

private:

	std::vector<VectorGrid<Real>> myVelocities;
	std::vector<LevelSet2D> mySurfaces;
	std::vector<Real> myDensities;

	LevelSet2D myCollisionSurface;

	const Vec2ui myGridSize;
	const Transform myXform;
	const unsigned myMaterialCount;
	unsigned myInitializedMaterialsCount;
};

#endif