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
		myFluidVelocities.resize(myMaterialCount);
		myFluidSurfaces.resize(myMaterialCount);
		myFluidDensities.resize(myMaterialCount);

		for (unsigned i = 0; i < myMaterialCount; ++i)
			myFluidVelocities[i] = VectorGrid<Real>(xform, size, 0, VectorGridSettings::SampleType::STAGGERED);
	
		for (unsigned i = 0; i < myMaterialCount; ++i)
			myFluidSurfaces[i] = LevelSet2D(xform, size, narrowBand);

		mySolidSurface = LevelSet2D(myXform, size, narrowBand);
	}

	template<typename ForceSampler>
	void addForce(const Real dt, const unsigned material, const ForceSampler& force);

	void addForce(const Real dt, const unsigned material, const Vec2R& force);

	void advectFluidSurfaces(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectFluidVelocities(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& renderer);

	// Useful for CFL
	Real maxVelocityMagnitude() const
	{
		Real maxVelocity = 0;
		for (int i = 0; i < myMaterialCount; ++i)
			maxVelocity = std::max(maxVelocity, myFluidVelocities[i].maxMagnitude());
		return maxVelocity;
	}

	void setSolidSurface(const LevelSet2D& solidSurface);

	void setMaterial(const LevelSet2D &surface, const Real density, const unsigned material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isMatched(myFluidSurfaces[material]));
	    myFluidSurfaces[material] = surface;
	    myFluidDensities[material] = density;
	}

	void setMaterial(const LevelSet2D &surface, const VectorGrid<Real> &velocity,
						const Real density, const unsigned material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isMatched(myFluidSurfaces[material]));
	    myFluidSurfaces[material] = surface;

	    assert(velocity.isMatched(myFluidVelocities[material]));
	    myFluidVelocities[material] = velocity;

	    myFluidDensities[material] = density;
	}

	void drawMaterialSurface(Renderer &renderer, unsigned material);
	void drawMaterialVelocity(Renderer &renderer, Real length, unsigned material) const;
	void drawSolidSurface(Renderer &renderer);

private:

	std::vector<VectorGrid<Real>> myFluidVelocities;
	std::vector<LevelSet2D> myFluidSurfaces;
	std::vector<Real> myFluidDensities;

	LevelSet2D mySolidSurface;

	const Vec2ui myGridSize;
	const Transform myXform;
	const unsigned myMaterialCount;
	unsigned myInitializedMaterialsCount;
};

#endif