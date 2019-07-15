#ifndef SIMULATIONS_MULTIMATERIALLIQUID_H
#define SIMULATIONS_MULTIMATERIALLIQUID_H

#include "AdvectField.h"
#include "Common.h"
#include "ExtrapolateField.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// MultiMaterialLiquid.h/cpp
// Ryan Goldade 2018
//
// A purely Eulerian implementation of
// the multi-FLIP style multi-material
// fluid simulator.
//
////////////////////////////////////

class MultiMaterialLiquid
{
public:
	MultiMaterialLiquid(const Transform& xform, Vec2i size, int materials, Real narrowBand = 5.)
		: myXform(xform)
		, myGridSize(size)
		, myMaterialCount(materials)
		, myInitializedMaterialsCount(0)
	{
		assert(myMaterialCount > 1);
		myFluidVelocities.resize(myMaterialCount);
		myFluidSurfaces.resize(myMaterialCount);
		myFluidDensities.resize(myMaterialCount);

		for (int i = 0; i < myMaterialCount; ++i)
			myFluidVelocities[i] = VectorGrid<Real>(xform, size, 0, VectorGridSettings::SampleType::STAGGERED);
	
		for (int i = 0; i < myMaterialCount; ++i)
			myFluidSurfaces[i] = LevelSet(xform, size, narrowBand);

		mySolidSurface = LevelSet(myXform, size, narrowBand);
	}

	template<typename ForceSampler>
	void addForce(Real dt, int material, const ForceSampler& force);

	void addForce(Real dt, int material, const Vec2R& force);

	void advectFluidSurfaces(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectFluidVelocities(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& renderer, int frame = 0);

	// Useful for CFL
	Real maxVelocityMagnitude() const
	{
		Real maxVelocity = 0;
		for (int i = 0; i < myMaterialCount; ++i)
			maxVelocity = std::max(maxVelocity, myFluidVelocities[i].maxMagnitude());
		return maxVelocity;
	}

	void setSolidSurface(const LevelSet& solidSurface);

	void setMaterial(const LevelSet &surface, Real density, int material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isGridMatched(myFluidSurfaces[material]));
	    myFluidSurfaces[material] = surface;
	    myFluidDensities[material] = density;
	}

	void setMaterial(const LevelSet &surface, const VectorGrid<Real> &velocity,
						Real density, int material)
	{
	    assert(material < myMaterialCount);

	    assert(surface.isGridMatched(myFluidSurfaces[material]));
	    myFluidSurfaces[material] = surface;

	    assert(velocity.isGridMatched(myFluidVelocities[material]));
	    myFluidVelocities[material] = velocity;

	    myFluidDensities[material] = density;
	}

	void drawMaterialSurface(Renderer &renderer, int material);
	void drawMaterialVelocity(Renderer &renderer, Real length, int material) const;
	void drawSolidSurface(Renderer &renderer);

private:

	std::vector<VectorGrid<Real>> myFluidVelocities;
	std::vector<LevelSet> myFluidSurfaces;
	std::vector<Real> myFluidDensities;

	LevelSet mySolidSurface;

	const Vec2i myGridSize;
	const Transform myXform;
	const unsigned myMaterialCount;
	unsigned myInitializedMaterialsCount;
};

#endif