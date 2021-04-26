#ifndef MULTI_MATERIAL_LIQUID_SIMULATOR_H
#define MULTI_MATERIAL_LIQUID_SIMULATOR_H

#include "ExtrapolateField.h"
#include "FieldAdvector.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// MultiMaterialLiquidSimulator.h/cpp
// Ryan Goldade 2018
//
// A purely Eulerian implementation of
// the multi-FLIP style multi-material
// fluid simulator.
//
////////////////////////////////////

namespace FluidSim2D
{

class MultiMaterialLiquidSimulator
{
public:
	MultiMaterialLiquidSimulator(const Transform& xform, Vec2i size, int materials, double narrowBand = 5.)
		: myXform(xform)
		, myGridSize(size)
		, myMaterialCount(materials)
	{
		assert(myMaterialCount > 1);
		myFluidVelocities.resize(myMaterialCount);
		myFluidSurfaces.resize(myMaterialCount);
		myFluidDensities.resize(myMaterialCount);

		for (int i = 0; i < myMaterialCount; ++i)
			myFluidVelocities[i] = VectorGrid<double>(xform, size, 0, VectorGridSettings::SampleType::STAGGERED);

		for (int i = 0; i < myMaterialCount; ++i)
			myFluidSurfaces[i] = LevelSet(xform, size, narrowBand);

		mySolidSurface = LevelSet(myXform, size, narrowBand);

		myOldPressure = ScalarGrid<double>(myXform, size, 0);
	}

	template<typename ForceSampler>
	void addForce(double dt, int material, const ForceSampler& force);

	void addForce(double dt, int material, const Vec2d& force);

	void advectFluidSurfaces(double dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectFluidVelocities(double dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(double dt);

	// Useful for CFL
	double maxVelocityMagnitude() const
	{
		double maxVelocity = 0;
		for (int i = 0; i < myMaterialCount; ++i)
			maxVelocity = std::max(maxVelocity, myFluidVelocities[i].maxMagnitude());
		return maxVelocity;
	}

	void setSolidSurface(const LevelSet& solidSurface);

	void setMaterial(const LevelSet& surface, double density, int material)
	{
		assert(material < myMaterialCount);

		assert(surface.isGridMatched(myFluidSurfaces[material]));
		myFluidSurfaces[material] = surface;
		myFluidDensities[material] = density;
	}

	void setMaterial(const LevelSet& surface, const VectorGrid<double>& velocity,
		double density, int material)
	{
		assert(material < myMaterialCount);

		assert(surface.isGridMatched(myFluidSurfaces[material]));
		myFluidSurfaces[material] = surface;

		assert(velocity.isGridMatched(myFluidVelocities[material]));
		myFluidVelocities[material] = velocity;

		myFluidDensities[material] = density;
	}

	void drawMaterialSurface(Renderer& renderer, int material);
	void drawMaterialVelocity(Renderer& renderer, double length, int material) const;
	void drawSolidSurface(Renderer& renderer);

private:

	std::vector<VectorGrid<double>> myFluidVelocities;
	std::vector<LevelSet> myFluidSurfaces;
	std::vector<double> myFluidDensities;

	LevelSet mySolidSurface;

	Vec2i myGridSize;
	Transform myXform;
	int myMaterialCount;

	ScalarGrid<double> myOldPressure;
};

}

#endif