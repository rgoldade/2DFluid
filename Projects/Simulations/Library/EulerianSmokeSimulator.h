#ifndef EULERIAN_SMOKE_SIMULATOR_H
#define EULERIAN_SMOKE_SIMULATOR_H

#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// EulerianSmokeSimulator.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid smoke simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

namespace FluidSim2D::RegularGridSim
{

using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::Utilities;

class EulerianSmokeSimulator
{
public:
	EulerianSmokeSimulator(const Transform& xform, Vec2i size, float ambientTemperature = 300)
		: myXform(xform), myAmbientTemperature(ambientTemperature)
	{
		myVelocity = VectorGrid<float>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<float>(myXform, size, 0, VectorGridSettings::SampleType::STAGGERED);

		mySolidSurface = LevelSet(myXform, size, 5);

		mySmokeDensity = ScalarGrid<float>(myXform, size, 0);
		mySmokeTemperature = ScalarGrid<float>(myXform, size, myAmbientTemperature);

		myOldPressure = ScalarGrid<float>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<float>& solidVelocity);

	void setFluidVelocity(const VectorGrid<float>& velocity);
	void setSmokeSource(const ScalarGrid<float>& density, const ScalarGrid<float>& temperature);

	void advectFluidMaterial(float dt, InterpolationOrder order);
	void advectFluidVelocity(float dt, InterpolationOrder order);
	void advectOldPressure(float dt, InterpolationOrder order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(float dt);

	// Useful for CFL
	float maxVelocityMagnitude() { return myVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer) const;
	void drawFluidDensity(Renderer& renderer, float maxDensity);
	void drawFluidVelocity(Renderer& renderer, float length) const;

	void drawSolidSurface(Renderer& renderer);
	void drawSolidVelocity(Renderer& renderer, float length) const;

private:

	// Simulation containers
	VectorGrid<float> myVelocity, mySolidVelocity;
	LevelSet mySolidSurface;
	ScalarGrid<float> mySmokeDensity, mySmokeTemperature;

	float myAmbientTemperature;

	Transform myXform;

	ScalarGrid<float> myOldPressure;
};

}
#endif