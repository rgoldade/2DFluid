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

namespace FluidSim2D
{

class EulerianSmokeSimulator
{
public:
	EulerianSmokeSimulator(const Transform& xform, Vec2i size, double ambientTemperature = 300)
		: myXform(xform), myAmbientTemperature(ambientTemperature)
	{
		myVelocity = VectorGrid<double>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<double>(myXform, size, 0, VectorGridSettings::SampleType::STAGGERED);

		mySolidSurface = LevelSet(myXform, size, 5);

		mySmokeDensity = ScalarGrid<double>(myXform, size, 0);
		mySmokeTemperature = ScalarGrid<double>(myXform, size, myAmbientTemperature);

		myOldPressure = ScalarGrid<double>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<double>& solidVelocity);

	void setFluidVelocity(const VectorGrid<double>& velocity);
	void setSmokeSource(const ScalarGrid<double>& density, const ScalarGrid<double>& temperature);

	void advectFluidMaterial(double dt, InterpolationOrder order);
	void advectFluidVelocity(double dt, InterpolationOrder order);
	void advectOldPressure(double dt, InterpolationOrder order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(double dt);

	// Useful for CFL
	double maxVelocityMagnitude() { return myVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer) const;
	void drawFluidDensity(Renderer& renderer, double maxDensity);
	void drawFluidVelocity(Renderer& renderer, double length) const;

	void drawSolidSurface(Renderer& renderer);
	void drawSolidVelocity(Renderer& renderer, double length) const;

private:

	// Simulation containers
	VectorGrid<double> myVelocity, mySolidVelocity;
	LevelSet mySolidSurface;
	ScalarGrid<double> mySmokeDensity, mySmokeTemperature;

	double myAmbientTemperature;

	Transform myXform;

	ScalarGrid<double> myOldPressure;
};

}
#endif