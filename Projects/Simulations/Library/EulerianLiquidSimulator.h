#ifndef EULERIAN_LIQUID_SIMULATOR_H
#define EULERIAN_LIQUID_SIMULATOR_H

#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// EulerianLiquidSimulator.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid liquid simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

namespace FluidSim2D
{

class EulerianLiquidSimulator
{
public:
	EulerianLiquidSimulator(const Transform& xform, Vec2i size, double cfl = 5.)
		: myXform(xform)
		, myDoSolveViscosity(false)
		, myCFL(cfl)
	{
		myLiquidVelocity = VectorGrid<double>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<double>(myXform, size, 0, VectorGridSettings::SampleType::STAGGERED);

		myLiquidSurface = LevelSet(myXform, size, myCFL);
		mySolidSurface = LevelSet(myXform, size, myCFL);

		myOldPressure = ScalarGrid<double>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<double>& solidVelocity);

	void setLiquidSurface(const LevelSet& liquidSurface);
	void setLiquidVelocity(const VectorGrid<double>& liquidVelocity);

	void setViscosity(const ScalarGrid<double>& viscosityGrid)
	{
		assert(myLiquidSurface.isGridMatched(viscosityGrid));
		myViscosity = viscosityGrid;
		myDoSolveViscosity = true;
	}

	void setViscosity(double constantViscosity = 1.)
	{
		myViscosity = ScalarGrid<double>(myLiquidSurface.xform(), myLiquidSurface.size(), constantViscosity);
		myDoSolveViscosity = true;
	}

	void unionLiquidSurface(const LevelSet& addedLiquidSurface);

	template<typename ForceSampler>
	void addForce(double dt, const ForceSampler& force);

	void addForce(double dt, const Vec2d& force);

	void advectOldPressure(const double dt);
	void advectLiquidSurface(double dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectViscosity(double dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER, InterpolationOrder interpolator = InterpolationOrder::LINEAR);
	void advectLiquidVelocity(double dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(double dt);

	// Useful for CFL
	double maxVelocityMagnitude() { return myLiquidVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer) const;

	void drawVolumetricSurface(Renderer& renderer) const;

	void drawLiquidSurface(Renderer& renderer);
	void drawLiquidVelocity(Renderer& renderer, double length) const;

	void drawSolidSurface(Renderer& renderer);
	void drawSolidVelocity(Renderer& renderer, double length) const;

private:

	// Simulation containers
	VectorGrid<double> myLiquidVelocity, mySolidVelocity;
	LevelSet myLiquidSurface, mySolidSurface;
	ScalarGrid<double> myViscosity;

	Transform myXform;

	bool myDoSolveViscosity;
	double myCFL;

	ScalarGrid<double> myOldPressure;
};

}

#endif