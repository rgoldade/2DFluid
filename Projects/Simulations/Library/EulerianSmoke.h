#ifndef SIMULATIONS_EULERIANSMOKE_H
#define SIMULATIONS_EULERIANSMOKE_H

#include <vector>

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
// EulerianSmoke.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid smoke simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

class EulerianSmoke
{
public:
	EulerianSmoke(const Transform& xform, Vec2i size, Real ambienttemp = 300)
		: myXform(xform), myAmbientTemperature(ambienttemp)
	{
		myVelocity = VectorGrid<Real>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<Real>(myXform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		mySolidSurface = LevelSet(myXform, size, 5);

		mySmokeDensity = ScalarGrid<Real>(myXform, size, 0);
		mySmokeTemperature = ScalarGrid<Real>(myXform, size, myAmbientTemperature);

		myOldPressure = ScalarGrid<Real>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<Real>& solidVelocity);

	void setFluidVelocity(const VectorGrid<Real>& vel);

	void setSmokeSource(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature);

	void advectFluidMaterial(const Real dt, const InterpolationOrder order);
	void advectFluidVelocity(const Real dt, const InterpolationOrder order);
	void advectOldPressure(const Real dt, const InterpolationOrder order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(const Real dt, Renderer& renderer);

	// Useful for CFL
	Real maxVelocityMagnitude() { return myVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer) const;
	void drawFluidDensity(Renderer& renderer, Real maxDensity);
	void drawFluidVelocity(Renderer& renderer, Real length) const;

	void drawSolidSurface(Renderer& renderer);
	void drawSolidVelocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> myVelocity, mySolidVelocity;
	LevelSet mySolidSurface;
	ScalarGrid<Real> mySmokeDensity, mySmokeTemperature;

	Real myAmbientTemperature;

	Transform myXform;

	ScalarGrid<Real> myOldPressure;
};

#endif