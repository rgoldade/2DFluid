#pragma once
#include <limits>
#include <vector>

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
	EulerianSmoke(const Transform& xform, Vec2ui size, Real ambienttemp = 300)
		: myXform(xform), myAmbientTemperature(ambienttemp)
	{
		myVelocity = VectorGrid<Real>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		myCollisionVelocity = VectorGrid<Real>(myXform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		myCollision = LevelSet2D(myXform, size, 5);

		mySmokeDensity = ScalarGrid<Real>(myXform, size, 0);
		mySmokeTemperature = ScalarGrid<Real>(myXform, size, myAmbientTemperature);
	}

	void setCollisionVolume(const LevelSet2D& collision);
	void setCollisionVelocity(const VectorGrid<Real>& collision_vel);

	void setSmokeVelocity(const VectorGrid<Real>& vel);

	void setSmokeSource(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature);

	void advectSmoke(Real dt, const InterpolationOrder& order);
	void advectVelocity(Real dt, const InterpolationOrder& order);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& renderer);

	// Useful for CFL
	Real maxVelocityMagnitude() { return myVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer) const;
	void drawSmoke(Renderer& renderer, Real maxDensity);
	void drawCollisionSurface(Renderer& renderer);
	void drawCollisionVelocity(Renderer& renderer, Real length) const;
	void drawVelocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> myVelocity, myCollisionVelocity;
	LevelSet2D myCollision;
	ScalarGrid<Real> mySmokeDensity, mySmokeTemperature;

	Real myAmbientTemperature;

	Transform myXform;
};