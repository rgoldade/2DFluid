#ifndef SIMULATIONS_EULERIANLIQUID_H
#define SIMULATIONS_EULERIANLIQUID_H

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
// EulerianLiquid.h/cpp
// Ryan Goldade 2016
//
// Wrapper class around the staggered MAC grid liquid simulator 
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

class EulerianLiquid
{
public:
	EulerianLiquid(const Transform& xform, Vec2i size, Real cfl = 5.)
		: myXform(xform)
		, myDoSolveViscosity(false)
		, myCFL(cfl)
	{
		myLiquidVelocity = VectorGrid<Real>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<Real>(myXform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		myLiquidSurface = LevelSet(myXform, size, myCFL);
		mySolidSurface = LevelSet(myXform, size, myCFL);

		myOldPressure = ScalarGrid<Real>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<Real>& solidVelocity);
	void setLiquidSurface(const LevelSet& liquidSurface);
	void setLiquidVelocity(const VectorGrid<Real>& liquidVelocity);

	void setViscosity(const ScalarGrid<Real>& viscosityGrid)
	{
		assert(myLiquidSurface.isGridMatched(viscosityGrid));
		myViscosity = viscosityGrid;
		myDoSolveViscosity = true;
	}

	void setViscosity(Real constantViscosity = 1.)
	{
		myViscosity = ScalarGrid<Real>(myLiquidSurface.xform(), myLiquidSurface.size(), constantViscosity);
		myDoSolveViscosity = true;
	}

	void unionLiquidSurface(const LevelSet& addedLiquidSurface);

	template<typename ForceSampler>
	void addForce(Real dt, const ForceSampler& force);
	
	void addForce(Real dt, const Vec2R& force);

	void advectOldPressure(const Real dt);
	void advectLiquidSurface(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectViscosity(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER, InterpolationOrder interpolator = InterpolationOrder::LINEAR);
	void advectLiquidVelocity(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& debugRenderer);

	// Useful for CFL
	Real maxVelocityMagnitude() { return myLiquidVelocity.maxMagnitude(); }
	
	// Rendering tools
	void drawGrid(Renderer& renderer) const;

	void drawLiquidSurface(Renderer& renderer);
	void drawLiquidVelocity(Renderer& renderer, Real length) const;

	void drawSolidSurface(Renderer& renderer);
	void drawSolidVelocity(Renderer& renderer, Real length) const;
	
private:

	// Simulation containers
	VectorGrid<Real> myLiquidVelocity, mySolidVelocity;
	LevelSet myLiquidSurface, mySolidSurface;
	ScalarGrid<Real> myViscosity;

	Transform myXform;

	bool myDoSolveViscosity;
	Real myCFL;

	ScalarGrid<Real> myOldPressure;
};

#endif