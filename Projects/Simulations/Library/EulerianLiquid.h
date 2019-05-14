#ifndef SIMULATIONS_EULERIANLIQUID_H
#define SIMULATIONS_EULERIANLIQUID_H

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
	EulerianLiquid(const Transform& xform, Vec2ui size, Real cfl = 5.)
		: myXform(xform)
		, myDoSolveViscosity(false)
		, mySurfaceTensionScale(0.)
		, myCFL(cfl)
	{
		myVelocity = VectorGrid<Real>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		myCollisionVelocity = VectorGrid<Real>(myXform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		mySurface = LevelSet2D(myXform, size, myCFL);
		myCollisionSurface = LevelSet2D(myXform, size, myCFL);
	}

	void setCollisionVolume(const LevelSet2D& collision);
	void setCollisionVelocity(const VectorGrid<Real>& collisionVelocity);
	void setSurfaceVolume(const LevelSet2D& surface);
	void setSurfaceVelocity(const VectorGrid<Real>& velocity);
	void setSurfaceTension(Real surfaceTensionScale)
	{
		mySurfaceTensionScale = surfaceTensionScale;
	}

	void setViscosity(const ScalarGrid<Real>& viscosityGrid)
	{
		assert(mySurface.isMatched(viscosityGrid));
		myViscosity = viscosityGrid;
		myDoSolveViscosity = true;
	}

	void setViscosity(Real constantViscosity = 1.)
	{
		myViscosity = ScalarGrid<Real>(mySurface.xform(), mySurface.size(), constantViscosity);
		myDoSolveViscosity = true;
	}

	void addSurfaceVolume(const LevelSet2D& surface);

	template<typename ForceSampler>
	void addForce(Real dt, const ForceSampler& force);
	
	void addForce(Real dt, const Vec2R& force);

	void advectSurface(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectViscosity(Real dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER, InterpolationOrder interpolator = InterpolationOrder::LINEAR);
	void advectVelocity(Real dt, IntegrationOrder integrator = IntegrationOrder::RK3, InterpolationOrder interpolator = InterpolationOrder::LINEAR);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(Real dt, Renderer& debugRenderer);

	// Useful for CFL
	Real maxVelocityMagnitude() { return myVelocity.maxMagnitude(); }
	
	// Rendering tools
	void drawGrid(Renderer& renderer) const;
	void drawSurface(Renderer& renderer);
	void drawCollisionSurface(Renderer& renderer);
	void drawCollisionVelocity(Renderer& renderer, Real length) const;
	void drawVelocity(Renderer& renderer, Real length) const;

private:

	// Simulation containers
	VectorGrid<Real> myVelocity, myCollisionVelocity;
	LevelSet2D mySurface, myCollisionSurface;
	ScalarGrid<Real> myViscosity;

	Transform myXform;

	bool myDoSolveViscosity;
	Real mySurfaceTensionScale, myCFL;
};

#endif