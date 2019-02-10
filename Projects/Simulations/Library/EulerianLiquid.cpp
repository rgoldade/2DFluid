#include <iostream>

#include "EulerianLiquid.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

void EulerianLiquid::drawGrid(Renderer& renderer) const
{
	mySurface.drawGrid(renderer);
}

void EulerianLiquid::drawSurface(Renderer& renderer)
{
	mySurface.drawSurface(renderer, Vec3f(0., 0., 1.0));
}

void EulerianLiquid::drawCollisionSurface(Renderer& renderer)
{
	myCollisionSurface.drawSurface(renderer, Vec3f(1.,0.,1.));
}

void EulerianLiquid::drawCollisionVelocity(Renderer& renderer, Real length) const
{
	myCollisionVelocity.drawSamplePointVectors(renderer, Vec3f(0,1,0), myCollisionVelocity.dx() * length);
}

void EulerianLiquid::drawVelocity(Renderer& renderer, Real length) const
{
	myVelocity.drawSamplePointVectors(renderer, Vec3f(0), myVelocity.dx() * length);
}

// Incoming collision volume must already be inverted
void EulerianLiquid::setCollisionVolume(const LevelSet2D& collision)
{
    assert(collision.inverted());

    Mesh2D tempMesh = collision.buildDCMesh();
    
    // TODO : consider making collision node sampled
    myCollisionSurface.setInverted();
    myCollisionSurface.init(tempMesh, false);
}

void EulerianLiquid::setSurfaceVolume(const LevelSet2D& surface)
{
	Mesh2D tempMesh = surface.buildDCMesh();
	mySurface.init(tempMesh, false);
}

void EulerianLiquid::setSurfaceVelocity(const VectorGrid<Real>& velocity)
{
	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R facePosition = myVelocity.indexToWorld(Vec2R(face), axis);
			myVelocity(face, axis) = velocity.interp(facePosition, axis);
		});
	}
}

void EulerianLiquid::setCollisionVelocity(const VectorGrid<Real>& collisionVelocity)
{
	for (auto axis : { 0,1 })
	{
		Vec2ui size = myCollisionVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R facePosition = myCollisionVelocity.indexToWorld(Vec2R(face), axis);
			myCollisionVelocity(face, axis) = collisionVelocity.interp(facePosition, axis);
		});
	}
}

void EulerianLiquid::addSurfaceVolume(const LevelSet2D& surface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R facePosition = myVelocity.indexToWorld(Vec2R(face), axis);
			if (surface.interp(facePosition) <= 0. && mySurface.interp(facePosition) > 0.)
				myVelocity(face, axis) = 0;
		});
	}

	// Combine surfaces
	mySurface.unionSurface(surface);
	mySurface.reinitMesh();
}

template<typename ForceSampler>
void EulerianLiquid::addForce(Real dt, const ForceSampler& force)
{
	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R facePosition = myVelocity.indexToWorld(Vec2R(face), axis);
			myVelocity(face, axis) = myVelocity(face, axis) + dt * force(facePosition, axis);
		});
	}
}

void EulerianLiquid::addForce(Real dt, const Vec2R& force)
{
	addForce(dt, [&](Vec2R, unsigned axis) {return force[axis]; });
}

void EulerianLiquid::advectSurface(Real dt, IntegrationOrder integrator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos);  };
	Mesh2D tempMesh = mySurface.buildDCMesh();
	tempMesh.advect(dt, velocityFunc, integrator);
	assert(tempMesh.unitTest());

	mySurface.init(tempMesh, false);

	// Remove collision volumes from surface
	forEachVoxelRange(Vec2ui(0), mySurface.size(), [&](const Vec2ui& cell)
	{
		mySurface(cell) = std::max(mySurface(cell), -myCollisionSurface(cell));
	});

	mySurface.reinitMesh();
}

void EulerianLiquid::advectViscosity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos); };

	AdvectField<ScalarGrid<Real>> advector(myViscosity);
	ScalarGrid<Real> tempViscosity(myViscosity.xform(), myViscosity.size());
	
	advector.advectField(dt, tempViscosity, velocityFunc, integrator, interpolator);
	std::swap(tempViscosity, myViscosity);
}

void EulerianLiquid::advectVelocity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos); };

	VectorGrid<Real> tempVelocity(myVelocity.xform(), myVelocity.gridSize(), VectorGridSettings::SampleType::STAGGERED);

	for (auto axis : { 0,1 })
	{
		AdvectField<ScalarGrid<Real>> advector(myVelocity.grid(axis));
		advector.advectField(dt, tempVelocity.grid(axis), velocityFunc, integrator, interpolator);
	}

	std::swap(myVelocity, tempVelocity);
}

void EulerianLiquid::runTimestep(Real dt, Renderer& debugRenderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	LevelSet2D extrapolatedSurface = mySurface;

	Real dx = extrapolatedSurface.dx();
	forEachVoxelRange(Vec2ui(0), extrapolatedSurface.size(), [&](const Vec2ui& cell)
	{
		if (myCollisionSurface(cell) <= 0)
			extrapolatedSurface(cell) -= dx;
	});

	extrapolatedSurface.reinitMesh();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> ghostFluidWeights = computeGhostFluidWeights(extrapolatedSurface);
	VectorGrid<Real> cutCellWeights = computeCutCellWeights(myCollisionSurface, true);

	std::cout << "  Compute weights: " << simTimer.stop() << "s" << std::endl;
	
	simTimer.reset();

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, extrapolatedSurface, myVelocity, myCollisionSurface, myCollisionVelocity);

	//if (m_surfacetension_scale != 0.)
	//{
	//	const ScalarGrid<Real> surface_tension = m_surface.get_curvature();
	//	projectdivergence.set_surface_pressure(surface_tension, m_st_scale);
	//}

	projectdivergence.project(ghostFluidWeights, cutCellWeights);
	
	// Update velocity field
	projectdivergence.applySolution(myVelocity, ghostFluidWeights);

	VectorGrid<Real> valid(extrapolatedSurface.xform(), extrapolatedSurface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
	
	if (myDoSolveViscosity)
	{
		std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();
		
		unsigned samples = 3;

		ScalarGrid<Real> centerAreas = computeSupersampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::CENTER, 3);
		ScalarGrid<Real> nodeAreas = computeSupersampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::NODE, 3);
		VectorGrid<Real> faceAreas = computeSupersampledFaceAreas(extrapolatedSurface, 3);
		
		ScalarGrid<Real> collisionCenterAreas = computeSupersampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::CENTER, 3);
		ScalarGrid<Real> collisionNodeAreas = computeSupersampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::NODE, 3);

		std::cout << "  Compute viscosity weights: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		ViscositySolver viscosity(dt, extrapolatedSurface, myVelocity, myCollisionSurface, myCollisionVelocity);

		viscosity.setViscosity(myViscosity);

		viscosity.solve(faceAreas, centerAreas, nodeAreas, collisionCenterAreas, collisionNodeAreas);

		std::cout << "  Solve for viscosity: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		// Initialize and call pressure projection		
		PressureProjection projectdivergence2(dt, extrapolatedSurface, myVelocity, myCollisionSurface, myCollisionVelocity);

		projectdivergence.project(ghostFluidWeights, cutCellWeights);

		// Update velocity field
		projectdivergence.applySolution(myVelocity, ghostFluidWeights);

		projectdivergence.applyValid(valid);

		std::cout << "  Solve for pressure after viscosity: " << simTimer.stop() << "s" << std::endl;
		
		simTimer.reset();
	}
	else
	{
	    // Label solved faces
	    projectdivergence.applyValid(valid);

	    std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
	    simTimer.reset();
	}

	// Extrapolate velocity
	ExtrapolateField<VectorGrid<Real>> extrapolator(myVelocity);
	extrapolator.extrapolate(valid, 1.5 * myCFL);

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	advectSurface(dt, IntegrationOrder::RK3);
	advectVelocity(dt, IntegrationOrder::RK3);

	if(myDoSolveViscosity)
		advectViscosity(dt, IntegrationOrder::FORWARDEULER);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
