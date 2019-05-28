#include <iostream>

#include "EulerianLiquid.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

void EulerianLiquid::drawGrid(Renderer& renderer) const
{
	myLiquidSurface.drawGrid(renderer);
}

void EulerianLiquid::drawLiquidSurface(Renderer& renderer)
{
	myLiquidSurface.drawSurface(renderer, Vec3f(0., 0., 1.0));
}

void EulerianLiquid::drawLiquidVelocity(Renderer& renderer, Real length) const
{
	myLiquidVelocity.drawSamplePointVectors(renderer, Vec3f(0), myLiquidVelocity.dx() * length);
}

void EulerianLiquid::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3f(1.,0.,1.));
}

void EulerianLiquid::drawSolidVelocity(Renderer& renderer, Real length) const
{
	mySolidVelocity.drawSamplePointVectors(renderer, Vec3f(0,1,0), mySolidVelocity.dx() * length);
}

// Incoming solid surface must already be inverted
void EulerianLiquid::setSolidSurface(const LevelSet& solidSurface)
{
    assert(solidSurface.isBoundaryNegative());

    EdgeMesh localMesh = solidSurface.buildDCMesh();
    
    mySolidSurface.setBoundaryNegative();
    mySolidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquid::setLiquidSurface(const LevelSet& surface)
{
	EdgeMesh localMesh = surface.buildDCMesh();
	myLiquidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquid::setLiquidVelocity(const VectorGrid<Real>& velocity)
{
	for (int axis : {0, 1})
	{
		Vec2i size = myLiquidVelocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
			myLiquidVelocity(face, axis) = velocity.interp(facePosition, axis);
		});
	}
}

void EulerianLiquid::setSolidVelocity(const VectorGrid<Real>& solidVelocity)
{
	for (int axis : {0, 1})
	{
		Vec2i size = mySolidVelocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2R facePosition = mySolidVelocity.indexToWorld(Vec2R(face), axis);
			mySolidVelocity(face, axis) = solidVelocity.interp(facePosition, axis);
		});
	}
}

void EulerianLiquid::unionLiquidSurface(const LevelSet& addedLiquidSurface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), myLiquidVelocity.size(axis), [&](const Vec2i& face)
		{
			Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
			if (addedLiquidSurface.interp(facePosition) <= 0. && myLiquidSurface.interp(facePosition) > 0.)
				myLiquidVelocity(face, axis) = 0;
		});
	}

	// Combine surfaces
	myLiquidSurface.unionSurface(addedLiquidSurface);
	myLiquidSurface.reinitMesh();
}

template<typename ForceSampler>
void EulerianLiquid::addForce(Real dt, const ForceSampler& force)
{
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), myLiquidVelocity.size(axis), [&](const Vec2i& face)
		{
			Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
			myLiquidVelocity(face, axis) = myLiquidVelocity(face, axis) + dt * force(facePosition, axis);
		});
	}
}

void EulerianLiquid::addForce(Real dt, const Vec2R& force)
{
	addForce(dt, [&](Vec2R, int axis) {return force[axis]; });
}

void EulerianLiquid::advectOldPressure(const Real dt, const InterpolationOrder order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> pressureAdvector(myOldPressure);

		ScalarGrid<Real> tempPressure(myOldPressure.xform(), myOldPressure.size());

		pressureAdvector.advectField(dt, tempPressure, velocityFunc, IntegrationOrder::RK3, order);

		std::swap(myOldPressure, tempPressure);
	}
}

void EulerianLiquid::advectLiquidSurface(Real dt, IntegrationOrder integrator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos);  };
	EdgeMesh localMesh = myLiquidSurface.buildDCMesh();
	localMesh.advect(dt, velocityFunc, integrator);
	assert(localMesh.unitTest());

	myLiquidSurface.initFromMesh(localMesh, false);

	// Remove solid regions from liquid surface
	forEachVoxelRange(Vec2i(0), myLiquidSurface.size(), [&](const Vec2i& cell)
	{
		myLiquidSurface(cell) = std::max(myLiquidSurface(cell), -mySolidSurface(cell));
	});

	myLiquidSurface.reinitMesh();
}

void EulerianLiquid::advectViscosity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos); };

	AdvectField<ScalarGrid<Real>> advector(myViscosity);
	ScalarGrid<Real> tempViscosity(myViscosity.xform(), myViscosity.size());
	
	advector.advectField(dt, tempViscosity, velocityFunc, integrator, interpolator);
	std::swap(tempViscosity, myViscosity);
}

void EulerianLiquid::advectLiquidVelocity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos); };

	VectorGrid<Real> tempVelocity(myLiquidVelocity.xform(), myLiquidVelocity.gridSize(), VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		AdvectField<ScalarGrid<Real>> advector(myLiquidVelocity.grid(axis));
		advector.advectField(dt, tempVelocity.grid(axis), velocityFunc, integrator, interpolator);
	}

	std::swap(myLiquidVelocity, tempVelocity);
}

void EulerianLiquid::runTimestep(Real dt, Renderer& debugRenderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	LevelSet extrapolatedSurface = myLiquidSurface;

	Real dx = extrapolatedSurface.dx();
	forEachVoxelRange(Vec2i(0), extrapolatedSurface.size(), [&](const Vec2i& cell)
	{
		if (mySolidSurface(cell) <= 0)
			extrapolatedSurface(cell) -= dx;
	});

	extrapolatedSurface.reinitMesh();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> ghostFluidWeights = computeGhostFluidWeights(extrapolatedSurface);
	VectorGrid<Real> cutCellWeights = computeCutCellWeights(mySolidSurface, true);

	std::cout << "  Compute weights: " << simTimer.stop() << "s" << std::endl;
	
	simTimer.reset();

	// Initialize and call pressure projection
	PressureProjection projectDivergence(extrapolatedSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity);

	projectDivergence.setInitialGuess(myOldPressure);
	projectDivergence.project(ghostFluidWeights, cutCellWeights);
	myOldPressure = projectDivergence.getPressureGrid();

	// Update velocity field
	projectDivergence.applySolution(myLiquidVelocity, ghostFluidWeights);

	VectorGrid<MarkedCells> valid(extrapolatedSurface.xform(), extrapolatedSurface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
	
	if (myDoSolveViscosity)
	{
		std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();
		
		int samples = 3;

		ScalarGrid<Real> centerAreas = computeSuperSampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::CENTER, 3);
		ScalarGrid<Real> nodeAreas = computeSuperSampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::NODE, 3);
		VectorGrid<Real> faceAreas = computeSuperSampledFaceAreas(extrapolatedSurface, 3);
		
		ScalarGrid<Real> solidCenterAreas = computeSuperSampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::CENTER, 3);
		ScalarGrid<Real> solidNodeAreas = computeSuperSampledAreas(extrapolatedSurface, ScalarGridSettings::SampleType::NODE, 3);

		std::cout << "  Compute viscosity weights: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		ViscositySolver viscosity(dt, extrapolatedSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity);

		viscosity.setViscosity(myViscosity);

		viscosity.solve(faceAreas, centerAreas, nodeAreas, solidCenterAreas, solidNodeAreas);

		std::cout << "  Solve for viscosity: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		// Initialize and call pressure projection		
		PressureProjection projectDivergence2(extrapolatedSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity);

		projectDivergence2.project(ghostFluidWeights, cutCellWeights);

		// Update velocity field
		projectDivergence2.applySolution(myLiquidVelocity, ghostFluidWeights);

		projectDivergence2.applyValid(valid);

		std::cout << "  Solve for pressure after viscosity: " << simTimer.stop() << "s" << std::endl;
		
		simTimer.reset();
	}
	else
	{
	    // Label solved faces
	    projectDivergence.applyValid(valid);

	    std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
	    simTimer.reset();
	}

	// Extrapolate velocity
	for (int axis : {0, 1})
	{
		ExtrapolateField<ScalarGrid<Real>> extrapolator(myLiquidVelocity.grid(axis));
		extrapolator.extrapolate(valid.grid(axis), 1.5 * myCFL);
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	advectOldPressure(dt, InterpolationOrder::LINEAR);
	advectLiquidSurface(dt, IntegrationOrder::RK3);
	advectLiquidVelocity(dt, IntegrationOrder::RK3);

	if(myDoSolveViscosity)
		advectViscosity(dt, IntegrationOrder::FORWARDEULER);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
