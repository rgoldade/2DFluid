#include <iostream>

#include "EulerianLiquid.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "GeometricPressureProjection.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

void EulerianLiquid::drawGrid(Renderer& renderer) const
{
	myLiquidSurface.drawGrid(renderer);
}

void EulerianLiquid::drawVolumetricSurface(Renderer &renderer) const
{
	ScalarGrid<Real> centerAreas = computeSuperSampledAreas(myLiquidSurface, ScalarGridSettings::SampleType::CENTER, 3);

	Vec2R nodeOffset[4] = { Vec2R(-.51), Vec2R(-.51, .51), Vec2R(.51), Vec2R(.51, -.51) };

	forEachVoxelRange(Vec2i(0), myLiquidSurface.size(), [&](const Vec2i& cell)
	{
		if (centerAreas(cell) > 0)
		{
			std::vector<Vec2R> quadVertex(4);
			std::vector<Vec4i> quadFace(1);
			std::vector<Vec3f> quadColour(1);

			for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
			{
				quadVertex[nodeIndex] = myLiquidSurface.indexToWorld(Vec2R(cell) + nodeOffset[nodeIndex]);
				quadFace[0][nodeIndex] = nodeIndex;
			}

			Real s = centerAreas(cell);
			quadColour[0] = (1 - s) * Vec3f(1) + s * Vec3f(0,1,1);
			renderer.addQuads(quadVertex, quadFace, quadColour);
		}
	});
}

void EulerianLiquid::drawLiquidSurface(Renderer& renderer)
{
	myLiquidSurface.drawSurface(renderer, Vec3f(0., 0., 1.0), 3);
}

void EulerianLiquid::drawLiquidVelocity(Renderer& renderer, Real length) const
{
	myLiquidVelocity.drawSamplePointVectors(renderer, Vec3f(0), myLiquidVelocity.dx() * length);
}

void EulerianLiquid::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3f(0), 3);
}

void EulerianLiquid::drawSolidVelocity(Renderer& renderer, Real length) const
{
	mySolidVelocity.drawSamplePointVectors(renderer, Vec3f(0,1,0), mySolidVelocity.dx() * length);
}

// Incoming solid surface must already be inverted
void EulerianLiquid::setSolidSurface(const LevelSet& solidSurface)
{
    assert(solidSurface.isBackgroundNegative());

    EdgeMesh localMesh = solidSurface.buildDCMesh();
    
    mySolidSurface.setBackgroundNegative();
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
		int totalFaces = myLiquidVelocity.size(axis)[0] * myLiquidVelocity.size(axis)[1];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaces, tbbGrainSize), [&](tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = myLiquidVelocity.grid(axis).unflatten(flatIndex);

				Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
				myLiquidVelocity(face, axis) = velocity.interp(facePosition, axis);
			}
		});
	}
}

void EulerianLiquid::setSolidVelocity(const VectorGrid<Real>& solidVelocity)
{
	for (int axis : {0, 1})
	{
		int totalFaces = mySolidVelocity.size(axis)[0] * mySolidVelocity.size(axis)[1];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaces, tbbGrainSize), [&](tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = mySolidVelocity.grid(axis).unflatten(flatIndex);

				Vec2R facePosition = mySolidVelocity.indexToWorld(Vec2R(face), axis);
				mySolidVelocity(face, axis) = solidVelocity.interp(facePosition, axis);
			}
		});
	}
}

void EulerianLiquid::unionLiquidSurface(const LevelSet& addedLiquidSurface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (int axis : {0, 1})
	{
		int totalFaces = myLiquidVelocity.size(axis)[0] * myLiquidVelocity.size(axis)[1];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaces, tbbGrainSize), [&](tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = myLiquidVelocity.grid(axis).unflatten(flatIndex);

				Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
				if (addedLiquidSurface.interp(facePosition) <= 0. && myLiquidSurface.interp(facePosition) > 0.)
					myLiquidVelocity(face, axis) = 0;
			}
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
		int totalFaces = myLiquidVelocity.size(axis)[0] * myLiquidVelocity.size(axis)[1];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaces, tbbGrainSize), [&](tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = myLiquidVelocity.grid(axis).unflatten(flatIndex);

				Vec2R facePosition = myLiquidVelocity.indexToWorld(Vec2R(face), axis);
				myLiquidVelocity(face, axis) = myLiquidVelocity(face, axis) + dt * force(facePosition, axis);
			}
		});
	}
}

void EulerianLiquid::addForce(Real dt, const Vec2R& force)
{
	addForce(dt, [&](Vec2R, int axis) {return force[axis]; });
}

void EulerianLiquid::advectOldPressure(const Real dt)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> pressureAdvector(myOldPressure);

		ScalarGrid<Real> tempPressure(myOldPressure.xform(), myOldPressure.size());

		pressureAdvector.advectField(dt, tempPressure, velocityFunc, IntegrationOrder::RK3, InterpolationOrder::LINEAR);

		std::swap(myOldPressure, tempPressure);
	}
}

void EulerianLiquid::advectLiquidSurface(Real dt, IntegrationOrder integrator)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myLiquidVelocity.interp(pos);  };
	EdgeMesh localMesh = myLiquidSurface.buildDCMesh();
	localMesh.advect(dt, velocityFunc, integrator);
	assert(localMesh.unitTestMesh());

	myLiquidSurface.initFromMesh(localMesh, false);

	// Remove solid regions from liquid surface
	const int totalVoxels = myLiquidSurface.size()[0] * myLiquidSurface.size()[1];
	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = myLiquidSurface.unflatten(flatIndex);
			myLiquidSurface(cell) = std::max(myLiquidSurface(cell), -mySolidSurface(cell));
		}
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

	const int totalVoxels = myLiquidSurface.size()[0] * myLiquidSurface.size()[1];
	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = myLiquidSurface.unflatten(flatIndex);

			if (mySolidSurface(cell) <= 0)
				extrapolatedSurface(cell) -= dx;
		}
	});

	extrapolatedSurface.reinitMesh();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	VectorGrid<Real> cutCellWeights = computeCutCellWeights(mySolidSurface, true);
	VectorGrid<Real> ghostFluidWeights = computeGhostFluidWeights(extrapolatedSurface);

	std::cout << "  Compute weights: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	// Initialize and call pressure projection
	PressureProjection projectDivergence(extrapolatedSurface,
											cutCellWeights,
											ghostFluidWeights,
											mySolidVelocity);

	projectDivergence.setInitialGuess(myOldPressure);
	projectDivergence.project(myLiquidVelocity);
	
	myOldPressure = projectDivergence.getPressureGrid();

	const VectorGrid<MarkedCells>& validFaces = projectDivergence.getValidFaces();
	
	if (myDoSolveViscosity)
	{
		std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		ViscositySolver(dt, extrapolatedSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity, myViscosity);

		std::cout << "  Solve for viscosity: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		// Initialize and call pressure projection
		projectDivergence.disableInitialGuess();
		projectDivergence.project(myLiquidVelocity);

		std::cout << "  Solve for pressure after viscosity: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();
	}
	else
	{
	    std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
	    simTimer.reset();
	}

	// Extrapolate velocity
	for (int axis : {0, 1})
	{
		ExtrapolateField<ScalarGrid<Real>> extrapolator(myLiquidVelocity.grid(axis));
		extrapolator.extrapolate(validFaces.grid(axis), 1.5 * myCFL);
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	advectOldPressure(dt);
	advectLiquidSurface(dt, IntegrationOrder::RK3);
	
	if (myDoSolveViscosity)
		advectViscosity(dt, IntegrationOrder::FORWARDEULER);

	advectLiquidVelocity(dt, IntegrationOrder::RK3);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
