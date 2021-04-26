#include "EulerianLiquidSimulator.h"

#include <iostream>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "FieldAdvector.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

namespace FluidSim2D
{

void EulerianLiquidSimulator::drawGrid(Renderer& renderer) const
{
	myLiquidSurface.drawGrid(renderer, false);
}

void EulerianLiquidSimulator::drawVolumetricSurface(Renderer& renderer) const
{
	ScalarGrid<double> centerAreas = computeSupersampledAreas(myLiquidSurface, ScalarGridSettings::SampleType::CENTER, 3);

	Vec2d nodeOffset[4] = { Vec2d(-.51, -.51), Vec2d(-.51, .51), Vec2d(.51, .51), Vec2d(.51, -.51) };

	forEachVoxelRange(Vec2i::Zero(), myLiquidSurface.size(), [&](const Vec2i& cell)
	{
		if (centerAreas(cell) > 0)
		{
			VecVec2d quadVertex(4);
			VecVec4i quadFace(1);
			VecVec3d quadColour(1);

			for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
			{
				quadVertex[nodeIndex] = myLiquidSurface.indexToWorld(cell.cast<double>() + nodeOffset[nodeIndex]);
				quadFace[0][nodeIndex] = nodeIndex;
			}

			double s = centerAreas(cell);
			quadColour[0] = (1 - s) * Vec3d::Ones() + s * Vec3d(0, 1, 1);
			renderer.addQuadFaces(quadVertex, quadFace, quadColour);
		}
	});
}

void EulerianLiquidSimulator::drawLiquidSurface(Renderer& renderer)
{
	myLiquidSurface.drawSurface(renderer, Vec3d(0, 0, 1), 3);
}

void EulerianLiquidSimulator::drawLiquidVelocity(Renderer& renderer, double length) const
{
	myLiquidVelocity.drawSamplePointVectors(renderer, Vec3d::Zero(), myLiquidVelocity.dx() * length);
}

void EulerianLiquidSimulator::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3d::Zero(), 3);
}

void EulerianLiquidSimulator::drawSolidVelocity(Renderer& renderer, double length) const
{
	mySolidVelocity.drawSamplePointVectors(renderer, Vec3d(0, 1, 0), mySolidVelocity.dx() * length);
}

// Incoming solid surface must already be inverted
void EulerianLiquidSimulator::setSolidSurface(const LevelSet& solidSurface)
{
	assert(solidSurface.isBackgroundNegative());

	EdgeMesh localMesh = solidSurface.buildDCMesh();

	mySolidSurface.setBackgroundNegative();
	mySolidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquidSimulator::setSolidVelocity(const VectorGrid<double>& solidVelocity)
{
	assert(mySolidVelocity.isGridMatched(solidVelocity));

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, mySolidVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = solidVelocity.grid(axis).unflatten(faceIndex);
				mySolidVelocity(face, axis) = solidVelocity(face, axis);
			}
		});
	}
}

void EulerianLiquidSimulator::setLiquidSurface(const LevelSet& surface)
{
	EdgeMesh localMesh = surface.buildDCMesh();
	myLiquidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquidSimulator::setLiquidVelocity(const VectorGrid<double>& velocity)
{
	assert(myLiquidVelocity.isGridMatched(velocity));

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = velocity.grid(axis).unflatten(faceIndex);
					myLiquidVelocity(face, axis) = velocity(face, axis);
				}
			});
	}
}

void EulerianLiquidSimulator::unionLiquidSurface(const LevelSet& addedLiquidSurface)
{
	// Need to zero out velocity in this added region as it could get extrapolated values
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = myLiquidVelocity.grid(axis).unflatten(faceIndex);

					Vec2d facePosition = myLiquidVelocity.indexToWorld(face.cast<double>(), axis);
					if (addedLiquidSurface.biLerp(facePosition) <= 0 && myLiquidSurface.biLerp(facePosition) > 0)
						myLiquidVelocity(face, axis) = 0;
				}
			});
	}

	// Combine surfaces
	myLiquidSurface.unionSurface(addedLiquidSurface);
	myLiquidSurface.reinitMesh();
}

template<typename ForceSampler>
void EulerianLiquidSimulator::addForce(double dt, const ForceSampler& force)
{
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount()), [&](tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i face = myLiquidVelocity.grid(axis).unflatten(cellIndex);

				Vec2d facePosition = myLiquidVelocity.indexToWorld(face.cast<double>(), axis);
				myLiquidVelocity(face, axis) += dt * force(facePosition, axis);
			}
		});
	}
}

void EulerianLiquidSimulator::addForce(double dt, const Vec2d& force)
{
	addForce(dt, [&](Vec2d, int axis) {return force[axis]; });
}

void EulerianLiquidSimulator::advectOldPressure(const double dt)
{
	auto velocityFunc = [&](double, const Vec2d& pos) { return myLiquidVelocity.biLerp(pos); };

	ScalarGrid<double> tempPressure(myOldPressure.xform(), myOldPressure.size());
	advectField(dt, tempPressure, myOldPressure, velocityFunc, IntegrationOrder::RK3, InterpolationOrder::LINEAR);

	std::swap(myOldPressure, tempPressure);
}

void EulerianLiquidSimulator::advectLiquidSurface(double dt, IntegrationOrder integrator)
{
	auto velocityFunc = [&](double, const Vec2d& pos) { return myLiquidVelocity.biLerp(pos);  };

	EdgeMesh localMesh = myLiquidSurface.buildDCMesh();
	localMesh.advectMesh(dt, velocityFunc, integrator);
	assert(localMesh.unitTestMesh());

	myLiquidSurface.initFromMesh(localMesh, false);

	// Remove solid regions from liquid surface
	tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidSurface.voxelCount()), [&](tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = myLiquidSurface.unflatten(cellIndex);
			myLiquidSurface(cell) = std::max(myLiquidSurface(cell), -mySolidSurface(cell));
		}
	});

	myLiquidSurface.reinitMesh();
}

void EulerianLiquidSimulator::advectViscosity(double dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](double, const Vec2d& pos) { return myLiquidVelocity.biLerp(pos); };

	ScalarGrid<double> tempViscosity(myViscosity.xform(), myViscosity.size());

	advectField(dt, tempViscosity, myViscosity, velocityFunc, integrator, interpolator);

	std::swap(tempViscosity, myViscosity);
}

void EulerianLiquidSimulator::advectLiquidVelocity(double dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	auto velocityFunc = [&](double, const Vec2d& pos) { return myLiquidVelocity.biLerp(pos); };

	VectorGrid<double> tempVelocity(myLiquidVelocity.xform(), myLiquidVelocity.gridSize(), VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
		advectField(dt, tempVelocity.grid(axis), myLiquidVelocity.grid(axis), velocityFunc, integrator, interpolator);

	std::swap(myLiquidVelocity, tempVelocity);
}

void EulerianLiquidSimulator::runTimestep(double dt)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	LevelSet extrapolatedSurface = myLiquidSurface;

	double dx = extrapolatedSurface.dx();

	tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidSurface.voxelCount()), [&](tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = myLiquidSurface.unflatten(cellIndex);

			if (mySolidSurface(cell) <= 0 ||
				(mySolidSurface(cell) <= dx && myLiquidSurface(cell) <= 0))
				extrapolatedSurface(cell) -= dx;
		}
	});

	extrapolatedSurface.reinitMesh();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	VectorGrid<double> cutCellWeights = computeCutCellWeights(mySolidSurface, true);
	VectorGrid<double> ghostFluidWeights = computeGhostFluidWeights(extrapolatedSurface);

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

	const VectorGrid<VisitedCellLabels>& validFaces = projectDivergence.getValidFaces();

	if (myDoSolveViscosity)
	{
		// Extrapolate velocity
		for (int axis : {0, 1})
		{
			// Zero out non-valid faces
			tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount()), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = validFaces.grid(axis).unflatten(faceIndex);

					if (validFaces(face, axis) != VisitedCellLabels::FINISHED_CELL)
						myLiquidVelocity(face, axis) = 0;
				}
			});
		}

		{
			double velMag = myLiquidVelocity.maxMagnitude();

			for (int axis : {0, 1})
				extrapolateField(myLiquidVelocity.grid(axis), validFaces.grid(axis), int(2. * velMag * dt / myLiquidSurface.dx()));
		}

		std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

		ViscositySolver(dt, myLiquidSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity, myViscosity);

		std::cout << "  Solve for viscosity: " << simTimer.stop() << "s" << std::endl;
		simTimer.reset();

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
		// Zero out non-valid faces
		tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = validFaces.grid(axis).unflatten(faceIndex);

					if (validFaces(face, axis) != VisitedCellLabels::FINISHED_CELL)
						myLiquidVelocity(face, axis) = 0;
				}
			});
	}

	{
		double velMag = myLiquidVelocity.maxMagnitude();

		for (int axis : {0, 1})
			extrapolateField(myLiquidVelocity.grid(axis), validFaces.grid(axis), int(2. * velMag * dt / myLiquidSurface.dx()));
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

}