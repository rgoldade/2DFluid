#include "MultiMaterialLiquidSimulator.h"

#include <iostream>

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "MultiMaterialPressureProjection.h"
#include "Timer.h"

namespace FluidSim2D
{

void MultiMaterialLiquidSimulator::drawMaterialSurface(Renderer& renderer, int material)
{
	myFluidSurfaces[material].drawSurface(renderer, Vec3d(0., 0., 1.), 2.);
}

void MultiMaterialLiquidSimulator::drawMaterialVelocity(Renderer& renderer, double length, int material) const
{
	myFluidVelocities[material].drawSamplePointVectors(renderer, Vec3d::Zero(), myFluidVelocities[material].dx() * length);
}

void MultiMaterialLiquidSimulator::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3d::Zero(), 2);
}

template<typename ForceSampler>
void MultiMaterialLiquidSimulator::addForce(double dt, int material, const ForceSampler& force)
{
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myFluidVelocities[material].grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = myFluidVelocities[0].grid(axis).unflatten(faceIndex);
					Vec2d worldPosition = myFluidVelocities[0].indexToWorld(face.cast<double>(), axis);

					for (int material = 0; material < myMaterialCount; ++material)
						myFluidVelocities[material](face, axis) += dt * force(worldPosition, axis);
				}
			});
	}
}

void MultiMaterialLiquidSimulator::addForce(double dt, int material, const Vec2d& force)
{
	addForce(dt, material, [&](Vec2d, int axis) { return force[axis]; });
}

void MultiMaterialLiquidSimulator::advectFluidVelocities(double dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
	for (int material = 0; material < myMaterialCount; ++material)
	{
		auto velocityFunc = [&](double, const Vec2d& point) { return myFluidVelocities[material].biLerp(point); };

		VectorGrid<double> tempVelocity(myFluidVelocities[material].xform(), myFluidVelocities[material].gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

		for (int axis : {0, 1})
			advectField(dt, tempVelocity.grid(axis), myFluidVelocities[material].grid(axis), velocityFunc, integrator, interpolator);

		std::swap(myFluidVelocities[material], tempVelocity);
	}
}

void MultiMaterialLiquidSimulator::advectFluidSurfaces(double dt, IntegrationOrder integrator)
{
	for (int material = 0; material < myMaterialCount; ++material)
	{
		auto velocityFunc = [&](double, const Vec2d& point) { return myFluidVelocities[material].biLerp(point); };

		EdgeMesh localMesh = myFluidSurfaces[material].buildMSMesh();
		localMesh.advectMesh(dt, velocityFunc, integrator);
		myFluidSurfaces[material].initFromMesh(localMesh, false);
	}

	// Fix possible overlaps between the materials.
	tbb::parallel_for(tbb::blocked_range<int>(0, myFluidSurfaces[0].voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = myFluidSurfaces[0].unflatten(cellIndex);

				double firstMin = std::min(myFluidSurfaces[0](cell), mySolidSurface(cell));
				double secondMin = std::max(myFluidSurfaces[0](cell), mySolidSurface(cell));

				for (int material = 1; material < myMaterialCount; ++material)
				{
					double localMin = myFluidSurfaces[material](cell);
					if (localMin < firstMin)
					{
						secondMin = firstMin;
						firstMin = localMin;
					}
					else secondMin = std::min(localMin, secondMin);
				}

				double avgSDF = .5 * (firstMin + secondMin);

				for (int material = 0; material < myMaterialCount; ++material)
					myFluidSurfaces[material](cell) -= avgSDF;
			}
		});

	for (int material = 0; material < myMaterialCount; ++material)
		myFluidSurfaces[material].reinit();
}

void MultiMaterialLiquidSimulator::setSolidSurface(const LevelSet& solidSurface)
{
	assert(solidSurface.isBackgroundNegative());

	EdgeMesh localMesh = solidSurface.buildDCMesh();

	mySolidSurface.setBackgroundNegative();
	mySolidSurface.initFromMesh(localMesh, false /* don't resize grid*/);
}

void MultiMaterialLiquidSimulator::runTimestep(double dt)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	//
	// Extrapolate materials into solids
	//

	std::vector<LevelSet> extrapolatedSurfaces(myMaterialCount);

	for (int material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material] = myFluidSurfaces[material];

	double dx = mySolidSurface.dx();

	tbb::parallel_for(tbb::blocked_range<int>(0, myFluidSurfaces[0].voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = myFluidSurfaces[0].unflatten(cellIndex);

				for (int material = 0; material < myMaterialCount; ++material)
				{
					if (mySolidSurface(cell) <= 0. ||
						(mySolidSurface(cell) <= dx && myFluidSurfaces[material](cell) <= 0))
						extrapolatedSurfaces[material](cell) -= dx;
				}
			}
		});

	for (int material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material].reinit();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	MultiMaterialPressureProjection pressureSolver(extrapolatedSurfaces, myFluidDensities, mySolidSurface);

	pressureSolver.setInitialGuess(myOldPressure);
	pressureSolver.project(myFluidVelocities);

	myOldPressure = pressureSolver.getPressureGrid();

	std::cout << "  Solve for multi-material pressure: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Get valid faces for each material so we can extrapolate velocities.
	//

	for (int material = 0; material < myMaterialCount; ++material)
	{
		const VectorGrid<VisitedCellLabels>& validFaces = pressureSolver.getValidFaces(material);

		// Zero out-of-bounds velocity
		for (int axis : {0, 1})
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
					{
						Vec2i face = validFaces.grid(axis).unflatten(flatIndex);

						if (validFaces(face, axis) != VisitedCellLabels::FINISHED_CELL)
							myFluidVelocities[material](face, axis) = 0;
					}
				});
		}

		// Extrapolate velocity
		for (int axis : {0, 1})
			extrapolateField(myFluidVelocities[material].grid(axis), validFaces.grid(axis), 5);
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	advectFluidSurfaces(dt, IntegrationOrder::RK3);
	advectFluidVelocities(dt, IntegrationOrder::RK3);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}

}