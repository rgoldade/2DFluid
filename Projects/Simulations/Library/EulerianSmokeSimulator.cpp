#include "EulerianSmokeSimulator.h"

#include <iostream>

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "FieldAdvector.h"
#include "GeometricPressureProjection.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

namespace FluidSim2D::RegularGridSim
{

void EulerianSmokeSimulator::drawGrid(Renderer& renderer) const
{
	mySolidSurface.drawGrid(renderer, false);
}

void EulerianSmokeSimulator::drawFluidDensity(Renderer& renderer, float maxDensity)
{
	mySmokeDensity.drawVolumetric(renderer, Vec3f(1), Vec3f(0), 0, maxDensity);
}

void EulerianSmokeSimulator::drawFluidVelocity(Renderer& renderer, float length) const
{
	myVelocity.drawSamplePointVectors(renderer, Vec3f(0), myVelocity.dx() * length);
}

void EulerianSmokeSimulator::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3f(0), 3);
}

void EulerianSmokeSimulator::drawSolidVelocity(Renderer& renderer, float length) const
{
	mySolidVelocity.drawSamplePointVectors(renderer, Vec3f(0, 1, 0), mySolidVelocity.dx() * length);
}

// Incoming solid surface must already be inverted
void EulerianSmokeSimulator::setSolidSurface(const LevelSet& solidSurface)
{
	assert(solidSurface.isGridMatched(mySolidSurface));
	assert(solidSurface.isBackgroundNegative());

	EdgeMesh localMesh = solidSurface.buildDCMesh();

	mySolidSurface.setBackgroundNegative();
	mySolidSurface.initFromMesh(localMesh, false);
}

void EulerianSmokeSimulator::setSolidVelocity(const VectorGrid<float>& solidVelocity)
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

void EulerianSmokeSimulator::setFluidVelocity(const VectorGrid<float>& velocity)
{
	assert(velocity.isGridMatched(myVelocity));

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = velocity.grid(axis).unflatten(faceIndex);
					myVelocity(face, axis) = velocity(face, axis);
				}
			});
	}

}

void EulerianSmokeSimulator::setSmokeSource(const ScalarGrid<float>& density, const ScalarGrid<float>& temperature)
{
	assert(density.isGridMatched(mySmokeDensity));
	assert(temperature.isGridMatched(mySmokeTemperature));

	tbb::parallel_for(tbb::blocked_range<int>(0, mySmokeDensity.voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mySmokeDensity.unflatten(cellIndex);

				if (density(cell) > 0)
				{
					mySmokeDensity(cell) = density(cell);
					mySmokeTemperature(cell) = temperature(cell);
				}
			}
		});
}

void EulerianSmokeSimulator::advectOldPressure(float dt, InterpolationOrder order)
{
	auto velocityFunc = [&](float, const Vec2f& pos) { return myVelocity.biLerp(pos); };

	ScalarGrid<float> tempPressure(myOldPressure.xform(), myOldPressure.size());
	advectField(dt, tempPressure, myOldPressure, velocityFunc, IntegrationOrder::RK3, order);

	std::swap(myOldPressure, tempPressure);
}

void EulerianSmokeSimulator::advectFluidMaterial(float dt, InterpolationOrder order)
{
	auto velocityFunc = [&](float, const Vec2f& pos) { return myVelocity.biLerp(pos); };

	{
		ScalarGrid<float> tempDensity(mySmokeDensity.xform(), mySmokeDensity.size());

		advectField(dt, tempDensity, mySmokeDensity, velocityFunc, IntegrationOrder::RK3, order);

		std::swap(mySmokeDensity, tempDensity);
	}

	{
		ScalarGrid<float> tempTemperature(mySmokeTemperature.xform(), mySmokeTemperature.size());

		advectField(dt, tempTemperature, mySmokeTemperature, velocityFunc, IntegrationOrder::RK3, order);

		std::swap(mySmokeTemperature, tempTemperature);
	}
}

void EulerianSmokeSimulator::advectFluidVelocity(float dt, InterpolationOrder order)
{
	auto velocityFunc = [&](float, const Vec2f& pos) { return myVelocity.biLerp(pos);  };

	VectorGrid<float> tempVelocity(myVelocity.xform(), myVelocity.gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
		advectField(dt, tempVelocity.grid(axis), myVelocity.grid(axis), velocityFunc, IntegrationOrder::RK3, order);

	std::swap(myVelocity, tempVelocity);
}

void EulerianSmokeSimulator::runTimestep(float dt)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	//
	// Add bouyancy forces
	//

	float alpha = 1.;
	float beta = 1.;

	tbb::parallel_for(tbb::blocked_range<int>(0, myVelocity.grid(1).voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = myVelocity.grid(1).unflatten(faceIndex);

				// Average density and temperature values at velocity face
				Vec2f worldPosition = myVelocity.indexToWorld(Vec2f(face), 1);
				float density = mySmokeDensity.biLerp(worldPosition);
				float temperature = mySmokeTemperature.biLerp(worldPosition);

				float force = dt * (-alpha * density + beta * (temperature - myAmbientTemperature));
				myVelocity(face, 1) += force;
			}
		});

	std::cout << "Add forces: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	//
	// Build weights for pressure projection
	//

	VectorGrid<float> cutCellWeights = computeCutCellWeights(mySolidSurface, true);
	VectorGrid<float> ghostFluidWeights(myVelocity.xform(), myVelocity.gridSize(), 1, myVelocity.sampleType());

	std::cout << "  Compute weights: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	LevelSet dummySurface(mySolidSurface.xform(), mySolidSurface.size(), 5, true);

	//
	// Project divergence out of velocity field
	//

	// Initialize and call pressure projection
	GeometricPressureProjection projectDivergence(dummySurface,
		cutCellWeights,
		ghostFluidWeights,
		mySolidVelocity);


	projectDivergence.setInitialGuess(myOldPressure);
	projectDivergence.project(myVelocity, true);
	myOldPressure = projectDivergence.getPressureGrid();

	std::cout << "Pressure projection: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	const VectorGrid<VisitedCellLabels>& validFaces = projectDivergence.getValidFaces();

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
						myVelocity(face, axis) = 0;
				}
			});

		extrapolateField(myVelocity.grid(axis), validFaces.grid(axis), 4);
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	//
	// Advect fluid state
	//

	advectOldPressure(dt, InterpolationOrder::LINEAR);
	advectFluidMaterial(dt, InterpolationOrder::CUBIC);
	advectFluidVelocity(dt, InterpolationOrder::LINEAR);

	std::cout << "Advection: " << simTimer.stop() << "s" << std::endl;
}

}