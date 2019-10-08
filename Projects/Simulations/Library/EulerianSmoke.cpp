#include <iostream>

#include "EulerianSmoke.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "GeometricPressureProjection.h"
#include "PressureProjection.h"
#include "Timer.h"

void EulerianSmoke::drawGrid(Renderer& renderer) const
{
	mySolidSurface.drawGrid(renderer);
}

void EulerianSmoke::drawFluidDensity(Renderer& renderer, Real maxDensity)
{
	mySmokeDensity.drawVolumetric(renderer, Vec3f(1), Vec3f(0), 0, maxDensity);
}

void EulerianSmoke::drawFluidVelocity(Renderer& renderer, Real length) const
{
	myVelocity.drawSamplePointVectors(renderer, Vec3f(0), myVelocity.dx() * length);
}

void EulerianSmoke::drawSolidSurface(Renderer& renderer)
{
	mySolidSurface.drawSurface(renderer, Vec3f(1., 0., 1.));
}

void EulerianSmoke::drawSolidVelocity(Renderer& renderer, Real length) const
{
	mySolidVelocity.drawSamplePointVectors(renderer, Vec3f(0, 1, 0), mySolidVelocity.dx() * length);
}

// Incoming solid surface must already be inverted
void EulerianSmoke::setSolidSurface(const LevelSet& solidSurface)
{
	assert(solidSurface.isGridMatched(mySolidSurface));
	assert(solidSurface.isBackgroundNegative());

	EdgeMesh localMesh = solidSurface.buildDCMesh();

	mySolidSurface.setBackgroundNegative();
	mySolidSurface.initFromMesh(localMesh, false);
}

void EulerianSmoke::setSolidVelocity(const VectorGrid<Real>& solidVelocity)
{
	assert(solidVelocity.isGridMatched(mySolidVelocity));
	for (int axis : {0, 1})
	{
		Vec2i size = mySolidVelocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
		{
			Vec2R worldPosition = mySolidVelocity.indexToWorld(Vec2R(cell), axis);
			mySolidVelocity(cell, axis) = solidVelocity.interp(worldPosition, axis);
		});
	}
}

void EulerianSmoke::setFluidVelocity(const VectorGrid<Real>& velocity)
{
	assert(velocity.isGridMatched(myVelocity));

	for (int axis : {0, 1})
	{
		Vec2i size = myVelocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
		{
			Vec2R worldPosition = myVelocity.indexToWorld(Vec2R(cell), axis);
			myVelocity(cell, axis) = velocity.interp(worldPosition, axis);
		});
	}
}

void EulerianSmoke::setSmokeSource(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature)
{
	assert(density.isGridMatched(mySmokeDensity));
	assert(temperature.isGridMatched(mySmokeTemperature));

	Vec2i size = mySmokeDensity.size();

	forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
	{
		if (density(cell) > 0)
		{
			mySmokeDensity(cell) = density(cell);
			mySmokeTemperature(cell) = temperature(cell);
		}
	});
}

void EulerianSmoke::advectOldPressure(const Real dt, const InterpolationOrder order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> pressureAdvector(myOldPressure);

		ScalarGrid<Real> tempPressure(myOldPressure.xform(), myOldPressure.size());

		pressureAdvector.advectField(dt, tempPressure, velocityFunc, IntegrationOrder::RK3, order);

		std::swap(myOldPressure, tempPressure);
	}
}

void EulerianSmoke::advectFluidMaterial(const Real dt, const InterpolationOrder order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> densityAdvector(mySmokeDensity);

		ScalarGrid<Real> tempDensity(mySmokeDensity.xform(), mySmokeDensity.size());

		densityAdvector.advectField(dt, tempDensity, velocityFunc, IntegrationOrder::RK3, order);
		
		std::swap(mySmokeDensity, tempDensity);
	}

	{
		AdvectField<ScalarGrid<Real>> temperatureAdvector(mySmokeTemperature);

		ScalarGrid<Real> tempTemperature(mySmokeTemperature.xform(), mySmokeTemperature.size());

		temperatureAdvector.advectField(dt, tempTemperature, velocityFunc, IntegrationOrder::RK3, order);

		std::swap(mySmokeTemperature, tempTemperature);
	}
}

void EulerianSmoke::advectFluidVelocity(const Real dt, const InterpolationOrder order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos);  };
	
	VectorGrid<Real> tempVelocity(myVelocity.xform(), myVelocity.gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		AdvectField<ScalarGrid<Real>> advector(myVelocity.grid(axis));

		advector.advectField(dt, tempVelocity.grid(axis), velocityFunc, IntegrationOrder::RK3, order);
	}

	std::swap(myVelocity, tempVelocity);
}

void EulerianSmoke::runTimestep(Real dt, Renderer& renderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	//
	// Add bouyancy forces
	//

	Real alpha = 1.;
	Real beta = 1.;

	forEachVoxelRange(Vec2i(0), myVelocity.size(1), [&](const Vec2i& face)
	{
		// Average density and temperature values at velocity face
		Vec2R worldPosition = myVelocity.indexToWorld(Vec2R(face), 1);
		Real density = mySmokeDensity.interp(worldPosition);
		Real temperature = mySmokeTemperature.interp(worldPosition);

		Real force = dt * (-alpha * density + beta * (temperature - myAmbientTemperature));
		myVelocity(face, 1) += force;

	});
	
	std::cout << "Add forces: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	//
	// Build weights for pressure projection
	//
	
	LevelSet dummySurface(mySolidSurface.xform(), mySolidSurface.size(), 5, true);

	//
	// Project divergence out of velocity field
	//

	// Initialize and call pressure projection
	GeometricPressureProjection projectDivergence(dummySurface, mySolidSurface, mySolidVelocity);
	
	projectDivergence.setInitialGuess(myOldPressure);
	projectDivergence.project(myVelocity, true);
	myOldPressure = projectDivergence.getPressureGrid();

	std::cout << "Pressure projection: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	const VectorGrid<MarkedCells>& validFaces = projectDivergence.getValidFaces();

	// Extrapolate velocity
	for (int axis : {0, 1})
	{
		ExtrapolateField<ScalarGrid<Real>> extrapolator(myVelocity.grid(axis));
		extrapolator.extrapolate(validFaces.grid(axis), 4);
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