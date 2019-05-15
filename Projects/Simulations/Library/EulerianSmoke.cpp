#include <iostream>

#include "EulerianSmoke.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
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
	myFluidVelocity.drawSamplePointVectors(renderer, Vec3f(0), myFluidVelocity.dx() * length);
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
void EulerianSmoke::setSolidSurface(const LevelSet2D& solidSurface)
{
	assert(solidSurface.isMatched(mySolidSurface));

	assert(solidSurface.inverted());

	Mesh2D localMesh = solidSurface.buildDCMesh();

	mySolidSurface.setInverted();
	mySolidSurface.init(localMesh, false);
}

void EulerianSmoke::setSolidVelocity(const VectorGrid<Real>& solidVelocity)
{
	assert(solidVelocity.isMatched(mySolidVelocity));
	for (auto axis : { 0,1 })
	{
		Vec2ui size = mySolidVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R worldPosition = mySolidVelocity.indexToWorld(Vec2R(cell), axis);
			mySolidVelocity(cell, axis) = solidVelocity.interp(worldPosition, axis);
		});
	}
}

void EulerianSmoke::setFluidVelocity(const VectorGrid<Real>& velocity)
{
	assert(velocity.isMatched(myFluidVelocity));

	for (auto axis : { 0,1 })
	{
		Vec2ui size = myFluidVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R worldPosition = myFluidVelocity.indexToWorld(Vec2R(cell), axis);
			myFluidVelocity(cell, axis) = velocity.interp(worldPosition, axis);
		});
	}
}

void EulerianSmoke::setSmokeSource(const ScalarGrid<Real>& density, const ScalarGrid<Real>& temperature)
{
	assert(density.isMatched(mySmokeDensity));
	assert(temperature.isMatched(mySmokeTemperature));

	Vec2ui size = mySmokeDensity.size();

	forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		if (density(cell) > 0)
		{
			mySmokeDensity(cell) = density(cell);
			mySmokeTemperature(cell) = temperature(cell);
		}
	});
}

void EulerianSmoke::advectFluidDensity(Real dt, const InterpolationOrder& order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myFluidVelocity.interp(pos); };

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

void EulerianSmoke::advectFluidVelocity(Real dt, const InterpolationOrder& order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myFluidVelocity.interp(pos);  };
	
	VectorGrid<Real> tempVelocity(myFluidVelocity.xform(), myFluidVelocity.gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (auto axis : { 0,1 })
	{
		AdvectField<ScalarGrid<Real>> advector(myFluidVelocity.grid(axis));

		advector.advectField(dt, tempVelocity.grid(axis), velocityFunc, IntegrationOrder::RK3, order);
	}

	std::swap(myFluidVelocity, tempVelocity);
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

	forEachVoxelRange(Vec2ui(0), myFluidVelocity.size(1), [&](const Vec2ui& face)
	{
		// Average density and temperature values at velocity face
		Vec2R worldPosition = myFluidVelocity.indexToWorld(Vec2R(face), 1);
		Real density = mySmokeDensity.interp(worldPosition);
		Real temperature = mySmokeTemperature.interp(worldPosition);

		Real force = dt * (-alpha * density + beta * (temperature - myAmbientTemperature));
		myFluidVelocity(face, 1) += force;

	});
	
	std::cout << "Add forces: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Build weights for pressure projection
	//
	
	LevelSet2D dummySurface(mySolidSurface.xform(), mySolidSurface.size(), 5, true);
		
	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> ghostFluidWeights(mySolidSurface.xform(), mySolidSurface.size(), 1., VectorGridSettings::SampleType::STAGGERED);
	VectorGrid<Real> cutCellWeights = computeCutCellWeights(mySolidSurface, true);

	std::cout << "Compute weights: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Project divergence out of velocity field
	//

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dummySurface, myFluidVelocity, mySolidSurface, mySolidVelocity);
	
	// TODO: handle moving boundaries.
	projectdivergence.project(ghostFluidWeights, cutCellWeights);

	// Update velocity field
	projectdivergence.applySolution(myFluidVelocity, ghostFluidWeights);
	
	std::cout << "Pressure projection: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();


	//
	// Advect fluid state
	//

	advectFluidDensity(dt, InterpolationOrder::CUBIC);
	advectFluidVelocity(dt, InterpolationOrder::LINEAR);

	std::cout << "Advection: " << simTimer.stop() << "s" << std::endl;
}