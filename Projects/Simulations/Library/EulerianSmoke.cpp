#include <iostream>

#include "EulerianSmoke.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "PressureProjection.h"
#include "Timer.h"

void EulerianSmoke::drawGrid(Renderer& renderer) const
{
	myCollision.drawGrid(renderer);
}

void EulerianSmoke::drawSmoke(Renderer& renderer, Real maxDensity)
{
	mySmokeDensity.drawVolumetric(renderer, Vec3f(1), Vec3f(0), 0, maxDensity);
}

void EulerianSmoke::drawCollisionSurface(Renderer& renderer)
{
	myCollision.drawSurface(renderer, Vec3f(1., 0., 1.));
}

void EulerianSmoke::drawCollisionVelocity(Renderer& renderer, Real length) const
{
	myCollisionVelocity.drawSamplePointVectors(renderer, Vec3f(0, 1, 0), myCollisionVelocity.dx() * length);
}

void EulerianSmoke::drawVelocity(Renderer& renderer, Real length) const
{
	myVelocity.drawSamplePointVectors(renderer, Vec3f(0), myVelocity.dx() * length);
}

// Incoming collision volume must already be inverted
void EulerianSmoke::setCollisionVolume(const LevelSet2D& collision)
{
	assert(collision.isMatched(myCollision));

	assert(collision.inverted());

	Mesh2D tempMesh = collision.buildDCMesh();

	// TODO : consider making collision node sampled
	myCollision.setInverted();
	myCollision.init(tempMesh, false);
}

void EulerianSmoke::setCollisionVelocity(const VectorGrid<Real>& collisionVelocity)
{
	assert(collisionVelocity.isMatched(myCollisionVelocity));
	for (auto axis : { 0,1 })
	{
		Vec2ui size = myCollisionVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R worldPosition = myCollisionVelocity.indexToWorld(Vec2R(cell), axis);
			myCollisionVelocity(cell, axis) =  collisionVelocity.interp(worldPosition, axis);
		});
	}
}

void EulerianSmoke::setSmokeVelocity(const VectorGrid<Real>& velocity)
{
	assert(velocity.isMatched(myVelocity));

	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
		{
			Vec2R worldPosition = myVelocity.indexToWorld(Vec2R(cell), axis);
			myVelocity(cell, axis) = velocity.interp(worldPosition, axis);
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

void EulerianSmoke::advectSmoke(Real dt, const InterpolationOrder& order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos); };

	{
		AdvectField<ScalarGrid<Real>> densityAdvector(mySmokeDensity);

		ScalarGrid<Real> tempDensity(mySmokeDensity.xform(), mySmokeDensity.size());
		// TODO: add some sort of collision management
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

void EulerianSmoke::advectVelocity(Real dt, const InterpolationOrder& order)
{
	auto velocityFunc = [&](Real, const Vec2R& pos) { return myVelocity.interp(pos);  };
	
	VectorGrid<Real> tempVelocity(myVelocity.xform(), myVelocity.gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (auto axis : { 0,1 })
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

	forEachVoxelRange(Vec2ui(0), myVelocity.size(1), [&](const Vec2ui& face)
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
	
	LevelSet2D dummySurface(myCollision.xform(), myCollision.size(), 5, true);
		
	// Compute weights for both liquid-solid side and air-liquid side
	VectorGrid<Real> ghostFluidWeights(myCollision.xform(), myCollision.size(), 1., VectorGridSettings::SampleType::STAGGERED);
	VectorGrid<Real> cutCellWeights = computeCutCellWeights(myCollision, true);

	std::cout << "Compute weights: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Project divergence out of velocity field
	//

	// Initialize and call pressure projection
	PressureProjection projectdivergence(dt, dummySurface, myVelocity, myCollision, myCollisionVelocity);
	// TODO: handle moving boundaries.

	VectorGrid<Real> valid(myCollision.xform(), myCollision.size(), 0, VectorGridSettings::SampleType::STAGGERED);
	
	projectdivergence.project(ghostFluidWeights, cutCellWeights);

	// Update velocity field
	projectdivergence.applySolution(myVelocity, ghostFluidWeights);

	projectdivergence.applyValid(valid);

	std::cout << "Pressure projection: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();


	//
	// Advect fluid state
	//

	advectSmoke(dt, InterpolationOrder::CUBIC);
	advectVelocity(dt, InterpolationOrder::LINEAR);

	std::cout << "Advection: " << simTimer.stop() << "s" << std::endl;
}