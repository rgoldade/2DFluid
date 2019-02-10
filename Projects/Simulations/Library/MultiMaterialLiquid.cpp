#include <limits>

#include "MultiMaterialLiquid.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "MultiMaterialPressureProjection.h"
#include "Timer.h"

void MultiMaterialLiquid::drawMaterialSurface(Renderer& renderer, unsigned material)
{
	mySurfaces[material].drawSurface(renderer, Vec3f(0., 0., 1.0));
}

void MultiMaterialLiquid::drawMaterialVelocity(Renderer& renderer, Real length, unsigned material) const
{
	myVelocities[material].drawSamplePointVectors(renderer, Vec3f(0), myVelocities[material].dx() * length);
}

void MultiMaterialLiquid::drawCollisionSurface(Renderer &renderer)
{
    myCollisionSurface.drawSurface(renderer, Vec3f(1.,0.,1.));
}

template<typename ForceSampler>
void MultiMaterialLiquid::addForce(const Real dt, const unsigned material, const ForceSampler& force)
{
	for (auto axis : { 0,1 })
    {
		Vec2ui size = myVelocities[material].size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2R worldPosition = myVelocities[material].indexToWorld(Vec2R(face), axis);
			myVelocities[material](face, axis) += dt * force(worldPosition, axis);
		});
    }
}

void MultiMaterialLiquid::addForce(const Real dt, const unsigned material, const Vec2R& force)
{
    addForce(dt, material, [&](Vec2R, unsigned axis) { return force[axis]; });
}

void MultiMaterialLiquid::advectVelocity(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
    VectorGrid<Real> tempVelocity;
    for (unsigned material = 0; material < myMaterialCount; ++material)
    {
		auto velocityFunc = [&](Real, const Vec2R& point)
		{
			return myVelocities[material].interp(point);
		};

		tempVelocity = VectorGrid<Real>(myVelocities[material].xform(), myVelocities[material].gridSize(), VectorGridSettings::SampleType::STAGGERED);

		for (auto axis : { 0,1 })
		{
			AdvectField<ScalarGrid<Real>> advector(myVelocities[material].grid(axis));
			advector.advectField(dt, tempVelocity.grid(axis), velocityFunc, integrator, interpolator);
		}

		std::swap(myVelocities[material], tempVelocity);
    }
}

void MultiMaterialLiquid::advectSurface(Real dt, IntegrationOrder integrator)
{
    for (unsigned material = 0; material < myMaterialCount; ++material)
    {
		auto velocityFunc = [&](Real, const Vec2R& point)
		{
			return myVelocities[material].interp(point);
		};

		Mesh2D tempMesh = mySurfaces[material].buildMSMesh();
		tempMesh.advect(dt, velocityFunc, integrator);
		mySurfaces[material].init(tempMesh, false);
    }

	// Fix possible overlaps between the materials.
	forEachVoxelRange(Vec2ui(0), myGridSize, [&](const Vec2ui& cell)
	{
		/*Real firstMin = std::min(mySurfaces[0](cell), myCollisionSurface(cell));
		Real secondMin = std::max(mySurfaces[0](cell), myCollisionSurface(cell));*/

		Real firstMin = std::min(mySurfaces[0](cell), myCollisionSurface(cell));
		Real secondMin = std::max(mySurfaces[0](cell), myCollisionSurface(cell));

		if (myMaterialCount > 1)
		{
			for (unsigned material = 1; material < myMaterialCount; ++material)
			{
				Real localMin = mySurfaces[material](cell);
				if (localMin < firstMin)
				{
					secondMin = firstMin;
					firstMin = localMin;
				}
				else secondMin = std::min(localMin, secondMin);
			}
		}

		Real avgSDF = .5 * (firstMin + secondMin);

		for (unsigned material = 0; material < myMaterialCount; ++material)
			mySurfaces[material](cell) -= avgSDF;
	});

	for (unsigned material = 0; material < myMaterialCount; ++material)
		mySurfaces[material].reinitMesh();
}

void MultiMaterialLiquid::setCollisionVolume(const LevelSet2D &collision)
{
    assert(collision.inverted());

    Mesh2D tempMesh = collision.buildDCMesh();
    
    // TODO : consider making collision node sampled
    myCollisionSurface.setInverted();
    myCollisionSurface.init(tempMesh, false);
}

void MultiMaterialLiquid::runTimestep(Real dt, Renderer& renderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	//
	// Extrapolate materials into solids
	//

	std::vector<LevelSet2D> extrapolatedSurfaces(myMaterialCount);

	for (unsigned material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material] = mySurfaces[material];

	Real dx = myCollisionSurface.dx();
	forEachVoxelRange(Vec2ui(0), myGridSize, [&](const Vec2ui& cell)
	{
		// Extrapolate the closest fluid into the collision surfaces
		if (myCollisionSurface(cell) <= 0.)
		{
			Real minSDF = std::numeric_limits<Real>::max();
			unsigned minMaterial = -1;
			for (unsigned material = 0; material < myMaterialCount; ++material)
			{
				Real localSDF = extrapolatedSurfaces[material](cell);

				if (localSDF < minSDF)
				{
					minSDF = localSDF;
					minMaterial = material;
				}
			}
				
			extrapolatedSurfaces[minMaterial](cell) -= dx;
		}
	});

	for (unsigned material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material].reinitMesh();

	for (unsigned material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material].drawSurface(renderer);

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	
	simTimer.reset();

	//
	// Compute cut-cell weights for each material. Compute fraction of
	// edge inside the material. After, normalize the weights to sum to
	// one.
	//

	VectorGrid<Real> collisionCutCellWeight = computeCutCellWeights(myCollisionSurface);

	std::vector<VectorGrid<Real>> materialCutCellWeights(myMaterialCount);

	for (unsigned material = 0; material < myMaterialCount; ++material)
		materialCutCellWeights[material] = computeCutCellWeights(extrapolatedSurfaces[material]);

	// Now normalize the weights, removing the collision contribution first.
	for (auto axis : { 0,1 })
	{
		forEachVoxelRange(Vec2ui(0), myVelocities[0].size(axis), [&](const Vec2ui& face)
		{
			Real weight = 1;
			weight -= collisionCutCellWeight(face, axis);
			weight = Util::clamp(weight, 0., weight);

			if (weight > 0)
			{
				Real accumulatedWeight = 0;
				for (unsigned material = 0; material < myMaterialCount; ++material)
					accumulatedWeight += materialCutCellWeights[material](face, axis);

				if (accumulatedWeight > 0)
				{
					weight /= accumulatedWeight;

					for (unsigned material = 0; material < myMaterialCount; ++material)
						materialCutCellWeights[material](face, axis) *= weight;
				}
			}
			else
			{
				for (unsigned material = 0; material < myMaterialCount; ++material)
					materialCutCellWeights[material](face, axis) = 0;
			}

			// Debug check
			Real totalWeight = collisionCutCellWeight(face, axis);

			for (unsigned material = 0; material < myMaterialCount; ++material)
				totalWeight += materialCutCellWeights[material](face, axis);

			assert(Util::isEqual(totalWeight, 1.));
		});
	}

	std::cout << "  Compute cut-cell weights: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	//
	// Solve for pressure for each material to return their velocities to an incompressible state
	//

	MultiMaterialPressureProjection pressureSolver(extrapolatedSurfaces, myVelocities, myDensities, myCollisionSurface);

	pressureSolver.project(materialCutCellWeights, collisionCutCellWeight);
	pressureSolver.applySolution(myVelocities, materialCutCellWeights);

	pressureSolver.drawPressure(renderer);
	std::cout << "  Solve for multi-material pressure: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Build a list of valid faces for each material so we can extrapolate with.
	//

	for (unsigned material = 0; material < myMaterialCount; ++material)
	{
		VectorGrid<Real> valid(myXform, myGridSize, 0, VectorGridSettings::SampleType::STAGGERED);

		for (auto axis : { 0,1 })
		{
			forEachVoxelRange(Vec2ui(0), myVelocities[material].size(axis), [&](const Vec2ui& face)
			{
				if (materialCutCellWeights[material](face, axis) > 0.)
					valid(face, axis) = 1;
			});
		}

		// Extrapolate velocity
		ExtrapolateField<VectorGrid<Real>> extrapolator(myVelocities[material]);
		extrapolator.extrapolate(valid, 5);
	}

	//for (auto axis : { 0,1 })
	//{
	//	Vec2ui velSize = myVelocities[0].size(axis);

	//	forEachVoxelRange(Vec2ui(0), velSize, [&](const Vec2ui& face)
	//	{
	//		Real totalVelocity = 0.;
	//		Real totalWeight = 0.;

	//		for (unsigned material = 0; material < myMaterialCount; ++material)
	//		{
	//			Real localWeight = materialCutCellWeights[material](face, axis);
	//			if (localWeight > 0.)
	//			{
	//				totalWeight += localWeight;
	//				totalVelocity += localWeight * myVelocities[material](face, axis);
	//			}
	//		}

	//		if (totalWeight > 0.)
	//			totalVelocity /= totalWeight;

	//		for (unsigned material = 0; material < myMaterialCount; ++material)
	//			myVelocities[material](face, axis) = totalVelocity;
	//	});
	//}

    std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
    simTimer.reset();
    
    advectSurface(dt, IntegrationOrder::RK3);
    advectVelocity(dt, IntegrationOrder::RK3);

    std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
