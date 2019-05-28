#include "MultiMaterialLiquid.h"

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "MultiMaterialPressureProjection.h"
#include "Timer.h"

void MultiMaterialLiquid::drawMaterialSurface(Renderer& renderer, int material)
{
	myFluidSurfaces[material].drawSurface(renderer, Vec3f(0., 0., 1.0));
}

void MultiMaterialLiquid::drawMaterialVelocity(Renderer& renderer, Real length, int material) const
{
	myFluidVelocities[material].drawSamplePointVectors(renderer, Vec3f(0), myFluidVelocities[material].dx() * length);
}

void MultiMaterialLiquid::drawSolidSurface(Renderer &renderer)
{
    mySolidSurface.drawSurface(renderer, Vec3f(1.,0.,1.));
}

template<typename ForceSampler>
void MultiMaterialLiquid::addForce(Real dt, int material, const ForceSampler& force)
{
	for (int axis : {0, 1})
    {
		Vec2i size = myFluidVelocities[material].size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2R worldPosition = myFluidVelocities[material].indexToWorld(Vec2R(face), axis);
			myFluidVelocities[material](face, axis) += dt * force(worldPosition, axis);
		});
    }
}

void MultiMaterialLiquid::addForce(Real dt, int material, const Vec2R& force)
{
    addForce(dt, material, [&](Vec2R, int axis) { return force[axis]; });
}

void MultiMaterialLiquid::advectFluidVelocities(Real dt, IntegrationOrder integrator, InterpolationOrder interpolator)
{
    VectorGrid<Real> localVelocity;
    for (int material = 0; material < myMaterialCount; ++material)
    {
		auto velocityFunc = [&](Real, const Vec2R& point)
		{
			return myFluidVelocities[material].interp(point);
		};

		localVelocity = VectorGrid<Real>(myFluidVelocities[material].xform(), myFluidVelocities[material].gridSize(), VectorGridSettings::SampleType::STAGGERED);

		for (int axis : {0, 1})
		{
			AdvectField<ScalarGrid<Real>> advector(myFluidVelocities[material].grid(axis));
			advector.advectField(dt, localVelocity.grid(axis), velocityFunc, integrator, interpolator);
		}

		std::swap(myFluidVelocities[material], localVelocity);
    }
}

void MultiMaterialLiquid::advectFluidSurfaces(Real dt, IntegrationOrder integrator)
{
    for (int material = 0; material < myMaterialCount; ++material)
    {
		auto velocityFunc = [&](Real, const Vec2R& point)
		{
			return myFluidVelocities[material].interp(point);
		};

		EdgeMesh localMesh = myFluidSurfaces[material].buildMSMesh();
		localMesh.advect(dt, velocityFunc, integrator);
		myFluidSurfaces[material].initFromMesh(localMesh, false);
    }

	// Fix possible overlaps between the materials.
	forEachVoxelRange(Vec2i(0), myGridSize, [&](const Vec2i& cell)
	{
		Real firstMin = std::min(myFluidSurfaces[0](cell), mySolidSurface(cell));
		Real secondMin = std::max(myFluidSurfaces[0](cell), mySolidSurface(cell));

		if (myMaterialCount > 1)
		{
			for (int material = 1; material < myMaterialCount; ++material)
			{
				Real localMin = myFluidSurfaces[material](cell);
				if (localMin < firstMin)
				{
					secondMin = firstMin;
					firstMin = localMin;
				}
				else secondMin = std::min(localMin, secondMin);
			}
		}

		Real avgSDF = .5 * (firstMin + secondMin);

		for (int material = 0; material < myMaterialCount; ++material)
			myFluidSurfaces[material](cell) -= avgSDF;
	});

	for (int material = 0; material < myMaterialCount; ++material)
		myFluidSurfaces[material].reinitMesh();
}

void MultiMaterialLiquid::setSolidSurface(const LevelSet &solidSurface)
{
    assert(solidSurface.isBoundaryNegative());

    EdgeMesh localMesh = solidSurface.buildDCMesh();
    
    mySolidSurface.setBoundaryNegative();
    mySolidSurface.initFromMesh(localMesh, false /* don't resize grid*/);
}

void MultiMaterialLiquid::runTimestep(Real dt, Renderer& renderer)
{
	std::cout << "\nStarting simulation loop\n" << std::endl;

	Timer simTimer;

	//
	// Extrapolate materials into solids
	//

	std::vector<LevelSet> extrapolatedSurfaces(myMaterialCount);

	for (unsigned material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material] = myFluidSurfaces[material];

	Real dx = mySolidSurface.dx();
	forEachVoxelRange(Vec2i(0), myGridSize, [&](const Vec2i& cell)
	{
		for (int material = 0; material < myMaterialCount; ++material)
		{
			if (mySolidSurface(cell) <= 0. ||
				(mySolidSurface(cell) <= dx && myFluidSurfaces[material](cell) <= 0))
				extrapolatedSurfaces[material](cell) -= dx;
		}
	});

	for (int material = 0; material < myMaterialCount; ++material)
		extrapolatedSurfaces[material].reinitMesh();

	std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
	
	simTimer.reset();

	//
	// Compute cut-cell weights for each material. Compute fraction of
	// edge inside the material. After, normalize the weights to sum to
	// one.
	//

	VectorGrid<Real> solidCutCellWeights = computeCutCellWeights(mySolidSurface);

	std::vector<VectorGrid<Real>> materialCutCellWeights(myMaterialCount);

	for (int material = 0; material < myMaterialCount; ++material)
		materialCutCellWeights[material] = computeCutCellWeights(extrapolatedSurfaces[material]);

	// Now normalize the weights, removing the solid boundary contribution first.
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), myFluidVelocities[0].size(axis), [&](const Vec2i& face)
		{
			Real weight = 1;
			weight -= solidCutCellWeights(face, axis);
			weight = Util::clamp(weight, 0., weight);

			if (weight > 0)
			{
				Real accumulatedWeight = 0;
				for (int material = 0; material < myMaterialCount; ++material)
					accumulatedWeight += materialCutCellWeights[material](face, axis);

				if (accumulatedWeight > 0)
				{
					weight /= accumulatedWeight;

					for (int material = 0; material < myMaterialCount; ++material)
						materialCutCellWeights[material](face, axis) *= weight;
				}
			}
			else
			{
				for (int material = 0; material < myMaterialCount; ++material)
					materialCutCellWeights[material](face, axis) = 0;
			}

			// Debug check
			Real totalWeight = solidCutCellWeights(face, axis);

			for (int material = 0; material < myMaterialCount; ++material)
				totalWeight += materialCutCellWeights[material](face, axis);

			if (totalWeight == 0)
			{
				// If there is a zero total weight it is likely due to a fluid-fluid boundary
				// falling exactly across a grid face. There should never be a zero weight
				// along a fluid-solid boundary.

				std::vector<int> faceAlignedSurfaces;

				int otherAxis = (axis + 1) % 2;

				Vec2R offset(0); offset[otherAxis] = .5;

				for (int material = 0; material < myMaterialCount; ++material)
				{
					Vec2R pos0 = materialCutCellWeights[material].indexToWorld(Vec2R(face) - offset, axis);
					Vec2R pos1 = materialCutCellWeights[material].indexToWorld(Vec2R(face) + offset, axis);

					Real weight = lengthFraction(myFluidSurfaces[material].interp(pos0), myFluidSurfaces[material].interp(pos1));

					if (weight == 0)
						faceAlignedSurfaces.push_back(material);
				}

				if (!(faceAlignedSurfaces.size() > 1))
				{
					std::cout << "Zero weight problems!!" << std::endl;
					exit(-1);
				}
				assert(faceAlignedSurfaces.size() > 1);

				materialCutCellWeights[faceAlignedSurfaces[0]](face, axis) = 1.;
			}
		});
	}

	std::cout << "  Compute cut-cell weights: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();

	//
	// Solve for pressure for each material to return their velocities to an incompressible state
	//

	MultiMaterialPressureProjection pressureSolver(extrapolatedSurfaces, myFluidVelocities, myFluidDensities, mySolidSurface);

	pressureSolver.project(materialCutCellWeights, solidCutCellWeights);
	pressureSolver.applySolution(myFluidVelocities, materialCutCellWeights);

	std::cout << "  Solve for multi-material pressure: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Build a list of valid faces for each material so we can extrapolate with.
	//

	for (int material = 0; material < myMaterialCount; ++material)
	{
		VectorGrid<MarkedCells> valid(myXform, myGridSize, MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i(0), myFluidVelocities[material].size(axis), [&](const Vec2i& face)
			{
				if (materialCutCellWeights[material](face, axis) > 0.)
					valid(face, axis) = MarkedCells::FINISHED;
			});

			// Extrapolate velocity
			ExtrapolateField<ScalarGrid<Real>> extrapolator(myFluidVelocities[material].grid(axis));
			extrapolator.extrapolate(valid.grid(axis), 5);
		}
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();
    
	advectFluidSurfaces(dt, IntegrationOrder::RK3);
	advectFluidVelocities(dt, IntegrationOrder::RK3);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
