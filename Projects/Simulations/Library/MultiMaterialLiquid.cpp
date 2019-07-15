#include <string>

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
    for (int material = 0; material < myMaterialCount; ++material)
    {
		auto velocityFunc = [&](Real, const Vec2R& point)
		{
			return myFluidVelocities[material].interp(point);
		};

		VectorGrid<Real> localVelocity(myFluidVelocities[material].xform(), myFluidVelocities[material].gridSize(), 0, VectorGridSettings::SampleType::STAGGERED);

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

void MultiMaterialLiquid::runTimestep(Real dt, Renderer& renderer, int frame)
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

	MultiMaterialPressureProjection pressureSolver(extrapolatedSurfaces, myFluidDensities, mySolidSurface);

	pressureSolver.project(myFluidVelocities);

	std::cout << "  Solve for multi-material pressure: " << simTimer.stop() << "s" << std::endl;

	simTimer.reset();

	//
	// Get valid faces for each material so we can extrapolate velocities.
	//

	for (int material = 0; material < myMaterialCount; ++material)
	{
		VectorGrid<MarkedCells> validFaces = pressureSolver.getValidFaces(material);

		for (int axis : {0, 1})
		{
			// Extrapolate velocity
			ExtrapolateField<ScalarGrid<Real>> extrapolator(myFluidVelocities[material].grid(axis));
			extrapolator.extrapolate(validFaces.grid(axis), 5);
		}
	}

	std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
	simTimer.reset();
    
	advectFluidSurfaces(dt, IntegrationOrder::RK3);
	advectFluidVelocities(dt, IntegrationOrder::RK3);

	std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}
