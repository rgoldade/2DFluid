#include <memory>

#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

static std::unique_ptr<Renderer> gRenderer;

static bool gDoScalarTest = false;
static bool gDoVectorTest = false;
static bool gDoLevelSetTest = true;

int main(int argc, char** argv)
{
	double dx = .5;
	Vec2d topRightCorner(20, 20);
	Vec2d bottomLeftCorner(-20, -20);
	Vec2i size = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
	Transform xform(dx, bottomLeftCorner);
	Vec2d center = .5f * (topRightCorner + bottomLeftCorner);

	gRenderer = std::make_unique<Renderer>("Scalar Grid Test", Vec2i(1000, 1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Test scalar grid
	if (gDoScalarTest)
	{
		ScalarGrid<double> testGrid(xform, size, ScalarGridSettings::SampleType::CENTER, ScalarGridSettings::BorderType::CLAMP);

		// Make sure flatten and unflatten are working
		forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
		{
			int flatIndex = testGrid.flatten(cell);
			Vec2i testCell = testGrid.unflatten(flatIndex);

			assert(cell == testCell);
		});

		forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
		{
			Vec2d worldPosition = testGrid.indexToWorld(cell.cast<double>());
			testGrid(cell) = (worldPosition - center).norm();
		});

		testGrid.drawGrid(*gRenderer);
		testGrid.drawSamplePoints(*gRenderer, Vec3d(1, 0, 0), 5);
		testGrid.drawSupersampledValues(*gRenderer, .5, 3, 5);
	}
	// Test vector grid. TODO: move to vector grid test.. this is a scalar grid test after all.
	else if (gDoVectorTest)
	{
		VectorGrid<double> testVectorGrid(xform, size, VectorGridSettings::SampleType::STAGGERED, ScalarGridSettings::BorderType::CLAMP);

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i::Zero(), testVectorGrid.size(axis), [&](const Vec2i& cell)
			{
				Vec2d offset(0); offset[axis] += .5;
				Vec2d worldPosition0 = testVectorGrid.indexToWorld(cell.cast<double>() - offset, axis);
				Vec2d worldPosition1 = testVectorGrid.indexToWorld(cell.cast<double>() + offset, axis);

				double gradient  = ((worldPosition1 - center).norm() - (worldPosition0 - center).norm()) / dx;
				testVectorGrid(cell, axis) = gradient;
			});
		}

		testVectorGrid.drawGrid(*gRenderer);
		testVectorGrid.drawSamplePoints(*gRenderer, Vec3d(1, 0, 0), Vec3d(0, 1, 0), Vec2d(5, 5));
		testVectorGrid.drawSamplePointVectors(*gRenderer, Vec3d::Zero(), .5);
	}
	else if (gDoLevelSetTest)
	{
		LevelSet testLevelSet(xform, size, 5);

		double radius = .3 * (topRightCorner - center).norm();
		forEachVoxelRange(Vec2i::Zero(), testLevelSet.size(), [&](const Vec2i& cell)
		{
			Vec2d worldPosition = testLevelSet.indexToWorld(cell.cast<double>());
			testLevelSet(cell) = (worldPosition - center).norm() - radius;
		});
		
		testLevelSet.reinit(true);
		testLevelSet.drawGrid(*gRenderer, true);
		testLevelSet.drawSupersampledValues(*gRenderer, .25, 3, 5);
		testLevelSet.drawNormals(*gRenderer, Vec3d::Zero(), .5);

		testLevelSet.drawSurface(*gRenderer, Vec3d(1, 0, 0));
	}

	gRenderer->run();
}
