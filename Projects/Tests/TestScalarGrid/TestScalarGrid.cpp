#include <memory>

#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;

static std::unique_ptr<Renderer> renderer;

static bool doScalarTest = false;
static bool doVectorTest = false;
static bool doLevelSetTest = true;

int main(int argc, char** argv)
{
	float dx = .5;
	Vec2f topRightCorner(20);
	Vec2f bottomLeftCorner(-20);
	Vec2i size((topRightCorner - bottomLeftCorner) / dx);
	Transform xform(dx, bottomLeftCorner);
	Vec2f center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Scalar Grid Test", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Test scalar grid
	if (doScalarTest)
	{
		ScalarGrid<float> testGrid(xform, size, ScalarGridSettings::SampleType::CENTER, ScalarGridSettings::BorderType::CLAMP);

		// Make sure flatten and unflatten are working
		forEachVoxelRange(Vec2i(0), testGrid.size(), [&](const Vec2i& cell)
		{
			int flatIndex = testGrid.flatten(cell);
			Vec2i testCell = testGrid.unflatten(flatIndex);

			assert(cell == testCell);
		});

		forEachVoxelRange(Vec2i(0), testGrid.size(), [&](const Vec2i& cell)
		{
			Vec2f worldPosition = testGrid.indexToWorld(Vec2f(cell));
			testGrid(cell) = mag(worldPosition - center);
		});

		testGrid.drawGrid(*renderer);
		testGrid.drawSamplePoints(*renderer, Vec3f(1, 0, 0), 5);
		testGrid.drawSupersampledValues(*renderer, .5, 3, 5);
	}
	// Test vector grid. TODO: move to vector grid test.. this is a scalar grid test after all.
	else if (doVectorTest)
	{
		VectorGrid<float> testVectorGrid(xform, size, VectorGridSettings::SampleType::STAGGERED, ScalarGridSettings::BorderType::CLAMP);

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i(0), testVectorGrid.size(axis), [&](const Vec2i& cell)
			{
				Vec2f offset(0); offset[axis] += .5;
				Vec2f worldPosition0 = testVectorGrid.indexToWorld(Vec2f(cell) - offset, axis);
				Vec2f worldPosition1 = testVectorGrid.indexToWorld(Vec2f(cell) + offset, axis);

				float gradient  = (mag(worldPosition1 - center) - mag(worldPosition0 - center)) / dx;
				testVectorGrid(cell, axis) = gradient;
			});
		}

		testVectorGrid.drawGrid(*renderer);
		testVectorGrid.drawSamplePoints(*renderer, Vec3f(1,0,0), Vec3f(0,1,0), Vec2f(5));
		testVectorGrid.drawSamplePointVectors(*renderer, Vec3f(0), .5);
	}
	else if (doLevelSetTest)
	{
		LevelSet testLevelSet(xform, size, 5);

		float radius = .3 * mag(topRightCorner - center);
		forEachVoxelRange(Vec2i(0), testLevelSet.size(), [&](const Vec2i& cell)
		{
			Vec2f worldPosition = testLevelSet.indexToWorld(Vec2f(cell));
			testLevelSet(cell) = mag(worldPosition - center) - radius;
		});
		
		testLevelSet.reinit(true);
		testLevelSet.drawGrid(*renderer, true);
		testLevelSet.drawSupersampledValues(*renderer, .25, 3, 5);
		testLevelSet.drawNormals(*renderer, Vec3f(0,0,0), .5);

		testLevelSet.drawSurface(*renderer, Vec3f(1,0,0));
	}

	renderer->run();
}
