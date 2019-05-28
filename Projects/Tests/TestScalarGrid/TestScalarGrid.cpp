#include <memory>

#include "Common.h"

#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Util.h"
#include "Vec.h"
#include "VectorGrid.h"

static std::unique_ptr<Renderer> renderer;

static bool doScalarTest = false;
static bool doVectorTest = false;
static bool doLevelSetTest = true;

int main(int argc, char** argv)
{
	Real dx = 2;
	Vec2R topRightCorner(20);
	Vec2R bottomLeftCorner(-20);
	Vec2i size((topRightCorner - bottomLeftCorner) / dx);
	Transform xform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Scalar Grid Test", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Test scalar grid
	if (doScalarTest)
	{
		ScalarGrid<Real> testGrid(xform, size, ScalarGridSettings::SampleType::NODE, ScalarGridSettings::BorderType::CLAMP);

		forEachVoxelRange(Vec2i(0), testGrid.size(), [&](const Vec2i& cell)
		{
			Vec2R worldPosition = testGrid.indexToWorld(Vec2R(cell));
			testGrid(cell) = mag(worldPosition - center);
		});

		testGrid.drawGrid(*renderer);
		testGrid.drawSamplePoints(*renderer);
		testGrid.drawSupersampledValues(*renderer, .5, 3, 5);
	}
	// Test vector grid. TODO: move to vector grid test.. this is a scalar grid test after all.
	else if (doVectorTest)
	{
		VectorGrid<Real> testVectorGrid(xform, size, VectorGridSettings::SampleType::STAGGERED, ScalarGridSettings::BorderType::CLAMP);

		for (int axis : { 0, 1 })
		{
			forEachVoxelRange(Vec2i(0), testVectorGrid.size(axis), [&](const Vec2i& cell)
			{
				Vec2R offset(0); offset[axis] += .5;
				Vec2R worldPosition0 = testVectorGrid.indexToWorld(Vec2R(cell) - offset, axis);
				Vec2R worldPosition1 = testVectorGrid.indexToWorld(Vec2R(cell) + offset, axis);

				Real gradient  = (mag(worldPosition1 - center) - mag(worldPosition0 - center)) / dx;
				testVectorGrid(cell, axis) = gradient;
			});
		}

		testVectorGrid.drawGrid(*renderer);
		testVectorGrid.drawSamplePoints(*renderer, Vec3f(1,0,0), Vec3f(0,1,0), Vec2R(5));
		testVectorGrid.drawSamplePointVectors(*renderer, Vec3f(0), .5);
	}
	else if (doLevelSetTest)
	{
		LevelSet testLevelSet(xform, size, 5);

		Real radius = .3 * mag(topRightCorner - center);
		forEachVoxelRange(Vec2i(0), testLevelSet.size(), [&](const Vec2i& cell)
		{
			Vec2R worldPosition = testLevelSet.indexToWorld(Vec2R(cell));
			testLevelSet(cell) = mag(worldPosition - center) - radius;
		});
		
		testLevelSet.reinit();
		testLevelSet.drawGrid(*renderer);
		testLevelSet.drawSupersampledValues(*renderer, .25, 3, 5);
		testLevelSet.drawNormals(*renderer, Vec3f(0,0,0), .5);

		testLevelSet.drawSurface(*renderer, Vec3f(1,0,0));
	}

	renderer->run();
}
