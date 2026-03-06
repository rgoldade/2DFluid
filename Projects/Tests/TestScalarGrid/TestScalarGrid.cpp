#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static const char* gTestNames[] = { "Scalar Grid", "Vector Grid", "Level Set" };
static int gCurrentTest = 0;

static void registerTest(int testIndex,
						  const Transform& xform,
						  const Vec2i& size,
						  const Vec2d& center,
						  const Vec2d& topRightCorner,
						  double dx)
{
	polyscope::removeAllStructures();

	if (testIndex == 0)
	{
		ScalarGrid<double> testGrid(xform, size, ScalarGridSettings::SampleType::CENTER, ScalarGridSettings::BorderType::CLAMP);

		forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
		{
			Vec2d worldPosition = testGrid.indexToWorld(cell.cast<double>());
			testGrid(cell) = (worldPosition - center).norm();
		});

		testGrid.drawGrid("testGrid");
		testGrid.drawSamplePoints("testGrid", Vec3d(1, 0, 0), .001);
		testGrid.drawSupersampledValues("testGrid", .5, 3, .001);
	}
	else if (testIndex == 1)
	{
		VectorGrid<double> testVectorGrid(xform, size, VectorGridSettings::SampleType::STAGGERED, ScalarGridSettings::BorderType::CLAMP);

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i::Zero(), testVectorGrid.size(axis), [&](const Vec2i& cell)
			{
				Vec2d offset = Vec2d::Zero(); offset[axis] += .5;
				Vec2d worldPosition0 = testVectorGrid.indexToWorld(cell.cast<double>() - offset, axis);
				Vec2d worldPosition1 = testVectorGrid.indexToWorld(cell.cast<double>() + offset, axis);

				double gradient = ((worldPosition1 - center).norm() - (worldPosition0 - center).norm()) / dx;
				testVectorGrid(cell, axis) = gradient;
			});
		}

		testVectorGrid.drawGrid("testVectorGrid");
		testVectorGrid.drawSamplePoints("testVectorGrid", Vec3d(1, 0, 0), Vec3d(0, 1, 0), Vec2d(.001, .001));
		testVectorGrid.drawSamplePointVectors("testVectorGrid", Vec3d::Zero(), 1.0);
	}
	else if (testIndex == 2)
	{
		LevelSet testLevelSet(xform, size, 5);

		double radius = .3 * (topRightCorner - center).norm();
		forEachVoxelRange(Vec2i::Zero(), testLevelSet.size(), [&](const Vec2i& cell)
		{
			Vec2d worldPosition = testLevelSet.indexToWorld(cell.cast<double>());
			testLevelSet(cell) = (worldPosition - center).norm() - radius;
		});

		testLevelSet.reinit(true);
		testLevelSet.drawGrid("testLevelSet", true);
		testLevelSet.drawSupersampledValues("testLevelSet", .5, 3, .001);
		testLevelSet.drawNormals("testLevelSet", Vec3d::Zero(), 1.);
		testLevelSet.drawSurface("testLevelSet", Vec3d(1, 0, 0), .001);
	}
}

int main()
{
	double dx = .5;
	Vec2d topRightCorner(20, 20);
	Vec2d bottomLeftCorner(-20, -20);
	Vec2i size = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
	Transform xform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	registerTest(gCurrentTest, xform, size, center, topRightCorner, dx);

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Next test"))
		{
			gCurrentTest = (gCurrentTest + 1) % 3;
			registerTest(gCurrentTest, xform, size, center, topRightCorner, dx);
		}
		ImGui::Text("Current test: %s", gTestNames[gCurrentTest]);
	};

	polyscope::show();
}
