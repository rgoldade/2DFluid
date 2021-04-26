#include "gtest/gtest.h"

#include "UniformGrid.h"

using namespace FluidSim2D;

template<typename T, typename ValFunc>
static void fillUniformGrid(const ValFunc& valFunc, UniformGrid<T>& grid)
{
	forEachVoxelRange(Vec2i::Zero(), grid.size(), [&](const Vec2i& cell)
	{
		grid(cell) = valFunc(cell);
	});
}

TEST(UNIFORM_GRID_TESTS, CONSTRUCTOR_SIZE_TEST)
{
	Vec2i size(75, 100);
	UniformGrid<int> testGrid(size);

	EXPECT_EQ(size, testGrid.size());
}

TEST(UNIFORM_GRID_TESTS, CONSTRUCTOR_SIZE_AND_VALUE_TEST)
{
	Vec2i size(75, 100);
	double val = 10.;
	UniformGrid<int> testGrid(size, val);

	EXPECT_EQ(size, testGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(val, testGrid(cell));
		EXPECT_EQ(val, testGrid(cell[0], cell[1]));
	});
}

TEST(UNIFORM_GRID_TESTS, STORAGE_ROUND_TRIP)
{
	Vec2i size(75, 100);

	auto valFunc = [](const Vec2i& cell) -> double
	{
		return double(cell[0]) * double(cell[1]);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
		EXPECT_EQ(valFunc(cell), testGrid(cell[0], cell[1]));
	});
}

TEST(UNIFORM_GRID_TESTS, CLEAR_TEST)
{
	Vec2i size(75, 100);
	UniformGrid<double> testGrid(size);

	testGrid.clear();
	EXPECT_TRUE(testGrid.empty());
	EXPECT_EQ(testGrid.size(), Vec2i::Zero());
}

TEST(UNIFORM_GRID_TESTS, RESIZE_LARGER_TEST)
{
	Vec2i size(75, 100);

	auto valFunc = [](const Vec2i& cell) -> double
	{
		return double(cell[0]) * double(cell[1]);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	// Resize grid
	Vec2i expandSize = 2 * size;
	testGrid.resize(expandSize);
	fillUniformGrid<double>(valFunc, testGrid);

	EXPECT_EQ(expandSize, testGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESIZE_SMALLER_TEST)
{
	Vec2i size(75, 100);

	auto valFunc = [](const Vec2i& cell) -> double
	{
		return double(cell[0]) * double(cell[1]);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	// Resize grid
	Vec2i shrinkSize(size[0] / 2, size[1] / 2);
	testGrid.resize(shrinkSize);
	fillUniformGrid<double>(valFunc, testGrid);

	EXPECT_EQ(shrinkSize, testGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESIZE_VALUE_TEST)
{
	Vec2i size(75, 100);
	double startValue = 10.;
	UniformGrid<double> testGrid(size, startValue);

	// Resize grid
	Vec2i expandSize = 2 * size;
	double expandValue = 100.;
	testGrid.resize(expandSize, expandValue);

	EXPECT_EQ(expandSize, testGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(expandValue, testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESET_TEST)
{
	Vec2i size(75, 100);
	double startValue = 10.;
	UniformGrid<double> testGrid(size, startValue);

	double resetValue = 100.;
	testGrid.reset(resetValue);

	EXPECT_EQ(size, testGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(resetValue, testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, VOXEL_COUNT_TEST)
{
	Vec2i size(75, 100);
	UniformGrid<double> testGrid(size);

	EXPECT_EQ(size[0] * size[1], testGrid.voxelCount());
}

TEST(UNIFORM_GRID_TESTS, FLATTEN_UNFLATTEN_TEST)
{
	int testSize = 100;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i size = (100. * (Vec2d::Random() + Vec2d::Ones())).cast<int>();

		if (size[0] <= 0 || size[1] <= 0)
			continue;

		UniformGrid<double> testGrid(size);

		forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
		{
			EXPECT_EQ(cell, testGrid.unflatten(testGrid.flatten(cell)));
		});
	}
}

TEST(UNIFORM_GRID_TESTS, COPY_GRID_TEST)
{
	Vec2i size(75, 100);
	UniformGrid<int> testGrid(size, 10.);

	UniformGrid<int> copyGrid = testGrid;

	EXPECT_EQ(testGrid.size(), copyGrid.size());

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(testGrid(cell), copyGrid(cell));
	});

	copyGrid.reset(15.);

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		EXPECT_NE(testGrid(cell), copyGrid(cell));
	});
}