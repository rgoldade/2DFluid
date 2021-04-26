#include <random>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "gtest/gtest.h"

#include "GridUtilities.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"


using namespace FluidSim2D;

static void testSampleType(const ScalarGridSettings::SampleType sampleType)
{
	double dx = .01;
	Vec2d origin(-1., -1.);
	Vec2i cellSize(100, 100);

	Transform xform(dx, origin);

	ScalarGrid<double> testGrid(xform, cellSize, sampleType);

	EXPECT_TRUE(testGrid.xform() == xform);
	EXPECT_EQ(testGrid.sampleType(), sampleType);

	// Test sample count related to the grid size 
	switch (sampleType)
	{
		case ScalarGridSettings::SampleType::CENTER:
		{
			// CEnter
			EXPECT_TRUE(testGrid.size() == cellSize);
			break;
		}
		case ScalarGridSettings::SampleType::XFACE:
		{
			// X-face
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec2i(1, 0)).eval());
			break;
		}
		case ScalarGridSettings::SampleType::YFACE:
		{
			// Y-face
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec2i(0, 1)).eval());
			break;
		}
		case ScalarGridSettings::SampleType::NODE:
		{
			// Node
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec2i::Ones()).eval());
		}
	}

	// Test index-to-world and back
	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& coord)
	{
		Vec2d indexPoint = testGrid.worldToIndex(testGrid.indexToWorld(coord.cast<double>()));
		EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
		EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
	});
		
	// Test sampling
	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& coord)
	{
		Vec2d worldPoint;
		switch (sampleType)
		{
			case ScalarGridSettings::SampleType::CENTER:
			{
				worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d::Ones());
				break;
			}
			case ScalarGridSettings::SampleType::XFACE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d(0., 1.));
				break;
			}
			case ScalarGridSettings::SampleType::YFACE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d(1., 0.));
				break;
			}
			case ScalarGridSettings::SampleType::NODE:
			{
				worldPoint = origin + dx * coord.cast<double>();
			}
		}

		Vec2d indexPoint = testGrid.worldToIndex(worldPoint);

		EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false)) << "Sample type: " << int(sampleType) << ". Index point: " << indexPoint[0] << " " << indexPoint[1] << ". Coord point: " << coord[0] << " " << coord[1];
		EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false)) << "Sample type: " << int(sampleType) << ". Index point: " << indexPoint[0] << " " << indexPoint[1] << ". Coord point: " << coord[0] << " " << coord[1];
	});

	// Copy test
	ScalarGrid<double> copyGrid = testGrid;

	EXPECT_TRUE(copyGrid.isGridMatched(testGrid));

	// Same values test
	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& coord)
	{
		EXPECT_EQ(copyGrid(coord), testGrid(coord));
	});

	forEachVoxelRange(Vec2i::Zero(), copyGrid.size(), [&](const Vec2i& coord)
	{
		copyGrid(coord) += 5.;
	});

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& coord)
	{
		EXPECT_NE(copyGrid(coord), testGrid(coord));
	});

	// Transform test
	EXPECT_EQ(testGrid.dx(), xform.dx());
	EXPECT_EQ(testGrid.offset(), xform.offset());
	EXPECT_EQ(testGrid.xform(), xform);
}

TEST(SCALAR_GRID_TESTS, CENTER_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, NODE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::NODE);
}

// Min/max tests
TEST(SCALAR_GRID_TESTS, MIN_MAX_TEST)
{
	double dx = .01;
	Vec2d origin(-1., -1.);
	Vec2i cellSize(100, 100);

	Transform xform(dx, origin);
	ScalarGrid<double> testGrid(xform, cellSize, 1.);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-100., 100.);

	double minValue = std::numeric_limits<double>::max();
	double maxValue = std::numeric_limits<double>::lowest();

	forEachVoxelRange(Vec2i::Zero(), testGrid.size(), [&](const Vec2i& cell)
	{
		double value = distribution(generator);
		minValue = std::min(value, minValue);
		maxValue = std::max(value, maxValue);
		testGrid(cell) = value;
	});

	EXPECT_EQ(minValue, testGrid.minValue());
	EXPECT_EQ(maxValue, testGrid.maxValue());

	auto minMaxPair = testGrid.minAndMaxValue();

	EXPECT_EQ(minValue, minMaxPair.first);
	EXPECT_EQ(maxValue, minMaxPair.second);
}

// Interpolation test

static double interpolationErrorTest(const Transform& xform, const Vec2i& cellSize, const ScalarGridSettings::SampleType sampleType)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec2d& point) -> double
	{
		return std::sin(point[0] * PI) * std::cos(point[1] * PI);
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i coord = grid.unflatten(cellIndex);
			Vec2d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = grid.unflatten(cellIndex);
				
				if (coord[0] == grid.size()[0] - 1 || coord[1] == grid.size()[1] - 1)
					continue;

				Vec2d startPoint = coord.cast<double>();
				Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

				Vec2d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
					{
						Vec2d worldPoint = grid.indexToWorld(point);
						double localError = std::fabs(grid.biLerp(worldPoint) - testFunc(worldPoint));
						error = std::max(error, localError);
					}
			}

			return error;
		},
		[](double a, double b) -> double
		{
			return std::max(a, b);
		}
	);

	return error;
}

static void interpolationTest(const ScalarGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 6;
	std::vector<double> errors;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		double localError = interpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.85);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, NODE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::NODE);
}

// Cubic interpolation test

static double cubicInterpolationError(const Transform& xform, const Vec2i& cellSize, const ScalarGridSettings::SampleType sampleType, const Vec2i& startIndex, const Vec2i& endIndex)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec2d& point) -> double
	{
		return std::sin(point[0] * PI) * std::cos(point[1] * PI);
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i coord = grid.unflatten(cellIndex);
			Vec2d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = grid.unflatten(cellIndex);

				if (coord[0] < startIndex[0] || coord[1] < startIndex[1] ||
					coord[0] >= endIndex[0] || coord[1] >= endIndex[1])
					continue;

				Vec2d startPoint = coord.cast<double>();
				Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

				Vec2d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
					{
						Vec2d worldPoint = grid.indexToWorld(point);
						double localError = std::fabs(grid.biCubicInterp(worldPoint) - testFunc(worldPoint));
						error = std::max(error, localError);
					}
			}

			return error;
		},
		[](double a, double b) -> double
		{
			return std::max(a, b);
		});

	return error;
}

static void cubicInterpolationTest(const ScalarGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 5;
	std::vector<double> errors;
	Vec2i startIndex = Vec2i::Ones();
	Vec2i endIndex = cellSize - Vec2i::Ones();
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		Vec2i localStartIndex = int(std::pow(2, testIndex)) * startIndex;
		Vec2i localEndIndex = int(std::pow(2, testIndex)) * endIndex;

		double localError = cubicInterpolationError(xform, localCellSize, sampleType, localStartIndex, localEndIndex);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 8.);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_CUBIC_INTERPLOTATION_TEST)
{
	cubicInterpolationTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_CUBIC_INTERPLOTATION_TEST)
{
	cubicInterpolationTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_CUBIC_INTERPLOTATION_TEST)
{
	cubicInterpolationTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, NODE_CUBIC_INTERPLOTATION_TEST)
{
	cubicInterpolationTest(ScalarGridSettings::SampleType::NODE);
}

static double gradientErrorTest(const Transform& xform, const Vec2i& cellSize, const ScalarGridSettings::SampleType sampleType)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec2d& point) -> double
	{
		return std::sin(point[0] * PI) * std::cos(point[1] * PI);
	};

	auto testFuncGradient = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i coord = grid.unflatten(cellIndex);
			Vec2d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = grid.unflatten(cellIndex);

				if (coord[0] == grid.size()[0] - 1 || coord[1] == grid.size()[1] - 1)
					continue;

				Vec2d startPoint = coord.cast<double>();
				Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

				Vec2d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
					{
						Vec2d worldPoint = grid.indexToWorld(point);
						Vec2d lerpGrad = grid.biLerpGradient(worldPoint);
						Vec2d funcGrad = testFuncGradient(worldPoint);
						double localError = (lerpGrad - funcGrad).norm();
						error = std::max(localError, error);
					}
			}

			return error;
		},
		[](double a, double b) -> double
		{
			return std::max(a, b);
		});

	return error;
}

static void gradientTest(const ScalarGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 5;
	std::vector<double> errors;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		double localError = gradientErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 1.9);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, NODE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::NODE);
}

static double cubicGradientErrorTest(const Transform& xform, const Vec2i& cellSize, const ScalarGridSettings::SampleType sampleType, const Vec2i& startIndex, const Vec2i& endIndex)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec2d& point) -> double
	{
		return std::sin(point[0] * PI) * std::cos(point[1] * PI);
	};

	auto testFuncGradient = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = grid.unflatten(cellIndex);
				Vec2d point = grid.indexToWorld(coord.cast<double>());
				grid(coord) = testFunc(point);
			}
		});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = grid.unflatten(cellIndex);
				
				if (coord[0] < startIndex[0] || coord[1] < startIndex[1] ||
					coord[0] >= endIndex[0] || coord[1] >= endIndex[1])
					continue;

				Vec2d startPoint = coord.cast<double>();
				Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

				Vec2d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
					{
						Vec2d worldPoint = grid.indexToWorld(point);
						Vec2d cubicGrad = grid.biCubicGradient(worldPoint);
						Vec2d funcGrad = testFuncGradient(worldPoint);
						double localError = (cubicGrad - funcGrad).norm();
						error = std::max(localError, error);
					}
			}

			return error;
		},
		[](double a, double b) -> double
		{
			return std::max(a, b);
		});

	return error;
}

static void cubicGradientTest(const ScalarGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 5;
	std::vector<double> errors;
	Vec2i startIndex = Vec2i::Ones();
	Vec2i endIndex = cellSize - Vec2i::Ones();
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		Vec2i localStartIndex = int(std::pow(2, testIndex)) * startIndex;
		Vec2i localEndIndex = int(std::pow(2, testIndex)) * endIndex;

		double localError = cubicGradientErrorTest(xform, localCellSize, sampleType, localStartIndex, localEndIndex);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.85);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_CUBIC_GRADIENT_TEST)
{
	cubicGradientTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_CUBIC_GRADIENT_TEST)
{
	cubicGradientTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_CUBIC_GRADIENT_TEST)
{
	cubicGradientTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, NODE_CUBIC_GRADIENT_TEST)
{
	cubicGradientTest(ScalarGridSettings::SampleType::NODE);
}

