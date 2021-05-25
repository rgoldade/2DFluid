#include <random>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "gtest/gtest.h"

#include "GridUtilities.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

static void testSampleType(const VectorGridSettings::SampleType sampleType)
{
	double dx = .01;
	Vec2d origin(-1., -1.);
	Vec2i cellSize(100, 100);

	Transform xform(dx, origin);

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	EXPECT_EQ(testGrid.sampleType(), sampleType);

	EXPECT_TRUE(testGrid.gridSize() == cellSize);

	// Test sample count related to the grid size 
	switch (sampleType)
	{
	case VectorGridSettings::SampleType::CENTER:
	{
		EXPECT_TRUE(testGrid.size(0) == cellSize);
		EXPECT_TRUE(testGrid.size(1) == cellSize);
		break;
	}
	case VectorGridSettings::SampleType::STAGGERED:
	{
		EXPECT_TRUE(testGrid.size(0) == (cellSize + Vec2i(1, 0)).eval());
		EXPECT_TRUE(testGrid.size(1) == (cellSize + Vec2i(0, 1)).eval());
		break;
	}
	case VectorGridSettings::SampleType::NODE:
	{
		EXPECT_TRUE(testGrid.size(0) == (cellSize + Vec2i::Ones()).eval());
		EXPECT_TRUE(testGrid.size(1) == (cellSize + Vec2i::Ones()).eval());
	}
	}

	// Test index-to-world and back
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), testGrid.size(axis), [&](const Vec2i& coord)
			{
				Vec2d indexPoint = testGrid.worldToIndex(testGrid.indexToWorld(coord.cast<double>(), axis), axis);
				EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
				EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
			});
	}

	// Test sampling
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), testGrid.size(axis), [&](const Vec2i& coord)
		{
			Vec2d worldPoint;
			switch (sampleType)
			{
				case VectorGridSettings::SampleType::CENTER:
				{
					worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d::Ones());
					break;
				}
				case VectorGridSettings::SampleType::STAGGERED:
				{
					if (axis == 0)
						worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d(0., 1.));
					else
						worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec2d(1., 0.));
					break;
				}
				case VectorGridSettings::SampleType::NODE:
				{
					worldPoint = origin + dx * coord.cast<double>();
				}
			}

			Vec2d indexPoint = testGrid.worldToIndex(worldPoint, axis);

			EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
		});
	}

	// Copy test
	VectorGrid<double> copyGrid = testGrid;

	EXPECT_TRUE(copyGrid.isGridMatched(testGrid));

	// Same values test
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), testGrid.size(axis), [&](const Vec2i& coord)
		{
			EXPECT_EQ(copyGrid(coord, axis), testGrid(coord, axis));
		});
	}

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), copyGrid.size(axis), [&](const Vec2i& coord)
		{
			copyGrid(coord, axis) += 5.;
		});
	}

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), testGrid.size(axis), [&](const Vec2i& coord)
		{
			EXPECT_NE(copyGrid(coord, axis), testGrid(coord, axis));
		});
	}

	// Transform test
	EXPECT_EQ(testGrid.dx(), xform.dx());
	EXPECT_EQ(testGrid.offset()[0], xform.offset()[0]);
    EXPECT_EQ(testGrid.offset()[1], xform.offset()[1]);
	EXPECT_EQ(testGrid.xform(), xform);
}

TEST(VECTOR_GRID_TESTS, CENTER_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, XFACE_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, YFACE_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::NODE);
}

// Min/max tests
TEST(VECTOR_GRID_TESTS, MIN_MAX_TEST)
{
	double dx = .01;
	Vec2d origin(-1., -1.);
	Vec2i cellSize(100, 100);

	Transform xform(dx, origin);
	VectorGrid<double> testGrid(xform, cellSize, 1.);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-100., 100.);

	for (int axis : {0, 1})
	{
		double minValue = std::numeric_limits<double>::max();
		double maxValue = std::numeric_limits<double>::lowest();

		forEachVoxelRange(Vec2i::Zero(), testGrid.size(axis), [&](const Vec2i& coord)
		{
			double value = distribution(generator);
			minValue = std::min(value, minValue);
			maxValue = std::max(value, maxValue);
			testGrid(coord, axis) = value;
		});

		EXPECT_EQ(minValue, testGrid.grid(axis).minValue());
		EXPECT_EQ(maxValue, testGrid.grid(axis).maxValue());

		auto minMaxPair = testGrid.grid(axis).minAndMaxValue();

		EXPECT_EQ(minValue, minMaxPair.first);
		EXPECT_EQ(maxValue, minMaxPair.second);
	}
}

static void readWriteTest(const VectorGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(64, 64);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	Transform xform(dx, origin);

	auto testFunc = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), testGrid.grid(axis).size(), [&](const Vec2i& coord)
		{
			Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
			double val = testFunc(point)[axis];
			double storedVal = testGrid(coord, axis);

			EXPECT_EQ(val, storedVal);
		});
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::NODE);
}

// Component interpolation test

static double componentInterpolationErrorTest(const Transform& xform, const Vec2i& cellSize, const VectorGridSettings::SampleType sampleType)
{
	auto testFunc = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	double error = 0;
	for (int axis : {0, 1})
	{
		double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), double(0),
			[&](const tbb::blocked_range<int>& range, double error) -> double
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);

					if (coord[0] == testGrid.size(axis)[0] - 1 || coord[1] == testGrid.size(axis)[1] - 1)
						continue;

					Vec2d startPoint = coord.cast<double>();
					Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

					Vec2d point;
					for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
						for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
						{
							Vec2d worldPoint = testGrid.indexToWorld(point, axis);

							double localError = std::fabs(testGrid.biLerp(worldPoint, axis) - testFunc(worldPoint)[axis]);
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

		error = std::max(error, localError);
	}

	return error;
}

static void componentInterpolationTest(const VectorGridSettings::SampleType sampleType)
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

		double localError = componentInterpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.85);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::NODE);
}

// Vector interpolation test

static double vectorInterpolationErrorTest(const Transform& xform, const Vec2i& cellSize, const VectorGridSettings::SampleType sampleType)
{
	auto testFunc = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount()), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
					Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
					testGrid(coord, axis) = testFunc(point)[axis];
				}
			});
	}

	double error = 0;
	for (int axis : {0, 1})
	{
		double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount()), double(0),
			[&](const tbb::blocked_range<int>& range, double error) -> double
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);

					if (coord[0] == testGrid.size(axis)[0] - 1 || coord[1] == testGrid.size(axis)[1] - 1)
						continue;

					Vec2d startPoint = coord.cast<double>();
					Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

					Vec2d point;
					for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
						for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
						{
							Vec2d worldPoint = testGrid.indexToWorld(point, axis);

							// For staggered grids, this world point could fall outside of the sample range for the other grid
							if (testGrid.sampleType() == VectorGridSettings::SampleType::STAGGERED)
							{
								Vec2d staggeredIndexPoint = testGrid.worldToIndex(worldPoint, (axis + 1) % 2);
								if (staggeredIndexPoint[0] < 0 || staggeredIndexPoint[1] < 0 ||
									staggeredIndexPoint[0] > testGrid.size((axis + 1) % 2)[0] - 1 ||
									staggeredIndexPoint[1] > testGrid.size((axis + 1) % 2)[0] - 1)
									continue;
							}

							double localError = (testGrid.biLerp(worldPoint) - testFunc(worldPoint)).norm();
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

		error = std::max(error, localError);
	}

	return error;
}

static void vectorInterpolationTest(const VectorGridSettings::SampleType sampleType)
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

		double localError = vectorInterpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.8);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::NODE);
}

// Component cubic interpolation test

static double componentCubicInterpolationErrorTest(const Transform& xform, const Vec2i& cellSize, const VectorGridSettings::SampleType sampleType, const Vec2i& startIndex, const Vec2i& endIndex)
{
	auto testFunc = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	double error = 0;
	for (int axis : {0, 1})
	{
		double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), double(0),
			[&](const tbb::blocked_range<int>& range, double error) -> double
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);

					if (coord[0] < startIndex[0] || coord[1] < startIndex[1] ||
						coord[0] >= endIndex[0] || coord[1] >= endIndex[1])
						continue;

					Vec2d startPoint = coord.cast<double>();
					Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

					Vec2d point;
					for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
						for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
						{
							Vec2d worldPoint = testGrid.indexToWorld(point, axis);

							double localError = std::fabs(testGrid.biCubicInterp(worldPoint, axis) - testFunc(worldPoint)[axis]);
							error = std::max(error, localError);
						}
				}

				return error;
			},
			[](double a, double b) -> double
			{
				return std::max(a, b);
			});

		error = std::max(error, localError);
	}

	return error;
}

static void componentCubicInterpolationTest(const VectorGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 6;
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

		double localError = componentCubicInterpolationErrorTest(xform, localCellSize, sampleType, localStartIndex, localEndIndex);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.85);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_COMPONENT_CUBIC_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_COMPONENT_CUBIC_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_COMPONENT_CUBIC_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::NODE);
}

// Vector interpolation test

static double vectorCubicInterpolationErrorTest(const Transform& xform, const Vec2i& cellSize, const VectorGridSettings::SampleType sampleType, const Vec2i& startIndex, const Vec2i& endIndex)
{
	auto testFunc = [](const Vec2d& point) -> Vec2d
	{
		return Vec2d(PI * std::cos(PI * point[0]) * std::cos(PI * point[1]),
			-PI * std::sin(PI * point[0]) * std::sin(PI * point[1]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec2d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	double error = 0;
	for (int axis : {0, 1})
	{
		double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount()), double(0),
			[&](const tbb::blocked_range<int>& range, double error) -> double
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i coord = testGrid.grid(axis).unflatten(cellIndex);
					
					if (coord[0] < startIndex[0] || coord[1] < startIndex[1] ||
						coord[0] >= endIndex[0] || coord[1] >= endIndex[1])
						continue;

					Vec2d startPoint = coord.cast<double>();
					Vec2d endPoint = (coord + Vec2i::Ones()).cast<double>();

					Vec2d point;
					for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
						for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
						{
							Vec2d worldPoint = testGrid.indexToWorld(point, axis);

							double localError = (testGrid.biCubicInterp(worldPoint) - testFunc(worldPoint)).norm();
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

		error = std::max(error, localError);
	}

	return error;
}

static void vectorCubicInterpolationTest(const VectorGridSettings::SampleType sampleType)
{
	Vec2d origin(-1., -1.);
	Vec2d topCorner(1., 1.);
	Vec2i cellSize(16, 16);
	double dx = (topCorner[0] - origin[0]) / double(cellSize[0]);

	int testSize = 6;
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

		double localError = vectorCubicInterpolationErrorTest(xform, localCellSize, sampleType, localStartIndex, localEndIndex);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.8);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_VECTOR_CUBIC_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_VECTOR_CUBIC_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_VECTOR_CUBIC_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::NODE);
}