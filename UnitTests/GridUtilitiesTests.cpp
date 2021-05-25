#include <array>
#include <numeric>

#include <gtest/gtest.h>

#include "GridUtilities.h"
#include "Utilities.h"

using namespace FluidSim2D;

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_HALF_TEST)
{
	{
		double theta = lengthFraction<double>(5., -5.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}

	{
		double theta = lengthFraction<double>(-5., 5.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}
	
	{
		double theta = lengthFraction<double>(1., -1.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}

	{
		double theta = lengthFraction<double>(-1., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}
}

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_ZERO_TEST)
{
	{
		double theta = lengthFraction<double>(0., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}

	{
		double theta = lengthFraction<double>(1., 0.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}

	{
		double theta = lengthFraction<double>(5., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}
}

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_ONE_TEST)
{
	{
		double theta = lengthFraction<double>(-1., 0.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}

	{
		double theta = lengthFraction<double>(0., -1.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}

	{
		double theta = lengthFraction<double>(-1., -5.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_CELL_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i cell = (10000. * Vec2d::Random()).cast<int>();

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);
				Vec2i returnCell = cellToCell(adjacentCell, axis, (direction + 1) % 2);

				EXPECT_TRUE(cell == returnCell);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_FACE_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i cell = (10000. * Vec2d::Random()).cast<int>();

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentFace = cellToFace(cell, axis, direction);
				Vec2i returnCell = faceToCell(adjacentFace, axis, (direction + 1) % 2);

				EXPECT_TRUE(cell == returnCell);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_FACE_CCW_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i cell = (10000. * Vec2d::Random()).cast<int>();

		for (int faceIndex = 0; faceIndex < 4; ++faceIndex)
		{
			Vec3i adjacentFace = cellToFaceCCW(cell, faceIndex);
			Vec2i face(adjacentFace[0], adjacentFace[1]);
			int axis = adjacentFace[2];

			std::array<Vec2i, 2> returnCells = { faceToCell(face, axis, 0), faceToCell(face, axis, 1) };

			EXPECT_TRUE(cell == returnCells[0] ^ cell == returnCells[1]);

		}
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_NODE_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i cell = (10000. * Vec2d::Random()).cast<int>();

		for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
		{
			Vec2i node = cellToNode(cell, nodeIndex);

			{
				int matchedCellCount = 0;
				for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
				{
					Vec2i returnedCell = nodeToCell(node, cellIndex);
					matchedCellCount += cell == returnedCell;
				}

				EXPECT_EQ(matchedCellCount, 1);
			}

			{
				int matchedCellCount = 0;
				for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
				{
					Vec2i returnedCell = nodeToCellCCW(node, cellIndex);
					matchedCellCount += cell == returnedCell;
				}

				EXPECT_EQ(matchedCellCount, 1);
			}
		}

		for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
		{
			Vec2i node = cellToNodeCCW(cell, nodeIndex);

			{
				int matchedCellCount = 0;
				for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
				{
					Vec2i returnedCell = nodeToCell(node, cellIndex);
					matchedCellCount += cell == returnedCell;
				}

				EXPECT_EQ(matchedCellCount, 1);
			}

			{
				int matchedCellCount = 0;
				for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
				{
					Vec2i returnedCell = nodeToCellCCW(node, cellIndex);
					matchedCellCount += cell == returnedCell;
				}

				EXPECT_EQ(matchedCellCount, 1);
			}
		}
	}
}

TEST(GRID_UTILITIES_TESTS, FACE_TO_NODE)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec2i face = (10000. * Vec2d::Random()).cast<int>();

		for (int faceAxis : {0, 1})
		{
			for (int direction : {0, 1})
			{
                Vec2i node = faceToNode(face, faceAxis, direction);

				int offsetAxis = (faceAxis + 1) % 2;
                Vec2i returnFace = nodeToFace(node, offsetAxis, (direction + 1) % 2);

				EXPECT_EQ(face[0], returnFace[0]);
                EXPECT_EQ(face[1], returnFace[1]);
			}
		}
	}
}