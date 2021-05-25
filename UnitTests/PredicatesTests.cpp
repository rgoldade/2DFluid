#include "gtest/gtest.h"

#include "Predicates.h"
#include "Utilities.h"

using namespace FluidSim2D;

TEST(PREDICATES_TEST, ORIENT_2D_ZERO_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		Vec2d testPoint = startPoint;
		
		EXPECT_EQ(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint;
		EXPECT_EQ(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_2D_POSITIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		if (startPoint == endPoint)
			continue;

		Vec2d vec = endPoint - startPoint;
		Vec2d norm(-vec[1], vec[0]);

		double offsetScalar = 1e-12;
		Vec2d testPoint = startPoint + offsetScalar * norm;

		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint + offsetScalar * norm;
		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = .5 * (startPoint + endPoint) + offsetScalar * norm;
		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_2D_NEGATIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		if (startPoint == endPoint)
			continue;

		Vec2d vec = endPoint - startPoint;
		Vec2d norm(-vec[1], vec[0]);

		double offsetScalar = -1e-12;
		Vec2d testPoint = startPoint + offsetScalar * norm;

		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint + offsetScalar * norm;
		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = .5 * (startPoint + endPoint) + offsetScalar * norm;
		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_CROSSING_EDGE_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			Vec2d offset = 1e-12 * rayStart.array().abs();

			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				if (axis == Axis::XAXIS)
					startPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				else
					startPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				Vec2d endPoint = startPoint;
				if (axis == Axis::XAXIS)
				{
					startPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
					endPoint[1] -= offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}
				else
				{
					startPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
					endPoint[0] -= offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::YES);
	
				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::YES);
			}
		}
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_CROSSING_POINT_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			double offset = 1e-12 * std::max(std::fabs(rayStart[0]), std::fabs(rayStart[1]));

			// Start down-left, end up-right
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				startPoint += Vec2d(-offset, -offset);
				endPoint += Vec2d(offset, offset);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
			}

			// Start up-left, end down-right
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				startPoint += Vec2d(-offset, offset);
				endPoint += Vec2d(offset, -offset);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
			}

			// Start up, end down
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				if (axis == Axis::XAXIS)
				{
					startPoint[1] += offset;
					endPoint[1] -= offset;
				}
				else
				{
					startPoint[0] += offset;
					endPoint[0] -= offset;
				}

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
			}
		}
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_DEGENERATE_EDGE_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			Vec2d offset = 1e-12 * rayStart.array().abs();
			{
				Vec2d startPoint = rayStart;
				if (axis == Axis::XAXIS)
					startPoint[0] += offset[0];
				else
					startPoint[1] += offset[1];

				Vec2d endPoint = startPoint;

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}

			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				if (axis == Axis::XAXIS)
				{
					startPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
					endPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}
				else
				{
					startPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
					endPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}
		}
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_DEGENERATE_POINT_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			Vec2d offset = 1e-12 * rayStart.array().abs();
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = startPoint;

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}

			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				if (axis == Axis::XAXIS)
				{
					endPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}
				else
				{
					endPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				}

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}
		}
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_JITTER_EDGE_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			Vec2d offset = 1e-12 * rayStart.array().abs();

			// Start point on grid edge, end point above
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				if (axis == Axis::XAXIS)
					startPoint[0] += offset[0];
				else
					startPoint[1] += offset[1];

				Vec2d endPoint = startPoint;

				if (axis == Axis::XAXIS)
					endPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				else
					endPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::YES);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::YES);
			}

			// Start point on grid edge, end point below
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				if (axis == Axis::XAXIS)
					startPoint[0] += offset[0];
				else
					startPoint[1] += offset[1];

				Vec2d endPoint = startPoint;

				if (axis == Axis::XAXIS)
					endPoint[1] -= offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				else
					endPoint[0] -= offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}
		}
	}
}

TEST(PREDICATES_TEST, EXACT_EDGE_INTERSECTION_JITTER_POINT_TEST)
{
	exactinit();

	int testCases = 1000;
	int offsetCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d rayStart = 1e5 * Vec2d::Random();
		Vec2d offset = 1e-12 * rayStart.array().abs();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS })
		{
			// Start point on ray start, end point above
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				if (axis == Axis::XAXIS)
					endPoint[1] += offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				else
					endPoint[0] += offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
			}

			// Start point on grid edge, end point below
			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec2d startPoint = rayStart;
				Vec2d endPoint = rayStart;

				if (axis == Axis::XAXIS)
					endPoint[1] -= offset[1] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
				else
					endPoint[0] -= offset[0] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);

				std::swap(startPoint, endPoint);

				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::NO);
			}
		}
	}
}