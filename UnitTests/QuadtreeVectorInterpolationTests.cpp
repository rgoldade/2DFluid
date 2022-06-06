#include "gtest/gtest.h"

#include "QuadtreeGrid.h"
#include "QuadtreeVectorInterpolator.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

class AnalyticalQuadtreeInterpolation
{
private:

	static constexpr int UNASSIGNED = -1;

public:
	template<typename Refinement>
	AnalyticalQuadtreeInterpolation(const Transform& xform, const Vec2i& size, int levels, const Refinement& refiner)
		: myQuadtree(xform, size, levels)
	{
		// Build quadtree according to the grid spacing, size and refinement criteria
		myQuadtree.buildTree(refiner);

		int quadtreeLevels = myQuadtree.levels();
		myQuadtreeVelocityIndex.resize(quadtreeLevels);
		myQuadtreeVelocity.resize(quadtreeLevels);

		myQuadtreeXforms.resize(quadtreeLevels);

		for (int level = 0; level < quadtreeLevels; ++level)
		{
			Transform localXform = myQuadtree.xform(level);

			myQuadtreeXforms[level] = localXform;

			Vec2i localGridSize = myQuadtree.size(level);
			myQuadtreeVelocityIndex[level] = VectorGrid<int>(localXform, localGridSize, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
			myQuadtreeVelocity[level] = VectorGrid<double>(localXform, localGridSize, 0, VectorGridSettings::SampleType::STAGGERED);
		}
	}

	void refine()
	{
		myQuadtree.refineGrid();

		myQuadtreeVelocityIndex.clear();

		unsigned quadtreeLevels = myQuadtree.levels();
		myQuadtreeVelocityIndex.resize(quadtreeLevels);
		myQuadtreeVelocity.resize(quadtreeLevels);

		myQuadtreeXforms.resize(quadtreeLevels);

		for (unsigned level = 0; level < quadtreeLevels; ++level)
		{
			Transform localXform = myQuadtree.xform(level);

			myQuadtreeXforms[level] = localXform;

			Vec2i localGridSize = myQuadtree.size(level);
			myQuadtreeVelocityIndex[level] = VectorGrid<int>(localXform, localGridSize, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
			myQuadtreeVelocity[level] = VectorGrid<double>(localXform, localGridSize, 0, VectorGridSettings::SampleType::STAGGERED);
		}
	}

	template<typename TestFunctor>
	double solve(TestFunctor& testFunctor);


private:

	int buildQuadtreeVelocityIndex();

	QuadtreeGrid myQuadtree;

	std::vector<Transform> myQuadtreeXforms;
	std::vector<VectorGrid<int>> myQuadtreeVelocityIndex;
	std::vector<VectorGrid<double>> myQuadtreeVelocity;
};

template<typename TestFunctor>
double AnalyticalQuadtreeInterpolation::solve(TestFunctor& testFunctor)
{
	// Set active faces according to active cells
	int velocityDOFCount = buildQuadtreeVelocityIndex();

	int quadtreeLevels = myQuadtree.levels();

	// Apply the test field to active velocity faces
	for (int level = 0; level < quadtreeLevels; ++level)
	{
		for (int axis = 0; axis < 2; ++axis)
		{
			forEachVoxelRange(Vec2i::Zero(), myQuadtreeVelocityIndex[level].size(axis), [&](const Vec2i& face)
			{
				int velocityIndex = myQuadtreeVelocityIndex[level](face, axis);
				if (velocityIndex >= 0)
				{
					Vec2d velocityPoint = myQuadtreeVelocity[level].indexToWorld(face.cast<double>(), axis);
					myQuadtreeVelocity[level](face, axis) = testFunctor(velocityPoint, axis);
				}
			});
		}
	}

	QuadtreeVectorInterpolator interpolator(myQuadtree, myQuadtreeVelocityIndex, myQuadtreeVelocity);

	double maxError = 0;
	
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Ones(), myQuadtree.size(0) - Vec2i::Ones(), [&](const Vec2i& face)
		{
			Vec2d velocityPoint = myQuadtreeVelocityIndex[0].indexToWorld(face.cast<double>(), axis);

			double localError = std::fabs(interpolator.biLerp(velocityPoint, axis) - testFunctor(velocityPoint, axis));

			if (maxError < localError)
			{
				maxError = localError;
			}
		});
	}

	return maxError;
}

int AnalyticalQuadtreeInterpolation::buildQuadtreeVelocityIndex()
{
	// Loop over each level of the grid. Loop over each face.
	// If the face has one ACTIVE adjacent cell and one DOWN adjacent cell OR 
	// has two ACTIVE adjacent cells, then the face is considered active.

	int index = 0;
	
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		Vec2i cellSize = myQuadtree.size(level);

		for (int axis = 0; axis < 2; ++axis)
		{
			forEachVoxelRange(Vec2i::Zero(), myQuadtreeVelocityIndex[level].size(axis), [&](const Vec2i& face)
			{
				// Grab adjacent cells
				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				// Boundary check
				if (backwardCell[axis] < 0)
				{
					if (myQuadtree.isCellActive(forwardCell, level))
					{
						myQuadtreeVelocityIndex[level](face, axis) = index++;
					}
				}
				else if (forwardCell[axis] >= cellSize[axis])
				{
					if (myQuadtree.isCellActive(backwardCell, level))
					{
						myQuadtreeVelocityIndex[level](face, axis) = index++;
					}
				}
				else
				{
					QuadtreeGrid::QuadtreeCellLabel backwardLabel = myQuadtree.getCellLabel(backwardCell, level);
					QuadtreeGrid::QuadtreeCellLabel forwardLabel = myQuadtree.getCellLabel(forwardCell, level);

					if ((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP))
					{
						myQuadtreeVelocityIndex[level](face, axis) = index++;
					}
				}
			});
		}
	}

	// Returning index gives the number of velocity positions required for a linear system
	return index;
}

TEST(QUADTREE_VECTOR_INTERPOLATION_TESTS, CONVERGENCE)
{
	auto velocity = [&](const Vec2d& pos, int axis)
	{
		if (axis == 0)
		{
			return std::sin(pos[0]) * std::sin(pos[1]);
		}
		else
		{
			return std::sin(pos[0]) * std::sin(pos[1]);
		}
	};

	int baseGrid = 64;
	double dx = PI /double(baseGrid);

	Vec2d origin = Vec2d::Zero();
	Vec2i size = Vec2i::Constant(baseGrid);
	Transform xform(dx, origin);

	auto isoSurface = [](const Vec2d& pos) -> double
	{
		return std::min(pos.norm() - .5 * std::sqrt(2.) * PI, (pos - Vec2d::Constant(PI)).norm() - .5 * std::sqrt(2.) * PI);
	};

	auto surfaceRefiner = [&](const Vec2d& pos) -> double
	{
		if (std::fabs(isoSurface(pos)) < dx)
		{
			return 0;
		}

		return -1;
	};

	int levels = 4;

	AnalyticalQuadtreeInterpolation interpolator(xform, size, levels, surfaceRefiner);

	std::vector<double> error(levels, 0);

	for (int i = 0; i < levels; ++i)
	{
		if (i > 0)
		{
			interpolator.refine();
		}

		error[i] = interpolator.solve(velocity);

		std::cout << "Error: " << error[i] << std::endl;
	}


	for (int i = 1; i < levels; ++i)
	{
		EXPECT_GT(error[i - 1] / error[i], 3.);
	}
}