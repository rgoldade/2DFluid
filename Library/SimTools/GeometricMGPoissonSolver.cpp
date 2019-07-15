#include "GeometricMGPoissonSolver.h"

#include "tbb/tbb.h"

#include "GeometricMGOperations.h"

using namespace tbb;

void GeometricMGPoissonSolver::applyMGVCycle(UniformGrid<Real> &solutionVector,
												const UniformGrid<Real> &rhsVector,
												const bool useInitialGuess)
{
	assert(rhsVector.size()[0] < mySolutionGrids[0].size()[0] &&
			rhsVector.size()[1] < mySolutionGrids[0].size()[1]);

	assert(solutionVector.size() == rhsVector.size());

	int totalFineVoxels = solutionVector.size()[0] * solutionVector.size()[1];
	// Clear out solution
	mySolutionGrids[0].reset(0);
	myRHSGrids[0].reset(0);

	// If there is an initial guess in the solution vector, copy it locally
	if (useInitialGuess)
	{
		parallel_for(blocked_range<int>(0, totalFineVoxels, 1000), [&](const blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = solutionVector.unflatten(flatIndex);

				Vec2i expandedCell = cell + myExteriorOffset;

				if (myDomainLabels[0](expandedCell) == CellLabels::INTERIOR)
					mySolutionGrids[0](expandedCell) = solutionVector(cell);
			}
		});
	}

	// Copy rhs vector to local rhs according to the exterior band padding
	parallel_for(blocked_range<int>(0, totalFineVoxels, 1000), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = solutionVector.unflatten(flatIndex);

			Vec2i expandedCell = cell + myExteriorOffset;

			if (myDomainLabels[0](expandedCell) == CellLabels::INTERIOR)
				myRHSGrids[0](expandedCell) = rhsVector(cell);
		}
	});

	int boundaryIterations = 2;
	int totalSmootherIterations = 1;

	// Down-stroke of the v-cycle
	for (int level = 0; level < myMGLevels - 1; ++level)
	{
		for (int smootherIterations = 0; smootherIterations < totalSmootherIterations; ++smootherIterations)
		{
			if (level == 0 && myDoApplyGradientWeights)
			{
				dampedJacobiWeightedPoissonSmoother(mySolutionGrids[level],
													myRHSGrids[level],
													myDomainLabels[level],
													myFineGradientWeights,
													myDx[level]);

				for (int iteration = 0; iteration < 2 * (level + 1) * boundaryIterations; ++iteration)
				{
					dampedJacobiWeightedPoissonSmoother(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myFineGradientWeights,
														myDx[level]);
				}
			}
			else
			{
				dampedJacobiPoissonSmoother(mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myDx[level]);

				for (int iteration = 0; iteration < 2 * (level + 1) * boundaryIterations; ++iteration)
				{
					dampedJacobiPoissonSmoother(mySolutionGrids[level],
												myRHSGrids[level],
												myDomainLabels[level],
												myBoundaryCells[level],
												myDx[level]);
				}
			}
		}

		// Compute residual to restrict to the next level
		if (level == 0 && myDoApplyGradientWeights)
		{
			computeWeightedPoissonResidual(myResidualGrids[level],
											mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myFineGradientWeights,
											myDx[level]);
		}
		else
		{
			computePoissonResidual(myResidualGrids[level],
									mySolutionGrids[level],
									myRHSGrids[level],
									myDomainLabels[level],
									myDx[level]);
		}

		downsample(myRHSGrids[level + 1],
					myResidualGrids[level],
					myDomainLabels[level + 1],
					myDomainLabels[level]);

		mySolutionGrids[level + 1].reset(0);
	}

	// Copy to Eigen and direct solve
	forEachVoxelRange(Vec2i(0), myDomainLabels[myMGLevels - 1].size(), [&](const Vec2i &cell)
	{
		if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR)
		{
			int index = myDirectSolverIndices(cell);
			assert(index >= 0);

			myCoarseRHSVector(index) = myRHSGrids[myMGLevels - 1](cell);
		}
	});

	Eigen::VectorXd directSolution = myCoarseSolver.solve(myCoarseRHSVector);

	// Copy solution back
	forEachVoxelRange(Vec2i(0), myDomainLabels[myMGLevels - 1].size(), [&](const Vec2i &cell)
	{
		if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR)
		{
			int index = myDirectSolverIndices(cell);
			assert(index >= 0);

			mySolutionGrids[myMGLevels - 1](cell) = directSolution(index);
		}
	});

	// Up-stroke of the v-cycle
	for (int level = myMGLevels - 2; level >= 0; --level)
	{
		upsampleAndAdd(mySolutionGrids[level],
						mySolutionGrids[level + 1],
						myDomainLabels[level],
						myDomainLabels[level + 1]);

		for (int smootherIterations = 0; smootherIterations < totalSmootherIterations; ++smootherIterations)
		{
			if (level == 0 && myDoApplyGradientWeights)
			{
				for (int iteration = 0; iteration < 2 * (level + 1) * boundaryIterations; ++iteration)
				{
					dampedJacobiWeightedPoissonSmoother(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myFineGradientWeights,
														myDx[level]);
				}

				dampedJacobiWeightedPoissonSmoother(mySolutionGrids[level],
					myRHSGrids[level],
					myDomainLabels[level],
					myFineGradientWeights,
					myDx[level]);
			}
			else
			{
				for (int iteration = 0; iteration < 2 * (level + 1) * boundaryIterations; ++iteration)
				{
					dampedJacobiPoissonSmoother(mySolutionGrids[level],
												myRHSGrids[level],
												myDomainLabels[level],
												myBoundaryCells[level],
												myDx[level]);
				}

				dampedJacobiPoissonSmoother(mySolutionGrids[level],
					myRHSGrids[level],
					myDomainLabels[level],
					myDx[level]);
			}
		}
	}

	// Copy local solution vector with expanded exterior band to 
	// an interior solution that matches the supplied RHS vector grid.
	int totalVoxels = solutionVector.size()[0] * solutionVector.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, 1000), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = solutionVector.unflatten(flatIndex);

			Vec2i expandedCell = cell + myExteriorOffset;

			if (myDomainLabels[0](expandedCell) == CellLabels::INTERIOR)
				solutionVector(cell) = mySolutionGrids[0](expandedCell);
		}
	});
}