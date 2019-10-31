#include "GeometricMultigridPoissonSolver.h"

#include "tbb/tbb.h"

#include "GeometricMultigridOperators.h"

using GeometricMultigridOperators::CellLabels;

template<typename Vector, typename StoreReal>
void copyGridToVector(Vector &vector,
						const UniformGrid<StoreReal> &vectorGrid,
						const UniformGrid<int> &solverIndices,
						const UniformGrid<CellLabels> &cellLabels)
{
	using GeometricMultigridOperators::CellLabels;

	assert(vectorGrid.size() == solverIndices.size() &&
		solverIndices.size() == cellLabels.size());

	int totalVoxels = cellLabels.size()[0] * cellLabels.size()[1];

	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = cellLabels.unflatten(flatIndex);

			int index = solverIndices(cell);

			if (index >= 0)
			{
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL);

				vector(index) = vectorGrid(cell);
			}
			else
			{
				assert(cellLabels(cell) != CellLabels::INTERIOR_CELL &&
						cellLabels(cell) != CellLabels::BOUNDARY_CELL);
			}
		}
	});
}

template<typename Vector, typename StoreReal>
void copyVectorToGrid(UniformGrid<StoreReal> &vectorGrid,
						const Vector &vector,
						const UniformGrid<int> &solverIndices,
						const UniformGrid<CellLabels> &cellLabels)
{
	using GeometricMultigridOperators::CellLabels;

	assert(vectorGrid.size() == solverIndices.size() &&
			solverIndices.size() == cellLabels.size());

	int totalVoxels = cellLabels.size()[0] * cellLabels.size()[1];

	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = cellLabels.unflatten(flatIndex);

			int index = solverIndices(cell);

			if (index >= 0)
			{
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL);

				vectorGrid(cell) = vector(index);
			}
			else
			{
				assert(cellLabels(cell) != CellLabels::INTERIOR_CELL &&
					cellLabels(cell) != CellLabels::BOUNDARY_CELL);
			}
		}
	});
}

GeometricMultigridPoissonSolver::GeometricMultigridPoissonSolver(const UniformGrid<CellLabels> &initialDomainLabels,
																	const VectorGrid<StoreReal> &boundaryWeights,
																	const int mgLevels,
																	const SolveReal dx)
	: myMGLevels(mgLevels)
	, myBoundarySmootherWidth(3)
	, myBoundarySmootherIterations(3)
{
	assert(mgLevels > 0 && dx > 0);

	assert(initialDomainLabels.size()[0] % 2 == 0 &&
			initialDomainLabels.size()[1] % 2 == 0);

	assert(int(std::log2(initialDomainLabels.size()[0])) + 1 >= mgLevels &&
			int(std::log2(initialDomainLabels.size()[1])) + 1 >= mgLevels);

	myDomainLabels.resize(myMGLevels);
	myDomainLabels[0] = initialDomainLabels;

	assert(boundaryWeights.size(0)[0] - 1 == myDomainLabels[0].size()[0] &&
			boundaryWeights.size(0)[1] == myDomainLabels[0].size()[1] &&
		
			boundaryWeights.size(1)[0] == myDomainLabels[0].size()[0] &&
			boundaryWeights.size(1)[1] - 1 == myDomainLabels[0].size()[1]);

	myFineBoundaryWeights = boundaryWeights;

	auto checkSolvableCell = [](const UniformGrid<CellLabels> &testGrid) -> bool
	{
		bool hasSolvableCell = false;

		int totalVoxels = testGrid.size()[0] * testGrid.size()[1];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			if (hasSolvableCell) return;
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = testGrid.unflatten(flatIndex);

				if (testGrid(cell) == CellLabels::INTERIOR_CELL ||
					testGrid(cell) == CellLabels::BOUNDARY_CELL)
					hasSolvableCell = true;
			}
		});

		return hasSolvableCell;
	};

	assert(checkSolvableCell(myDomainLabels[0]));
	assert(GeometricMultigridOperators::unitTestBoundaryCells<StoreReal>(myDomainLabels[0], &myFineBoundaryWeights));
	assert(GeometricMultigridOperators::unitTestExteriorCells(myDomainLabels[0]));

	// Precompute the coarsening strategy. Cap level if there are no longer interior cells
	for (int level = 1; level < myMGLevels; ++level)
	{
		myDomainLabels[level] = buildCoarseCellLabels(myDomainLabels[level - 1]);

		if (!checkSolvableCell(myDomainLabels[level]))
		{
			myMGLevels = level - 1;
			myDomainLabels.resize(myMGLevels);
			break;
		}
		assert(GeometricMultigridOperators::unitTestCoarsening(myDomainLabels[level], myDomainLabels[level - 1]));
		assert(GeometricMultigridOperators::unitTestBoundaryCells<StoreReal>(myDomainLabels[level]));
		assert(GeometricMultigridOperators::unitTestExteriorCells(myDomainLabels[0]));
	}

	myDx.resize(myMGLevels);
	myDx[0] = dx;

	for (int level = 1; level < myMGLevels; ++level)
		myDx[level] = 2. * myDx[level - 1];

	// Initialize solution vectors
	mySolutionGrids.resize(myMGLevels);
	myRHSGrids.resize(myMGLevels);
	myResidualGrids.resize(myMGLevels);

	for (int level = 0; level < myMGLevels; ++level)
	{
		mySolutionGrids[level] = UniformGrid<StoreReal>(myDomainLabels[level].size());
		myRHSGrids[level] = UniformGrid<StoreReal>(myDomainLabels[level].size());
		myResidualGrids[level] = UniformGrid<StoreReal>(myDomainLabels[level].size());
	}

	myBoundaryCells.resize(myMGLevels);
	for (int level = 0; level < myMGLevels; ++level)
		myBoundaryCells[level] = buildBoundaryCells(myDomainLabels[level], myBoundarySmootherWidth);

	// Pre-build matrix at the coarsest level
	{
		int interiorCellCount = 0;
		Vec2i coarsestSize = myDomainLabels[myMGLevels - 1].size();

		myDirectSolverIndices = UniformGrid<int>(coarsestSize, UNLABELLED_CELL);

		forEachVoxelRange(Vec2i(0), coarsestSize, [&](const Vec2i &cell)
		{
			if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR_CELL ||
				myDomainLabels[myMGLevels - 1](cell) == CellLabels::BOUNDARY_CELL)
				myDirectSolverIndices(cell) = interiorCellCount++;
		});

		// Build rows
		std::vector<Eigen::Triplet<SolveReal>> sparseElements;

		SolveReal gridScale = 1. / Util::sqr(myDx[myMGLevels - 1]);
		forEachVoxelRange(Vec2i(0), coarsestSize, [&](const Vec2i &cell)
		{
			if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR_CELL ||
				myDomainLabels[myMGLevels - 1](cell) == CellLabels::BOUNDARY_CELL)
			{
				int diagonal = 0;
				int index = myDirectSolverIndices(cell);
				assert(index >= 0);
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						auto cellLabels = myDomainLabels[myMGLevels - 1](adjacentCell);
						if (cellLabels == CellLabels::INTERIOR_CELL ||
							cellLabels == CellLabels::BOUNDARY_CELL)
						{
							int adjacentIndex = myDirectSolverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							sparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -gridScale));
							++diagonal;
						}
						else if (cellLabels == CellLabels::DIRICHLET_CELL)
							++diagonal;
					}

				sparseElements.push_back(Eigen::Triplet<double>(index, index, gridScale * diagonal));
			}
		});

		// Solve system
		mySparseMatrix = Eigen::SparseMatrix<SolveReal>(interiorCellCount, interiorCellCount);
		mySparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
		mySparseMatrix.makeCompressed();

		myCoarseSolver.compute(mySparseMatrix);
		assert(myCoarseSolver.info() == Eigen::Success);
	}
}

void GeometricMultigridPoissonSolver::applyMGVCycle(UniformGrid<StoreReal> &fineSolutionGrid,
													const UniformGrid<StoreReal> &fineRHSGrid,
													const bool useInitialGuess)
{
	using namespace GeometricMultigridOperators;

	assert(fineSolutionGrid.size() == fineRHSGrid.size() &&
			fineRHSGrid.size() == myDomainLabels[0].size());

	// If there is an initial guess in the solution vector, copy it locally
	if (!useInitialGuess)
		fineSolutionGrid.reset(0);

	// Apply fine-level smoothing pass
	{
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
														fineRHSGrid,
														myDomainLabels[0],
														myBoundaryCells[0],
														myDx[0],
														&myFineBoundaryWeights);
		}

		// Interior smoothing
		interiorJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
													fineRHSGrid,
													myDomainLabels[0],
													myDx[0],
													&myFineBoundaryWeights);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
														fineRHSGrid,
														myDomainLabels[0],
														myBoundaryCells[0],
														myDx[0],
														&myFineBoundaryWeights);
		}

		myResidualGrids[0].reset(0);

		computePoissonResidual<SolveReal>(myResidualGrids[0],
											fineSolutionGrid,
											fineRHSGrid,
											myDomainLabels[0],
											myDx[0],
											&myFineBoundaryWeights);

		myRHSGrids[1].reset(0);

		downsample<SolveReal>(myRHSGrids[1],
								myResidualGrids[0],
								myDomainLabels[1],
								myDomainLabels[0]);
	}

	// Down-stroke of the v-cycle
	for (int level = 1; level < myMGLevels - 1; ++level)
	{
		mySolutionGrids[level].reset(0);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myDx[level]);
		}

		// Interior smoothing
		interiorJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
													myRHSGrids[level],
													myDomainLabels[level],
													myDx[level]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myDx[level]);
		}

		myResidualGrids[level].reset(0);

		// Compute residual to restrict to the next level
		computePoissonResidual<SolveReal>(myResidualGrids[level],
											mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myDx[level]);

		myRHSGrids[level + 1].reset(0);

		downsample<SolveReal>(myRHSGrids[level + 1],
								myResidualGrids[level],
								myDomainLabels[level + 1],
								myDomainLabels[level]);
	}

	Vector coarseRHSVector = Vector::Zero(mySparseMatrix.rows());
	copyGridToVector(coarseRHSVector,
						myRHSGrids[myMGLevels - 1],
						myDirectSolverIndices,
						myDomainLabels[myMGLevels - 1]);

	Vector directSolution = myCoarseSolver.solve(coarseRHSVector);

	mySolutionGrids[myMGLevels - 1].reset(0);
	copyVectorToGrid(mySolutionGrids[myMGLevels - 1],
						directSolution,
						myDirectSolverIndices,
						myDomainLabels[myMGLevels - 1]);

	// Up-stroke of the v-cycle
	for (int level = myMGLevels - 2; level >= 1; --level)
	{
		upsampleAndAdd<SolveReal>(mySolutionGrids[level],
									mySolutionGrids[level + 1],
									myDomainLabels[level],
									myDomainLabels[level + 1]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myDx[level]);
		}

		interiorJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
													myRHSGrids[level],
													myDomainLabels[level],
													myDx[level]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
														myRHSGrids[level],
														myDomainLabels[level],
														myBoundaryCells[level],
														myDx[level]);
		}
	}

	// Apply fine-level smoother
	{
		upsampleAndAdd<SolveReal>(fineSolutionGrid,
									mySolutionGrids[1],
									myDomainLabels[0],
									myDomainLabels[1]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
														fineRHSGrid,
														myDomainLabels[0],
														myBoundaryCells[0],
														myDx[0],
														&myFineBoundaryWeights);
		}

		interiorJacobiPoissonSmoother(fineSolutionGrid,
										fineRHSGrid,
										myDomainLabels[0],
										myDx[0],
										&myFineBoundaryWeights);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
				fineRHSGrid,
				myDomainLabels[0],
				myBoundaryCells[0],
				myDx[0],
				&myFineBoundaryWeights);
		}
	}
}