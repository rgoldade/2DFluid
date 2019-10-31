#ifndef LIBRARY_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER
#define LIBRARY_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER

#include "Common.h"
#include "GeometricMultigridOperators.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

template<typename MatrixVectorMultiplyFunctor,
			typename PreconditionerFunctor,
			typename DotProductFunctor,
			typename SquaredL2NormFunctor,
			typename AddToVectorFunctor,
			typename AddScaledVectorFunctor,
			typename StoreReal>
void solveGeometricConjugateGradient(UniformGrid<StoreReal> &solutionGrid,
										const UniformGrid<StoreReal> &rhsGrid,
										const MatrixVectorMultiplyFunctor &matrixVectorMultiplyFunctor,
										const PreconditionerFunctor &preconditionerFunctor,
										const DotProductFunctor &dotProductFunctor,
										const SquaredL2NormFunctor &squaredNormFunctor,
										const AddToVectorFunctor &addToVectorFunctor,
										const AddScaledVectorFunctor &addScaledVectorFunctor,
										const StoreReal tolerance,
										const int maxIterations)
{
	using SolveReal = double;

	assert(solutionGrid.size() == rhsGrid.size());

	SolveReal rhsNorm2 = squaredNormFunctor(rhsGrid);
	if (rhsNorm2 == 0)
	{
		std::cout << "RHS is zero. Nothing to solve" << std::endl;
		return;
	}

	// Build initial residual vector using an initial guess
	UniformGrid<StoreReal> residualGrid(solutionGrid.size(), 0);

	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);

	SolveReal residualNorm2 = squaredNormFunctor(residualGrid);
	SolveReal threshold = SolveReal(tolerance) * SolveReal(tolerance) * rhsNorm2;

	if (residualNorm2 < threshold)
	{
		std::cout << "Residual already below error: " << std::sqrt(residualNorm2 / rhsNorm2) << std::endl;
		return;
	}

	// Apply preconditioner for initial search direction
	UniformGrid<StoreReal> pGrid(solutionGrid.size(), 0);
	
	preconditionerFunctor(pGrid, residualGrid);

	SolveReal absNew;
	absNew = dotProductFunctor(pGrid, residualGrid);

	UniformGrid<StoreReal> zGrid(solutionGrid.size(), 0);
	UniformGrid<StoreReal> tempGrid(solutionGrid.size(), 0);

	int iteration = 0;
	for (; iteration < maxIterations; ++iteration)
	{
		std::cout << "  Iteration: " << iteration << std::endl;

		// Matrix-vector multiplication
		matrixVectorMultiplyFunctor(tempGrid, pGrid);

		SolveReal alpha;
		alpha = absNew / dotProductFunctor(pGrid, tempGrid);

		// Update solution
		addToVectorFunctor(solutionGrid, pGrid, alpha);

		// Update residual
		addToVectorFunctor(residualGrid, tempGrid, -alpha);
		
		residualNorm2 = squaredNormFunctor(residualGrid);

		std::cout << "    Relative error: " << std::sqrt(residualNorm2 / rhsNorm2) << std::endl;

		if (residualNorm2 < threshold)
			break;

		preconditionerFunctor(zGrid, residualGrid);

		SolveReal absOld = absNew;
		SolveReal beta;
		
		absNew = dotProductFunctor(zGrid, residualGrid);
		beta = absNew / absOld;

		addScaledVectorFunctor(pGrid, zGrid, pGrid, beta);
	}

	std::cout << "Iterations: " << iteration << std::endl;
	SolveReal error = std::sqrt(residualNorm2 / rhsNorm2);
	std::cout << "Drifted relative L2 Error: " << error << std::endl;

	// Recompute residual
	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);
	error = std::sqrt(squaredNormFunctor(residualGrid) / rhsNorm2);
	std::cout << "Recomputed relative L2 Error: " << error << std::endl;
}

#endif