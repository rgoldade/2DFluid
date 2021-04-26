#ifndef FLUIDSIM2D_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H
#define FLUIDSIM2D_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H

#include "GeometricMultigridOperators.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim2D
{

template<typename MatrixVectorMultiplyFunctor,
			typename PreconditionerFunctor,
			typename DotProductFunctor,
			typename SquaredL2NormFunctor,
			typename AddToVectorFunctor,
			typename AddScaledVectorFunctor>
void solveGeometricConjugateGradient(UniformGrid<double>& solutionGrid,
										const UniformGrid<double>& rhsGrid,
										const MatrixVectorMultiplyFunctor& matrixVectorMultiplyFunctor,
										const PreconditionerFunctor& preconditionerFunctor,
										const DotProductFunctor& dotProductFunctor,
										const SquaredL2NormFunctor& squaredNormFunctor,
										const AddToVectorFunctor& addToVectorFunctor,
										const AddScaledVectorFunctor& addScaledVectorFunctor,
										const double tolerance,
										const int maxIterations)
{
	assert(solutionGrid.size() == rhsGrid.size());

	double rhsNorm2 = squaredNormFunctor(rhsGrid);
	if (rhsNorm2 == 0)
	{
		std::cout << "RHS is zero. Nothing to solve" << std::endl;
		return;
	}

	// Build initial residual vector using an initial guess
	UniformGrid<double> residualGrid(solutionGrid.size(), 0);

	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);

	double residualNorm2 = squaredNormFunctor(residualGrid);
	double threshold = double(tolerance) * double(tolerance) * rhsNorm2;

	if (residualNorm2 < threshold)
	{
		std::cout << "Residual already below error: " << std::sqrt(residualNorm2 / rhsNorm2) << std::endl;
		return;
	}

	// Apply preconditioner for initial search direction
	UniformGrid<double> pGrid(solutionGrid.size(), 0);

	preconditionerFunctor(pGrid, residualGrid);

	double absNew;
	absNew = dotProductFunctor(pGrid, residualGrid);

	UniformGrid<double> zGrid(solutionGrid.size(), 0);
	UniformGrid<double> tempGrid(solutionGrid.size(), 0);

	int iteration = 0;
	for (; iteration < maxIterations; ++iteration)
	{
		std::cout << "  Iteration: " << iteration << std::endl;

		// Matrix-vector multiplication
		matrixVectorMultiplyFunctor(tempGrid, pGrid);

		double alpha;
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

		double absOld = absNew;
		double beta;

		absNew = dotProductFunctor(zGrid, residualGrid);
		beta = absNew / absOld;

		addScaledVectorFunctor(pGrid, zGrid, pGrid, beta);
	}

	std::cout << "Iterations: " << iteration << std::endl;
	double error = std::sqrt(residualNorm2 / rhsNorm2);
	std::cout << "Drifted relative L2 Error: " << error << std::endl;

	// Recompute residual
	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);
	error = std::sqrt(squaredNormFunctor(residualGrid) / rhsNorm2);
	std::cout << "Recomputed relative L2 Error: " << error << std::endl;
}

}
#endif