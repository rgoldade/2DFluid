#ifndef LIBRARY_GEOMETRIC_CG_POISSON_SOLVER
#define LIBRARY_GEOMETRIC_CG_POISSON_SOLVER

#include "Common.h"
#include "GeometricMGOperations.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

using namespace GeometricMGOperations;

// TODO: allow for mixed precision

template<typename MatrixVectorMultiplyFunctor,
			typename PreconditionerFunctor,
			typename DotProductFunctor,
			typename L2NormFunctor,
			typename AddScaledVectorFunctor>
void solveGeometricCGPoisson(UniformGrid<Real> &solutionGrid,
								const UniformGrid<Real> &rhsGrid,
								MatrixVectorMultiplyFunctor &matrixVectorMultiplyFunctor,
								PreconditionerFunctor &preconditionerFunctor,
								DotProductFunctor &dotProductFunctor,
								L2NormFunctor &L2NormFunctor,
								AddScaledVectorFunctor &addScaledVectorFunctor,
								const Real threshold,
								bool doCheckEnergy = false,
								bool useRelativeError = true,
								bool printResidual = true)
{
	// Compute norm
	Real rhsNorm = L2NormFunctor(rhsGrid);
	if (rhsNorm == 0) return;

	Vec2i size = solutionGrid.size();

	// Compute residual with r = b - Ax
	UniformGrid<Real> residualGrid(size, 0);

	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);

	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);

	Real l2Error = L2NormFunctor(residualGrid);
	
	if (useRelativeError)
	{
		if ((l2Error / rhsNorm) < threshold) return;
	}
	else if (l2Error < threshold) return;

	// Apply preconditioner for initial search direction
	UniformGrid<Real> pGrid(size, 0);

	preconditionerFunctor(pGrid, residualGrid);

	Real rho = dotProductFunctor(pGrid, residualGrid);

	UniformGrid<Real> zGrid(size, 0);

	int maxIterations = 1000;

	Real oldEnergy = std::numeric_limits<Real>::max();

	for (int iteration = 0; iteration < maxIterations; ++iteration)
	{
		// Matrix-vector multiplication
		matrixVectorMultiplyFunctor(zGrid, pGrid);

		// Sigma
		Real sigma = dotProductFunctor(pGrid, zGrid);

		Real alpha = rho / sigma;

		// Update residual
		addScaledVectorFunctor(residualGrid, residualGrid, zGrid, -alpha);

		// Compuate norm
		l2Error = L2NormFunctor(residualGrid);

		// Check convergence
		if (useRelativeError)
		{
			if ((l2Error / rhsNorm) < threshold)
			{
				std::cout << "Geometric CG iteration: " << iteration << ". Relative L-2: " << l2Error / rhsNorm << std::endl;
				addScaledVectorFunctor(solutionGrid, solutionGrid, pGrid, alpha);
				return;
			}
		}
		else
		{
			if (l2Error < threshold)
			{
				std::cout << "Geometric CG iteration: " << iteration << ". Relative L-2: " << l2Error / rhsNorm << std::endl;
				addScaledVectorFunctor(solutionGrid, solutionGrid, pGrid, alpha);
				return;
			}
		}

		// Make sure energy is decreasing
		if (doCheckEnergy)
		{
			UniformGrid<Real> tempGrid(size, 0);
			matrixVectorMultiplyFunctor(tempGrid, solutionGrid);

			Real energy = .5 * dotProductFunctor(tempGrid, solutionGrid);

			energy -= dotProductFunctor(rhsGrid, solutionGrid);

			std::cout << "Geometric CG iteration: " << iteration
				<< ". Energy: " << energy
				<< ". L-2 norm: " << l2Error
				<< ". Relative L-2: " << l2Error / rhsNorm << std::endl;

			assert(energy < oldEnergy);
			if (energy >= oldEnergy)
				std::cout << "!!! ENERGY NOT MINIMIZED !!!" << std::endl;

			oldEnergy = energy;

			if (printResidual)
				residualGrid.printAsOBJ("residualGrid" + std::to_string(iteration));
		}

		preconditionerFunctor(zGrid, residualGrid);

		// Update rho
		double rhoNew = dotProductFunctor(zGrid, residualGrid);

		// Update beta
		double beta = rhoNew / rho;

		// Store rho
		rho = rhoNew;

		// Update vectors
		addScaledVectorFunctor(solutionGrid, solutionGrid, pGrid, alpha);
		addScaledVectorFunctor(pGrid, zGrid, pGrid, beta);
	}
}

#endif