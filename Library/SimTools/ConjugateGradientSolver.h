#ifndef CONJUGATE_GRADIENT_SOLVER_H
#define CONJUGATE_GRADIENT_SOLVER_H

#include <iostream>

#include <Eigen/Sparse>

// A custom CG solver derived from Eigen's implementation

namespace FluidSim2D::SimTools
{

template<typename SolveReal,
	typename MatrixVectorMultiplyFunctor,
	typename PreconditionerFunctor,
	typename ResidualPrinterFunctor,
	typename Vector>
	void solveConjugateGradient(Vector& solution,
		const Vector& rhs,
		const MatrixVectorMultiplyFunctor& matrixVectorMultiplyFunctor,
		const PreconditionerFunctor& preconditionerFunctor,
		const ResidualPrinterFunctor& residualPrinterFunctor,
		const SolveReal tolerance,
		const int maxIterations,
		const bool doProjectNullSpace = false)
{
	SolveReal rhsNorm2 = rhs.squaredNorm();
	if (rhsNorm2 == 0)
	{
		solution.setZero();
		std::cout << "RHS is zero. Nothing to solve" << std::endl;
		return;
	}

	Vector residual = rhs - matrixVectorMultiplyFunctor(solution); //initial residual

	if (doProjectNullSpace)
		residual.array() -= residual.sum() / SolveReal(residual.rows());

	SolveReal threshold = tolerance * tolerance * rhsNorm2;
	SolveReal residualNorm2 = residual.squaredNorm();

	if (residualNorm2 < threshold)
	{
		std::cout << "Residual already below error: " << std::sqrt(residualNorm2 / rhsNorm2) << std::endl;
		return;
	}

	Vector z(rhs.rows()), tmp(rhs.rows());

	//
	// Build initial search direction from preconditioner
	//

	Vector p = preconditionerFunctor(residual);

	if (doProjectNullSpace)
		p.array() -= p.sum() / SolveReal(p.rows());

	SolveReal absNew = residual.dot(p);

	int iteration = 0;

	while (iteration < maxIterations)
	{
		tmp.noalias() = matrixVectorMultiplyFunctor(p);

		SolveReal alpha = absNew / p.dot(tmp);
		solution += alpha * p;
		residual -= alpha * tmp;

		if (doProjectNullSpace)
			residual.array() -= residual.sum() / SolveReal(residual.rows());

		residualPrinterFunctor(residual, iteration);

		residualNorm2 = residual.squaredNorm();
		if (residualNorm2 < threshold)
			break;
		else std::cout << "    Residual: " << std::sqrt(residualNorm2 / rhs.squaredNorm()) << std::endl;

		// Start with the diagonal preconditioner
		z.noalias() = preconditionerFunctor(residual);

		SolveReal absOld = absNew;
		absNew = residual.dot(z);     // update the absolute value of r
		SolveReal beta = absNew / absOld;	  // calculate the Gram-Schmidt value used to create the new search direction
		p = z + beta * p;			   // update search direction

		if (doProjectNullSpace)
			p.array() -= p.sum() / SolveReal(p.rows());

		++iteration;
	}

	std::cout << "Iterations: " << iteration << std::endl;
	std::cout << "Relative L2 Error: " << std::sqrt(residualNorm2 / rhs.squaredNorm()) << std::endl;

	residual = rhs - matrixVectorMultiplyFunctor(solution);
	residualNorm2 = residual.squaredNorm();
	std::cout << "Re-computed Relative L2 Error: " << std::sqrt(residualNorm2 / rhs.squaredNorm()) << std::endl;

	std::cout << "L-infinity Error: " << residual.lpNorm<Eigen::Infinity>() << std::endl;
}

}

#endif