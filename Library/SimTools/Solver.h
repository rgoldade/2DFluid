#ifndef LIBRARY_SOLVER_H
#define LIBRARY_SOLVER_H

#include "Eigen/Sparse"

#include "Common.h"

///////////////////////////////////
//
// Solver.h
// Ryan Goldade 2017
//
// Sparse matrix solver with support
// for direct solve and iterative solve.
// The iterative solve can set a "guess"
// starting vector.
//
////////////////////////////////////

template<bool useDoublePrecision = false>
class Solver
{
	using SolverReal = typename std::conditional<useDoublePrecision, double, float>::type;
	using Vector = typename std::conditional<useDoublePrecision, Eigen::VectorXd, Eigen::VectorXf>::type;

public:
	Solver(int rowCount, int estimatedNonZeros = 0)
	{
		assert(rowCount >= 0 && estimatedNonZeros >= 0);
		Eigen::initParallel();

		myMatrix.reserve(estimatedNonZeros);
		myRhs = Vector::Zero(rowCount);
		mySolution = Vector::Zero(rowCount);
		myGuess = Vector::Zero(rowCount);
	}

	// Insert element item into sparse vector. Duplicates are allowed and will be summed together
	void addToElement(int row, int col, SolverReal value)
	{
		assert(row >= 0 && col >= 0);
		myMatrix.push_back(Eigen::Triplet<SolverReal>(row, col, value));
	}

	// Adds value to RHS. It's safe to assume that it is initialized to zeros
	void addToRhs(int row, SolverReal value)
	{
		assert(row >= 0);
		myRhs(row) += value;
	}
	
	SolverReal rhs(int row)
	{
		assert(row >= 0);
		return myRhs(row);
	}

	// Caller should make sure solution is up-to-date
	SolverReal solution(int row) const
	{
		assert(row >= 0);
		return mySolution(row);
	}

	void addToGuess(int row, SolverReal value)
	{
		assert(row >= 0);
		myGuess(row) += value;
	}

	//Call to solve linear system
	bool solveDirect()
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());

		sparseMatrix.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<SolverReal>> solver;

		solver.compute(sparseMatrix);

		if (solver.info() != Eigen::Success) return false;
		
		mySolution = solver.solve(myRhs);
		if (solver.info() != Eigen::Success) return false;
		
		return true;
	}

	//Call to solve linear system
	bool solveIterative(Real tolerance = 1E-5)
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		
		Eigen::ConjugateGradient<Eigen::SparseMatrix<SolverReal>, Eigen::Upper | Eigen::Lower> solver;
		solver.compute(sparseMatrix);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed on build" << std::endl;
			return false;
		}

		solver.setTolerance(tolerance);
		mySolution = solver.solveWithGuess(myRhs, myGuess);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed to converge" << std::endl;
			return false;
		}

		std::cout << "Solver iterations: " << solver.iterations() << std::endl;
		std::cout << "Solve error: " << solver.error() << std::endl;

		return true;
	}

	bool isSymmetric()
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		sparseMatrix.makeCompressed();

		for (int k = 0; k < sparseMatrix.outerSize(); ++k)
			for (typename Eigen::SparseMatrix<SolverReal>::InnerIterator it(sparseMatrix, k); it; ++it)
			{
				if (fabs(sparseMatrix.coeff(it.row(), it.col()) - sparseMatrix.coeff(it.col(), it.row())) > 1E-7)
				{
					std::cout << "Value at row " << it.row() << ", col " << it.col() << " is " << sparseMatrix.coeff(it.row(), it.col()) << std::endl;
					std::cout << "Value at row " << it.col() << ", col " << it.row() << " is " << sparseMatrix.coeff(it.col(), it.row()) << std::endl;
					return false;
				}
			}

		return true;
	}

	bool isFinite()
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		sparseMatrix.makeCompressed();

		for (int k = 0; k < sparseMatrix.outerSize(); ++k)
			for (typename Eigen::SparseMatrix<SolverReal>::InnerIterator it(sparseMatrix, k); it; ++it)
			{
				if (!std::isfinite(it.value()))
					return false;
			}

		return true;
	}

	void printMatrix(std::string filename) const
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		sparseMatrix.makeCompressed();

		std::ofstream writer(filename);

		if (writer)
		{
			for (int k = 0; k < sparseMatrix.outerSize(); ++k)
				for (typename Eigen::SparseMatrix<SolverReal>::InnerIterator it(sparseMatrix, k); it; ++it)
				{
					writer << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << "\n";
				}
		}
		else
			std::cerr << "Failed to write to file: " << filename << std::endl;
	}

private:
	Vector myRhs, mySolution, myGuess;

	std::vector<Eigen::Triplet<SolverReal>> myMatrix;

	std::vector<unsigned> myRemovedDOFs;
};

#endif