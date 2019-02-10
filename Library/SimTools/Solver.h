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
	Solver(unsigned rowcount, unsigned nonzeros = 0)
	{
		myMatrix.reserve(nonzeros);
		myRhs = Vector::Zero(rowcount);
		mySolution = Vector::Zero(rowcount);
		myGuess = Vector::Zero(rowcount);
	}

	// Insert element item into sparse vector. Duplicates are allowed and will be summed together
	void addElement(unsigned row, unsigned col, SolverReal val)
	{
		myMatrix.push_back(Eigen::Triplet<SolverReal>(row, col, val));
	}

	// Adds value to RHS. It's safe to assume that it is initialized to zeros
	void addRhs(unsigned row, SolverReal val)
	{
		myRhs(row) += val;
	}
	
	SolverReal rhs(unsigned row)
	{
		return myRhs(row);
	}

	void removeDOF(unsigned element)
	{
		myRemovedDOFs.push_back(element);
	}

	// Caller should make sure solution is up-to-date
	SolverReal solution(unsigned row) const
	{		
		return mySolution(row);
	}

	inline void addGuess(unsigned row, SolverReal val)
	{
		myGuess(row) += val;
	}

	//Call to solve linear system
	bool solve()
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		sparseMatrix.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<SolverReal>> solver;

		/*solver.analyzePattern(sparseMatrix);
		solver.factorize(sparseMatrix);*/
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

		for (auto removeElement : myRemovedDOFs)
		{
			for (int k = 0; k < sparseMatrix.outerSize(); ++k)
				for (typename Eigen::SparseMatrix<SolverReal>::InnerIterator it(sparseMatrix, k); it; ++it)
				{
					if (it.row() == removeElement && it.col() == removeElement)
						sparseMatrix.coeffRef(it.row(), it.col()) = 1.;

					if (it.row() == removeElement && it.col() != removeElement)
						sparseMatrix.coeffRef(it.row(), it.col()) = 0.;

					if (it.row() != removeElement && it.col() == removeElement)
						sparseMatrix.coeffRef(it.row(), it.col()) = 0.;
				}

			myRhs[removeElement] = 0.;
		}

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

		return true;
	}

	//Call to solve linear system
	bool solve_nonsymmetric()
	{
		Eigen::SparseMatrix<SolverReal> sparseMatrix(myRhs.rows(), myRhs.rows());
		sparseMatrix.setFromTriplets(myMatrix.begin(), myMatrix.end());
		sparseMatrix.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<SolverReal>> solver;

		solver.analyzePattern(sparseMatrix);
		solver.factorize(sparseMatrix);
		solver.compute(sparseMatrix);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed on build" << std::endl;
			return false;
		}

		mySolution = solver.solve(myRhs);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed" << std::endl;
			return false;
		}

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
				if (!Util::isEqual(sparseMatrix.coeff(it.row(), it.col()), sparseMatrix.coeff(it.col(), it.row())))
					return false;
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

private:
	Vector myRhs, mySolution, myGuess;

	std::vector<Eigen::Triplet<SolverReal>> myMatrix;

	std::vector<unsigned> myRemovedDOFs;
};

#endif