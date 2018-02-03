#pragma once

#include "Core.h"
#include "Vec.h"
#include "Eigen/Sparse"

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

class Solver
{
public:
	Solver(size_t rowcount, size_t nonzeros = 0)
	{
		m_A.reserve(nonzeros);
		m_b = Eigen::VectorXd::Zero(rowcount);
		m_x = Eigen::VectorXd::Zero(rowcount);
		m_guess = Eigen::VectorXd::Zero(rowcount);
	}

	// Insert element item into sparse vector. Duplicates are allowed and will be summed together
	inline void add_element(size_t col, size_t row, double val)
	{
		m_A.push_back(Eigen::Triplet<double>(row, col, val));
	}

	// Adds value to RHS. It's safe to assume that it is initialized to zeros
	inline void add_rhs(size_t row, double val)
	{
		m_b(row) += val;
	}
	
	// Caller should make sure solution is up-to-date
	inline double sol(size_t row) const
	{		
		return m_x(row);
	}

	inline void add_guess(size_t row, double val)
	{
		m_guess(row) += val;
	}
	//Call to solve linear system
	bool solve()
	{
		Eigen::SparseMatrix<double> A_sparse(m_b.rows(), m_b.rows());
		A_sparse.setFromTriplets(m_A.begin(), m_A.end());
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
		solver.compute(A_sparse);

		if (solver.info() != Eigen::Success) return false;
		
		m_x = solver.solve(m_b);
		if (solver.info() != Eigen::Success) return false;
		
		return true;
	}

	//Call to solve linear system
	bool solve_iterative()
	{
		Eigen::SparseMatrix<double> A_sparse(m_b.rows(), m_b.rows());
		A_sparse.setFromTriplets(m_A.begin(), m_A.end());
		Eigen::ConjugateGradient <Eigen::SparseMatrix<double>> solver;
		solver.compute(A_sparse);

		if (solver.info() != Eigen::Success) return false;
		solver.setTolerance(1E-5);
		Eigen::setNbThreads(8);
		m_x = solver.solveWithGuess(m_b, m_guess);
		if (solver.info() != Eigen::Success) return false;

		return true;
	}

private:
	Eigen::VectorXd m_b;
	Eigen::VectorXd m_x;
	Eigen::VectorXd m_guess;

	std::vector<Eigen::Triplet<double>> m_A;
};
