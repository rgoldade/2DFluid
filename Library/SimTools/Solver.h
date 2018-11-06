#pragma once

#include "Common.h"

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

template<bool use_double_precision = false>
class Solver
{
	using SolverReal = typename std::conditional<use_double_precision, double, float>::type;
	using Vector = typename std::conditional<use_double_precision, Eigen::VectorXd, Eigen::VectorXf>::type;
public:
	Solver(unsigned rowcount, unsigned nonzeros = 0)
	{
		m_A.reserve(nonzeros);
		m_b = Vector::Zero(rowcount);
		m_x = Vector::Zero(rowcount);
		m_guess = Vector::Zero(rowcount);
	}

	// Insert element item into sparse vector. Duplicates are allowed and will be summed together
	void add_element(unsigned row, unsigned col, SolverReal val)
	{
		m_A.push_back(Eigen::Triplet<SolverReal>(row, col, val));
	}

	// Adds value to RHS. It's safe to assume that it is initialized to zeros
	void add_rhs(unsigned row, SolverReal val)
	{
		m_b(row) += val;
	}
	
	// Caller should make sure solution is up-to-date
	SolverReal sol(unsigned row) const
	{		
		return m_x(row);
	}

	inline void add_guess(unsigned row, SolverReal val)
	{
		m_guess(row) += val;
	}
	//Call to solve linear system
	bool solve()
	{
		Eigen::SparseMatrix<SolverReal> A_sparse(m_b.rows(), m_b.rows());
		A_sparse.setFromTriplets(m_A.begin(), m_A.end());
		A_sparse.makeCompressed();
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolverReal>> solver;

		solver.analyzePattern(A_sparse);
		solver.factorize(A_sparse);
		solver.compute(A_sparse);

		if (solver.info() != Eigen::Success) return false;
		
		m_x = solver.solve(m_b);
		if (solver.info() != Eigen::Success) return false;
		
		return true;
	}

	//Call to solve linear system
	bool solve_iterative(Real tolerance = 1E-5)
	{
		Eigen::SparseMatrix<SolverReal> A_sparse(m_b.rows(), m_b.rows());
		A_sparse.setFromTriplets(m_A.begin(), m_A.end());
		Eigen::ConjugateGradient <Eigen::SparseMatrix<SolverReal>, Eigen::Lower | Eigen::Upper> solver;
		solver.compute(A_sparse);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed on build" << std::endl;
			return false;
		}

		solver.setTolerance(tolerance);
		m_x = solver.solveWithGuess(m_b, m_guess);

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
		Eigen::SparseMatrix<SolverReal> A_sparse(m_b.rows(), m_b.rows());
		A_sparse.setFromTriplets(m_A.begin(), m_A.end());
		A_sparse.makeCompressed();
		Eigen::SparseLU<Eigen::SparseMatrix<SolverReal>> solver;

		solver.analyzePattern(A_sparse);
		solver.factorize(A_sparse);
		solver.compute(A_sparse);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed on build" << std::endl;
			return false;
		}

		m_x = solver.solve(m_b);

		if (solver.info() != Eigen::Success)
		{
			std::cout << "Solve failed" << std::endl;
			return false;
		}

		return true;
	}

private:
	Vector m_b, m_x, m_guess;

	std::vector<Eigen::Triplet<SolverReal>> m_A;
};
