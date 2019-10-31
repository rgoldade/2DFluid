#ifndef GEOMETRIC_MULTIGRID_POISSONSOLVER_H
#define GEOMETRIC_MULTIGRID_POISSONSOLVER_H

#include "Eigen/Sparse"

#include "Common.h"
#include "GeometricMultigridOperators.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

class GeometricMultigridPoissonSolver
{
	static constexpr int UNLABELLED_CELL = -1;

	using StoreReal = double;
	using SolveReal = double;
	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	using CellLabels = GeometricMultigridOperators::CellLabels;

public:
	GeometricMultigridPoissonSolver(const UniformGrid<CellLabels> &initialDomainLabels,
									const VectorGrid<StoreReal> &boundaryWeights,
									const int mgLevels,
									const SolveReal dx);
	
	void applyMGVCycle(UniformGrid<StoreReal> &solutionVector,
						const UniformGrid<StoreReal> &rhsVector,
						const bool useInitialGuess = false);

private:

	std::vector<UniformGrid<CellLabels>> myDomainLabels;
	std::vector<UniformGrid<StoreReal>> mySolutionGrids, myRHSGrids, myResidualGrids;

	std::vector<std::vector<Vec2i>> myBoundaryCells;

	int myMGLevels;

	VectorGrid<StoreReal> myFineBoundaryWeights;

	std::vector<SolveReal> myDx;

	const int myBoundarySmootherIterations;
	const int myBoundarySmootherWidth;

	UniformGrid<int> myDirectSolverIndices;

	Eigen::SparseMatrix<SolveReal> mySparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> myCoarseSolver;
};

#endif