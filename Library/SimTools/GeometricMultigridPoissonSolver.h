#ifndef GEOMETRIC_MULTIGRID_POISSONSOLVER_H
#define GEOMETRIC_MULTIGRID_POISSONSOLVER_H

#include "Eigen/Sparse"

#include "GeometricMultigridOperators.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim2D::SimTools
{

using namespace Utilities;

class GeometricMultigridPoissonSolver
{
	using StoreReal = double;
	using SolveReal = double;
	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	using MGCellLabels = GeometricMultigridOperators::CellLabels;

public:
	GeometricMultigridPoissonSolver() : myMGLevels(0) {};
	GeometricMultigridPoissonSolver(const UniformGrid<MGCellLabels>& initialDomainLabels,
									const VectorGrid<StoreReal>& boundaryWeights,
									int mgLevels,
									SolveReal dx);

	void applyMGVCycle(UniformGrid<StoreReal>& solutionVector,
						const UniformGrid<StoreReal>& rhsVector,
						bool useInitialGuess = false);

private:

	std::vector<UniformGrid<MGCellLabels>> myDomainLabels;
	std::vector<UniformGrid<StoreReal>> mySolutionGrids, myRHSGrids, myResidualGrids;

	std::vector<std::vector<Vec2i>> myBoundaryCells;

	int myMGLevels;

	VectorGrid<StoreReal> myFineBoundaryWeights;

	std::vector<SolveReal> myDx;

	int myBoundarySmootherIterations;
	int myBoundarySmootherWidth;

	UniformGrid<int> myDirectSolverIndices;

	Eigen::SparseMatrix<SolveReal> mySparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> myCoarseSolver;
};

}

#endif