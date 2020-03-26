#ifndef TESTS_ANALYTICAL_POISSON_H
#define TESTS_ANALYTICAL_POISSON_H

#include <Eigen/Sparse>

#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::Utilities;

class AnalyticalPoissonSolver
{
	using SolveReal = double;
	using Vector = Eigen::VectorXd;

public:
	AnalyticalPoissonSolver(const Transform& xform, const Vec2i& size)
		: myXform(xform)
	{
		myPoissonGrid = ScalarGrid<float>(myXform, size, 0);
	}

	template<typename RHS, typename Solution>
	float solve(const RHS& rhsFunction, const Solution& solutionFunction);

	void drawGrid(Renderer& renderer) const;
	void drawValues(Renderer& renderer) const;

private:
	
	Transform myXform;
	ScalarGrid<float> myPoissonGrid;
};

template<typename RHS, typename Solution>
float AnalyticalPoissonSolver::solve(const RHS& rhsFuction, const Solution& solutionFunction)
{
	UniformGrid<int> solvableCells(myPoissonGrid.size(), -1);

	int solutionDOFCount = 0;

	Vec2i gridSize = myPoissonGrid.size();

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		solvableCells(cell) = solutionDOFCount++;
	});

	std::vector<Eigen::Triplet<SolveReal>> sparseMatrixElements;

	Vector rhsVector = Vector::Zero(solutionDOFCount);

	float dx = myPoissonGrid.dx();
	float coeff = sqr(dx);

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		int row = solvableCells(cell);

		assert(row >= 0);

		// Build RHS
		Vec2f gridPoint = myPoissonGrid.indexToWorld(Vec2f(cell));

		rhsVector(row) = coeff * rhsFuction(gridPoint);

		for (auto axis : { 0, 1 })
			for (auto direction : { 0,1 })
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);

				// Bounds check. Use analytical solution for Dirichlet condition.
				if ((direction == 0 && adjacentCell[axis] < 0) ||
					(direction == 1 && adjacentCell[axis] >= gridSize[axis]))
				{
					Vec2f adjacentPoint = myPoissonGrid.indexToWorld(Vec2f(adjacentCell));
					rhsVector(row) -= solutionFunction(adjacentPoint);
				}
				else
				{
					// If neighbouring cell is solvable, it should have an entry in the system
					int adjacentRow = solvableCells(adjacentCell);
					assert(adjacentRow >= 0);

					sparseMatrixElements.emplace_back(row, adjacentRow, 1);
				}
			}
		sparseMatrixElements.emplace_back(row, row, -4);
	});

	Eigen::SparseMatrix<SolveReal> sparseMatrix(solutionDOFCount, solutionDOFCount);
	sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<SolveReal>> solver;
	solver.compute(sparseMatrix);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return - 1;
	}

	Vector solutionVector = solver.solve(rhsVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to converge" << std::endl;
		return -1;
	}

	float error = 0;

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		int row = solvableCells(cell);

		assert(row >= 0);

		Vec2f gridPoint = myPoissonGrid.indexToWorld(Vec2f(cell));
		float localError = fabs(solutionVector(row) - solutionFunction(gridPoint));

		if (error < localError) error = localError;

		myPoissonGrid(cell) = solutionVector(row);
	});

	return error;
}

#endif