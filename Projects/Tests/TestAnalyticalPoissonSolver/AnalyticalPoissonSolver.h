#ifndef TESTS_ANALYTICALPOISSON_H
#define TESTS_ANALYTICALPOISSON_H

#include "Common.h"
#include "Integrator.h"
#include "LevelSet2D.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Solver.h"
#include "Transform.h"
#include "VectorGrid.h"

class AnalyticalPoissonSolver
{
public:
	AnalyticalPoissonSolver(const Transform& xform, const Vec2ui& size)
		: myXform(xform)
	{
		myPoissonGrid = ScalarGrid<Real>(myXform, size, 0);
	}

	template<typename RHS, typename Solution>
	Real solve(const RHS& rhsFunction, const Solution& solutionFunction);

	void drawGrid(Renderer& renderer) const;
	void drawValues(Renderer& renderer) const;

private:
	
	Transform myXform;
	ScalarGrid<Real> myPoissonGrid;
};

template<typename RHS, typename Solution>
Real AnalyticalPoissonSolver::solve(const RHS& rhsFuction, const Solution& solutionFunction)
{
	UniformGrid<int> solvableCells(myPoissonGrid.size(), -1);

	unsigned solutionDOFCount = 0;

	Vec2ui gridSize = myPoissonGrid.size();

	forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
	{
		solvableCells(cell) = solutionDOFCount++;
	});

	Solver<true> solver(solutionDOFCount, solutionDOFCount * 5);

	Real dx = myPoissonGrid.dx();
	Real coeff = Util::sqr(dx);

	forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
	{
		int row = solvableCells(cell);

		assert(row >= 0);

		// Build RHS
		Vec2R gridPoint = myPoissonGrid.indexToWorld(Vec2R(cell));

		solver.addRhs(row, coeff * rhsFuction(gridPoint));

		for (auto axis : { 0, 1 })
			for (auto direction : { 0,1 })
			{
				Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

				// Bounds check. Use analytical solution for Dirichlet condition.
				if ((direction == 0 && adjacentCell[axis] < 0) ||
					(direction == 1 && adjacentCell[axis] >= gridSize[axis]))
				{
					Vec2R adjacentPoint = myPoissonGrid.indexToWorld(Vec2R(adjacentCell));
					solver.addRhs(row, -solutionFunction(adjacentPoint));
				}
				else
				{
					// If neighbouring cell is solvable, it should have an entry in the system
					int adjacentRow = solvableCells(Vec2ui(adjacentCell));
					assert(adjacentRow >= 0);

					solver.addElement(row, adjacentRow, 1.);
				}
			}

		solver.addElement(row, row, -4.);
	});

	bool solved = solver.solve();

	if (!solved)
	{
		std::cout << "Analytical poisson test failed to solve" << std::endl;
		return -1;
	}

	Real error = 0;

	forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
	{
		int row = solvableCells(cell);

		assert(row >= 0);

		Vec2R gridPoint = myPoissonGrid.indexToWorld(Vec2R(cell));
		Real localError = fabs(solver.solution(row) - solutionFunction(gridPoint));

		if (error < localError) error = localError;

		myPoissonGrid(cell) = solver.solution(row);
	});

	return error;
}

#endif