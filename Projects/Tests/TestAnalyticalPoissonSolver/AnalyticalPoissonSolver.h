#pragma once

#include "Common.h"

#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "LevelSet2D.h"
#include "Transform.h"

#include "Integrator.h"
#include "Solver.h"
#include "Renderer.h"

class AnalyticalPoissonSolver
{
public:
	AnalyticalPoissonSolver(const Transform& xform, const Vec2ui& size)
		: m_xform(xform)
	{
		m_poissongrid = ScalarGrid<Real>(m_xform, size, 0);
	}

	template<typename RHS, typename Solution>
	Real solve(const RHS& rhs, const Solution& solution);

	void draw_grid(Renderer& renderer) const;
	void draw_values(Renderer& renderer) const;

private:
	
	Transform m_xform;
	ScalarGrid<Real> m_poissongrid;
};

template<typename RHS, typename Solution>
Real AnalyticalPoissonSolver::solve(const RHS& rhs, const Solution& solution)
{
	UniformGrid<int> solvable_cells(m_poissongrid.size(), -1);

	unsigned solvecount = 0;

	Vec2ui size = solvable_cells.size();

	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		solvable_cells(cell) = solvecount++;
	});

	Solver<true> solver(solvecount, solvecount * 5);

	Real dx = m_poissongrid.dx();
	Real coeff = sqr(dx);

	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		int idx = solvable_cells(cell);

		assert(idx >= 0);

		// Build RHS
		Vec2R pos = m_poissongrid.idx_to_ws(Vec2R(cell));

		solver.add_rhs(idx, rhs(pos) * coeff);

		for (unsigned dir = 0; dir < 4; ++dir)
		{
			Vec2i ncell = Vec2i(cell) + cell_to_cell[dir];

			// Bounds check. Use analytical solution for Dirichlet condition.
			if (ncell[0] < 0 || ncell[1] < 0 ||
				ncell[0] >= size[0] || ncell[1] >= size[1])
			{
				Vec2R npos = m_poissongrid.idx_to_ws(Vec2R(ncell));

				solver.add_rhs(idx, -solution(npos));
			}
			else
			{
				// If neighbouring cell is solvable, it should have an entry in the system
				int cidx = solvable_cells(Vec2ui(ncell));
				assert(cidx >= 0);

				solver.add_element(idx, cidx, 1.);
			}
		}

		solver.add_element(idx, idx, -4.);
	});

	bool solved = solver.solve();

	if (!solved)
	{
		std::cout << "Analytical poisson test failed to solve" << std::endl;
		return -1;
	}

	Real error = 0;

	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		int idx = solvable_cells(cell);

		assert(idx >= 0);

		Vec2R pos = m_poissongrid.idx_to_ws(Vec2R(cell));
		Real temperror = fabs(solver.sol(idx) - solution(pos));

		if (error < temperror) error = temperror;

		m_poissongrid(cell) = solver.sol(idx);
	});

	return error;
}