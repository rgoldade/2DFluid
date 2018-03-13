#pragma once

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "Transform.h"

#include "Renderer.h"
#include "Solver.h"

#include "Eigen/Sparse"

#include "core.h"

///////////////////////////////////
//
// AnalyticalViscositySolver.h/cpp
// Ryan Goldade 2017
// 
// Solves an analytical viscosity problem
// used to test convergence of the PDE from
// Batty and Bridson 2008. There are only
// grid-aligned solid boundary conditions.
// This is a useful starting point for 
// building tests on non-regular meshes.
//
////////////////////////////////////

class AnalyticalViscositySolver
{
	const int UNASSIGNED = -1;
public:
	AnalyticalViscositySolver(const Transform& xform, const Vec2st& nx)
		: m_xform(xform)
	{
		m_vel_index = VectorGrid<int>(xform, nx, UNASSIGNED, VectorGridSettings::STAGGERED);
	}

	// Returns the infinity-norm error of the numerical solution
	template<typename initial_functor, typename solution_functor, typename viscosity_functor>
	Real solve(initial_functor& initial, solution_functor& solution, viscosity_functor& viscosity,
				Real dt);

	Vec2R cell_idx_to_ws(const Vec2R& ipos) const { return m_xform.idx_to_ws(ipos + Vec2R(.5)); }
	Vec2R node_idx_to_ws(const Vec2R& ipos) const { return m_xform.idx_to_ws(ipos); }
	
	void draw_grid(Renderer& renderer) const;
	void draw_active_velocity(Renderer& renderer) const;

private:

	Transform m_xform;
	size_t set_velocity_index();
	VectorGrid<int> m_vel_index;
};

template<typename initial_functor, typename solution_functor, typename viscosity_functor>
Real AnalyticalViscositySolver::solve(initial_functor& initial, solution_functor& solution, viscosity_functor& viscosity, Real dt)
{
	size_t vcount = set_velocity_index();

	// Build reduced system.
	// (Note we don't need control volumes since the cells are the same size and there is no free surface).
	// (I - dt * mu * D^T K D) u(n+1) = u(n)

	Solver solver(vcount, vcount * 9);

	Real dx = m_vel_index.dx();
	Real base_coeff = dt / sqr(dx);

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel_index.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				int idx = m_vel_index(i, j, axis);

				if (idx >= 0)
				{
					Vec2i face(i, j);

					Vec2R fpos = m_vel_index.idx_to_ws(Vec2R(face), axis);
					solver.add_rhs(idx, initial(fpos, axis));
					solver.add_element(idx, idx, 1.);

					// Build cell-centered stresses.
					for (int c = 0; c < 2; ++c)
					{
						Vec2i cell = face + face_to_cell[axis][c];

						Vec2R pos = cell_idx_to_ws(Vec2R(cell));
						Real coeff = 2. * viscosity(pos) * base_coeff;

						Real csign = (c == 0) ? -1. : 1.;

						for (int f = 0; f < 2; ++f)
						{
							Vec2i adjface = cell + cell_to_face[axis * 2 + f];
							Real fsign = (f == 0) ? -1. : 1.;

							int fidx = m_vel_index(adjface[0], adjface[1], axis);
							if (fidx >= 0)
								solver.add_element(idx, fidx, -csign * fsign * coeff);
							// No solid boundary to deal with since faces on the boundary
							// are not included.
						}
					}

					// Build node stresses.
					for (int n = 0; n < 2; ++n)
					{
						Vec2i node = face + face_to_node[axis][n];

						Real nsign = (n == 0) ? -1. : 1.;

						Vec2R pos = node_idx_to_ws(Vec2R(node));
						Real coeff = viscosity(pos) * base_coeff;

						for (int f = 0; f < 4; ++f)
						{
							Vec4i faceoffset = node_to_face[f];

							int faxis = faceoffset[2];
							int graddir = faceoffset[3];

							Real fsign = (f % 2 == 0) ? -1. : 1.;

							Vec2i adjface = node + Vec2i(faceoffset[0], faceoffset[1]);

							// Check for out of bounds
							if (adjface[graddir] < 0 || adjface[graddir] >= size[graddir])
							{
								Vec2R adjpos = m_vel_index.idx_to_ws(Vec2R(adjface[0], adjface[1]), faxis);
								solver.add_rhs(idx, nsign * fsign * coeff * solution(adjpos, faxis));
							}
							// Check for on the bounds
							else if (adjface[faxis] == 0 || adjface[faxis] == m_vel_index.size(faxis)[faxis] - 1)
							{
								Vec2R adjpos = m_vel_index.idx_to_ws(Vec2R(adjface[0], adjface[1]), faxis);
								solver.add_rhs(idx, nsign * fsign * coeff * solution(adjpos, faxis));
							}
							else
							{
								int fidx = m_vel_index(adjface[0], adjface[1], faxis);
								assert(fidx >= 0);

								solver.add_element(idx, fidx, -nsign * fsign * coeff);
							}
						}
					}
				}
			}
	}

	bool solved = solver.solve();

	Real error = 0;

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel_index.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				int idx = m_vel_index(i, j, axis);

				if (idx >= 0)
				{
					Vec2R pos = m_vel_index.idx_to_ws(Vec2R(i, j), axis);
					Real temperror = fabs(solver.sol(idx) - solution(pos, axis));

					if (error < temperror) error = temperror;
				}
			}
	}

	return error;
}
