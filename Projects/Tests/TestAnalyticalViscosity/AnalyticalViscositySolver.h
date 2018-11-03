#pragma once

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "Transform.h"

#include "Renderer.h"
#include "Solver.h"

#include "Eigen/Sparse"

#include "Common.h"

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
	static constexpr int UNASSIGNED = -1;

public:
	AnalyticalViscositySolver(const Transform& xform, const Vec2ui& size)
		: m_xform(xform)
	{
		m_vel_index = VectorGrid<int>(xform, size, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
	}

	// Returns the infinity-norm error of the numerical solution
	template<typename Initial, typename Solution, typename Viscosity>
	Real solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity,
				const Real dt);

	Vec2R cell_idx_to_ws(const Vec2R& index_pos) const { return m_xform.idx_to_ws(index_pos + Vec2R(.5)); }
	Vec2R node_idx_to_ws(const Vec2R& index_pos) const { return m_xform.idx_to_ws(index_pos); }
	
	void draw_grid(Renderer& renderer) const;
	void draw_active_velocity(Renderer& renderer) const;

private:

	Transform m_xform;
	unsigned set_velocity_index();
	VectorGrid<int> m_vel_index;
};

template<typename Initial, typename Solution, typename Viscosity>
Real AnalyticalViscositySolver::solve(const Initial& initial,
										const Solution& solution,
										const Viscosity& viscosity,
										const Real dt)
{
	unsigned vcount = set_velocity_index();

	// Build reduced system.
	// (Note we don't need control volumes since the cells are the same size and there is no free surface).
	// (I - dt * mu * D^T K D) u(n+1) = u(n)

	Solver<false> solver(vcount, vcount * 9);

	Real dx = m_vel_index.dx();
	Real base_coeff = dt / sqr(dx);

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel_index.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int idx = m_vel_index(face, axis);

			if (idx >= 0)
			{
				Vec2R fpos = m_vel_index.idx_to_ws(Vec2R(face), axis);
				solver.add_rhs(idx, initial(fpos, axis));
				solver.add_element(idx, idx, 1.);

				// Build cell-centered stresses.
				for (unsigned c = 0; c < 2; ++c)
				{
					Vec2i cell = Vec2i(face) + face_to_cell[axis][c];

					Vec2R pos = cell_idx_to_ws(Vec2R(cell));
					Real coeff = 2. * viscosity(pos) * base_coeff;

					Real sign = (c == 0) ? -1. : 1.;

					for (int f = 0; f < 2; ++f)
					{
						Vec2ui adjface = Vec2ui(cell) + cell_to_face[axis * 2 + f];
						Real fsign = (f == 0) ? -1. : 1.;

						int fidx = m_vel_index(adjface[0], adjface[1], axis);
						if (fidx >= 0)
							solver.add_element(idx, fidx, -sign * fsign * coeff);
						// No solid boundary to deal with since faces on the boundary
						// are not included.
					}
				}

				// Build node stresses.
				for (unsigned n = 0; n < 2; ++n)
				{
					Vec2ui node = face + face_to_node[axis][n];

					Real node_sign = (n == 0) ? -1. : 1.;

					Vec2R pos = node_idx_to_ws(Vec2R(node));
					Real coeff = viscosity(pos) * base_coeff;

					for (int f = 0; f < 4; ++f)
					{
						Vec4i faceoffset = node_to_face[f];

						int faxis = faceoffset[2];
						int graddir = faceoffset[3];

						Real face_sign = (f % 2 == 0) ? -1. : 1.;

						Vec2i adjface = Vec2i(node) + Vec2i(faceoffset[0], faceoffset[1]);

						// Check for out of bounds
						if (adjface[graddir] < 0 || adjface[graddir] >= size[graddir])
						{
							Vec2R adjpos = m_vel_index.idx_to_ws(Vec2R(adjface[0], adjface[1]), faxis);
							solver.add_rhs(idx, node_sign * face_sign * coeff * solution(adjpos, faxis));
						}
						// Check for on the bounds
						else if (adjface[faxis] == 0 || adjface[faxis] == m_vel_index.size(faxis)[faxis] - 1)
						{
							Vec2R adjpos = m_vel_index.idx_to_ws(Vec2R(adjface[0], adjface[1]), faxis);
							solver.add_rhs(idx, node_sign * face_sign * coeff * solution(adjpos, faxis));
						}
						else
						{
							int fidx = m_vel_index(adjface[0], adjface[1], faxis);
							assert(fidx >= 0);

							solver.add_element(idx, fidx, -node_sign * face_sign * coeff);
						}
					}
				}
			}
		});
	}

	bool solved = solver.solve();

	Real error = 0;

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel_index.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int idx = m_vel_index(face, axis);

			if (idx >= 0)
			{
				Vec2R pos = m_vel_index.idx_to_ws(Vec2R(face), axis);
				Real temperror = fabs(solver.sol(idx) - solution(pos, axis));

				if (error < temperror) error = temperror;
			}
		});
	}

	return error;
}
