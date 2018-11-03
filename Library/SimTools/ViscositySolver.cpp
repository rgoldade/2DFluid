#include <iostream>

#include "ViscositySolver.h"
#include "VectorGrid.h"
#include "Solver.h"

void ViscositySolver::solve(const VectorGrid<Real>& face_volumes,
							ScalarGrid<Real>& center_volumes,
							ScalarGrid<Real>& node_volumes,
							const ScalarGrid<Real>& collision_center_volumes,
							const ScalarGrid<Real>& collision_node_volumes)
{
	// Debug check that grids are the same
	assert(m_vel.is_matched(face_volumes));
	assert(m_surface.is_matched(center_volumes));
	assert(m_surface.is_matched(collision_center_volumes));
	assert(m_surface.size() + Vec2ui(1) == collision_node_volumes.size());
	assert(m_surface.size() + Vec2ui(1) == node_volumes.size());

	VectorGrid<int> solvable_faces(m_surface.xform(), m_surface.size(), UNSOLVED, VectorGridSettings::SampleType::STAGGERED);

	// Build solvable faces. Assumes the grid limits are solid boundaries and left out
	// of the system.

	unsigned solve_count = 0;
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui vel_size = solvable_faces.size(axis);

		Vec2ui start(0); ++start[axis];
		Vec2ui end(vel_size); --end[axis];

		for_each_voxel_range(start, end, [&](const Vec2ui& face)
		{
			bool insolve = false;

			for (unsigned dir = 0; dir < 2; ++dir)
			{
				Vec2i cell = Vec2i(face) + face_to_cell[axis][dir];
				if (center_volumes(Vec2ui(cell)) > 0.) insolve = true;
			}

			if (!insolve)
			{
				for (unsigned dir = 0; dir < 2; ++dir)
				{
					Vec2ui node = face + face_to_node[axis][dir];

					if (node_volumes(node) > 0.) insolve = true;
				}
			}

			if (insolve)
			{
				if (m_collision.interp(solvable_faces.idx_to_ws(Vec2R(face), axis)) <= 0.)
					solvable_faces(face, axis) = COLLISION;
				else
					solvable_faces(face, axis) = solve_count++;
			}
		});
	}

	// Build a single container of the viscosity weights (liquid volumes, gf weights, viscosity coefficients)
	Real invdx2 = 1. / sqr(m_surface.dx());
	{
		for_each_voxel_range(Vec2ui(0), center_volumes.size(), [&](const Vec2ui& cell)
		{
			center_volumes(cell) *= m_dt * invdx2;
			center_volumes(cell) *= m_viscosity(cell);
			center_volumes(cell) *= clamp(collision_center_volumes(cell), 0.01, 1.);
		});
	}

	{
		for_each_voxel_range(Vec2ui(0), node_volumes.size(), [&](const Vec2ui& node)
		{
			node_volumes(node) *= m_dt * invdx2;
			node_volumes(node) *= m_viscosity.interp(node_volumes.idx_to_ws(Vec2R(node)));
			node_volumes(node) *= clamp(collision_node_volumes(node), 0.01, 1.);
		});
	}

	Solver<true> solver(solve_count, solve_count * 9);

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int idx = solvable_faces(face, axis);
			if (idx >= 0)
			{
				Real vol = face_volumes(face, axis);

				// Build RHS with weight velocities
				solver.add_rhs(idx, m_vel(face, axis) * vol);
				solver.add_guess(idx, m_vel(face, axis));

				// Add control volume weight on the diagonal.
				solver.add_element(idx, idx, vol);

				// Build cell-centered stresses.
				for (unsigned dir = 0; dir < 2; ++dir)
				{
					Vec2i cell = Vec2i(face) + face_to_cell[axis][dir];

					Real coeff = 2. * center_volumes(Vec2ui(cell));

					Real csign = (dir == 0) ? -1. : 1.;

					for (unsigned face_dir = 0; face_dir < 2; ++face_dir)
					{
						// Since we've assumed grid boundaries are static solids
						// we don't have any solveable faces with adjacent cells
						// out of the grid bounds. We can skip that check here.

						Vec2ui adjface = Vec2ui(cell) + cell_to_face[axis * 2 + face_dir];

						Real fsign = (face_dir == 0) ? -1. : 1.;

						int fidx = solvable_faces(adjface, axis);
						if (fidx >= 0)
							solver.add_element(idx, fidx, -csign * fsign * coeff);
						else if (fidx == COLLISION)
							solver.add_rhs(idx, csign * fsign * coeff * m_collision_vel(adjface, axis));
					}
				}

				// Build node stresses.
				for (unsigned dir = 0; dir < 2; ++dir)
				{
					Vec2ui node = face + face_to_node[axis][dir];

					Real nsign = (dir == 0) ? -1. : 1.;

					Real coeff = node_volumes(node);

					for (unsigned face_dir = 0; face_dir < 4; ++face_dir)
					{
						Vec4i faceoffset = node_to_face[face_dir];

						unsigned faxis = faceoffset[2];
						unsigned graddir = faceoffset[3];

						Real fsign = (face_dir % 2 == 0) ? -1. : 1.;

						Vec2i adjface = Vec2i(node) + Vec2i(faceoffset[0], faceoffset[1]);

						// Check for in bounds
						if (adjface[graddir] >= 0 && adjface[graddir] < size[graddir])
						{
							int fidx = solvable_faces(Vec2ui(adjface), faxis);

							if (fidx >= 0)
								solver.add_element(idx, fidx, -nsign * fsign * coeff);
							else if (fidx == COLLISION)
								solver.add_rhs(idx, nsign * fsign * coeff * m_collision_vel(Vec2ui(adjface), faxis));
						}
					}
				}
			}
		});
	}

	bool solved = solver.solve_iterative();

	if (!solved)
	{
		std::cout << "Viscosity failed to solve" << std::endl;
		assert(false);
	}

	// Update velocity
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int idx = solvable_faces(face, axis);
			if (idx >= 0)
				m_vel(face, axis) = solver.sol(idx);
			else if (idx == COLLISION)
				m_vel(face, axis) = m_collision_vel(face, axis);
		});
	}
}

void ViscositySolver::set_viscosity(Real mu)
{
	m_viscosity = ScalarGrid<Real>(m_surface.xform(), m_surface.size(), mu);
}

void ViscositySolver::set_viscosity(const ScalarGrid<Real>& mu)
{
	assert(mu.size() == m_surface.size());
	m_viscosity = mu;
}