#include "ViscositySolver.h"
#include "VectorGrid.h"
#include "Solver.h"

static int UNSOLVED = -2;
static int COLLISION = -1;
static int FLUID = 0;

void ViscositySolver::solve(const VectorGrid<Real>& face_volumes, ScalarGrid<Real> center_volumes, ScalarGrid<Real> node_volumes)
{
	// Debug check that grids are the same
	assert(m_vel.size(0) == face_volumes.size(0) && m_vel.size(1) == face_volumes.size(1));
	assert(m_surface.size() == center_volumes.size());
	assert(m_surface.size() + Vec2st(1) == node_volumes.size());

	VectorGrid<int> solvable_faces(m_surface.xform(), m_surface.size(), UNSOLVED, VectorGridSettings::STAGGERED);

	// Build solvable faces. Assumes the grid limits are solid boundaries and left out
	// of the system.
	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = solvable_faces.size(axis);

		Vec2i start(0); ++start[axis];
		Vec2i end(size); --end[axis];

		for (int i = start[0]; i < end[0]; ++i)
			for (int j = start[1]; j < end[1]; ++j)
			{
				bool insolve = false;

				Vec2i face(i, j);

				for (int c = 0; c < 2; ++c)
				{
					Vec2i cell = face + face_to_cell[axis][c];
					if (center_volumes(cell[0], cell[1]) > 0.) insolve = true;
				}

				if (!insolve)
				{
					for (int n = 0; n < 2; ++n)
					{
						Vec2i node = face + face_to_node[axis][n];

						if (node_volumes(node[0], node[1]) > 0.) insolve = true;
					}
				}

				if (insolve)
				{ 
					if (m_collision.interp(solvable_faces.idx_to_ws(Vec2R(i, j), axis)) <= 0.)
						solvable_faces(i, j, axis) = COLLISION;
					else
						solvable_faces(i, j, axis) = FLUID;
				}
			}
	}

	int solvecount = 0;

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = solvable_faces.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				if (solvable_faces(i, j, axis) == FLUID) solvable_faces(i, j, axis) = solvecount++;
			}
	}
	
	// Build a single container of the viscosity weights (liquid volumes, gf weights, viscosity coefficients)
	Real invdx = 1. / sqr(m_surface.dx());
	{
		Vec2st size = center_volumes.size();

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				center_volumes(i, j) *= m_dt * invdx;
				center_volumes(i, j) *= m_viscosity(i, j);

				if (m_colweight_set)
				{
					Real weight = clamp((*m_col_center_vol)(i, j), 0.01, 1.);
					center_volumes(i, j) *= weight;
				}
			}
	}

	{
		Vec2st size = node_volumes.size();
		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				node_volumes(i, j) *= m_dt * invdx;
				node_volumes(i, j) *= m_viscosity.interp(node_volumes.idx_to_ws(Vec2R(i, j)));
				
				if (m_colweight_set)
				{
					Real weight = clamp((*m_col_node_vol)(i, j), 0.01, 1.);
					node_volumes(i, j) *= weight;
				}
			}
	}

	Solver solver(solvecount, solvecount * 9);

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				int idx = solvable_faces(i, j, axis);

				if (idx >= 0)
				{
					Vec2i face(i, j);
			
					Real vol = face_volumes(i, j, axis);
					
					// Build RHS with weight velocities
					solver.add_rhs(idx, m_vel(i, j, axis) * vol);
					solver.add_guess(idx, m_vel(i, j, axis));

					// Add control volume weight on the diagonal.
					solver.add_element(idx, idx, vol);

					// Build cell-centered stresses.
					for (int c = 0; c < 2; ++c)
					{
						Vec2i cell = face + face_to_cell[axis][c];

						Real coeff = 2. * center_volumes(cell[0], cell[1]);

						Real csign = (c == 0) ? -1. : 1.;

						for (int f = 0; f < 2; ++f)
						{
							// Since we've assumed grid boundaries are static solids
							// we don't have any solveable faces with adjacent cells
							// out of the grid bounds. We can skip that check here.

							Vec2i adjface = cell + cell_to_face[axis * 2 + f];

							Real fsign = (f == 0) ? -1. : 1.;

							int fidx = solvable_faces(adjface[0], adjface[1], axis);
							if (fidx >= 0)
								solver.add_element(idx, fidx, -csign * fsign * coeff);
							else if (fidx == COLLISION && m_colvel_set)
								solver.add_rhs(idx, csign * fsign * coeff * m_collision_vel(adjface[0], adjface[1], axis));
						}
					}

					// Build node stresses.
					for (int n = 0; n < 2; ++n)
					{
						Vec2i node = face + face_to_node[axis][n];

						Real nsign = (n == 0) ? -1. : 1.;
						
						Real coeff = node_volumes(node[0], node[1]);

						for (int f = 0; f < 4; ++f)
						{
							Vec4i faceoffset = node_to_face[f];

							int faxis = faceoffset[2];
							int graddir = faceoffset[3];

							Real fsign = (f % 2 == 0) ? -1. : 1.;

							Vec2i adjface = node + Vec2i(faceoffset[0], faceoffset[1]);

							// Check for in bounds
							if (adjface[graddir] >= 0 && adjface[graddir] < size[graddir])
							{
								int fidx = solvable_faces(adjface[0], adjface[1], faxis);
								
								if (fidx >= 0)
									solver.add_element(idx, fidx, -nsign * fsign * coeff);
								else if (fidx == COLLISION && m_colvel_set)
									solver.add_rhs(idx, nsign * fsign * coeff * m_collision_vel(adjface[0], adjface[1], faxis));
							}
						}
					}
				}
			}
	}

	std::cout << "Solving for viscosity" << std::endl;

	bool solved = solver.solve_iterative();

	if (!solved)
	{
		std::cout << "Viscosity failed" << std::endl;
		assert(false);
	}

	// Update velocity
	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel.size(axis);
		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				int idx = solvable_faces(i, j, axis);
				if (idx >= 0)
					m_vel(i, j, axis) = solver.sol(idx);
				else if (idx == COLLISION)
				{
					if (m_colvel_set)
						m_vel(i, j, axis) = m_collision_vel(i, j, axis);
					else
						m_vel(i, j, axis) = 0.;
				}
			}
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