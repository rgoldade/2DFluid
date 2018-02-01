#include "ViscositySolver.h"
#include "VectorGrid.h"
#include "Solver.h"

static int UNSOLVED = -2;
static int COLLISION = -1;
static int FLUID = 0;

void ViscositySolver::solve(const VectorGrid<Real>& face_volumes, ScalarGrid<Real> center_volumes, ScalarGrid<Real> node_volumes, Renderer& renderer)
{
	//m_vel.draw_sample_point_vectors(renderer, Vec3f(0,0,1), m_dt * m_surface.dx());

	VectorGrid<int> solvable_faces(m_surface.xform(), m_surface.size(), UNSOLVED, VectorGridSettings::STAGGERED);

	// Build solvable faces
	for (size_t x = 1; x < solvable_faces.size(0)[0] - 1; ++x)
		for (size_t y = 1; y < solvable_faces.size(0)[1] - 1; ++y)
		{
			if (center_volumes(x - 1, y) > 0. ||
				center_volumes(x, y) > 0. ||
				node_volumes(x, y) > 0. ||
				node_volumes(x, y + 1) > 0.)
			{
				if (m_collision.interp(solvable_faces.idx_to_ws(Vec2R(x, y), 0)) <= 0.)
					solvable_faces(x, y, 0) = COLLISION;
				else
					solvable_faces(x, y, 0) = FLUID;
			}
		}

	// Ideally the collision solves this problem -1 index problem
	for (size_t x = 1; x < solvable_faces.size(1)[0] - 1; ++x)
		for (size_t y = 1; y < solvable_faces.size(1)[1] - 1; ++y)
		{
			if (center_volumes(x, y - 1) > 0. ||
				center_volumes(x, y) > 0. ||
				node_volumes(x, y) > 0. ||
				node_volumes(x + 1, y) > 0.)
			{
				if (m_collision.interp(solvable_faces.idx_to_ws(Vec2R(x, y), 1)) <= 0.)
					solvable_faces(x, y, 1) = COLLISION;
				else
					solvable_faces(x, y, 1) = FLUID;
			}
		}

	int solvecount = 0;

	for (size_t x = 0; x < solvable_faces.size(0)[0]; ++x)
		for (size_t y = 0; y < solvable_faces.size(0)[1]; ++y)
		{
			if (solvable_faces(x, y, 0) == FLUID) solvable_faces(x, y, 0) = solvecount++;
		}

	for (size_t x = 0; x < solvable_faces.size(1)[0]; ++x)
		for (size_t y = 0; y < solvable_faces.size(1)[1]; ++y)
		{
			if (solvable_faces(x, y, 1) == FLUID) solvable_faces(x, y, 1) = solvecount++;
		}
	
	// Build a single container of the viscosity weights (liquid volumes, gf weights, viscosity coefficients)
	Real dx = m_surface.dx();
	for(size_t x = 0; x < center_volumes.size()[0]; ++x)
		for (size_t y = 0; y < center_volumes.size()[1]; ++y)
		{
			center_volumes(x, y) *= m_dt / dx / dx;
			center_volumes(x, y) *= m_viscosity(x, y);
			if (m_colweight_set)
			{
				Real weight = clamp((*m_col_center_vol)(x, y), 0.01, 1.);
				center_volumes(x, y) *= weight;
			}			
		}

	for (size_t x = 0; x < node_volumes.size()[0]; ++x)
		for (size_t y = 0; y < node_volumes.size()[1]; ++y)
		{
			node_volumes(x, y) *= m_dt / dx / dx;
			node_volumes(x, y) *= m_viscosity.interp(node_volumes.idx_to_ws(Vec2R(x, y + 1)));
			if (m_colweight_set)
			{
				Real weight = clamp((*m_col_node_vol)(x, y), 0.01, 1.);
				node_volumes(x, y) *= weight;
			}
		}

	Solver solver(solvecount, solvecount * 9);

	size_t x_size = m_vel.size(0)[0];
	size_t y_size = m_vel.size(0)[1];

	// For simplicity, we will assume that fluid never reaches the volume limits
	for (size_t x = 1; x < x_size - 1; ++x)
		for (size_t y = 1; y < y_size - 1; ++y)
		{
			int idx = solvable_faces(x, y, 0);
			if (idx >= 0)
			{
				solver.add_guess(idx, m_vel(x, y, 0));

				Real u_vol = face_volumes(x, y, 0);

				solver.add_rhs(idx, m_vel(x, y, 0) * u_vol);

				Real diag_coeff = u_vol;
				
				// TODO: do volume-based ghost fluid weights for solid boundaries

				// Left cell-centered term
				{
					Real left_coeff = 2. * center_volumes(x - 1, y);
					
					diag_coeff += left_coeff;

					int lidx = solvable_faces(x - 1, y, 0);
					if (lidx >= 0)
						solver.add_element(idx, lidx, -left_coeff);
					else if (m_colvel_set && lidx == COLLISION)
						solver.add_rhs(idx, left_coeff * m_collision_vel(x - 1, y, 0));
				}
				// Right cell-centered term
				{
					Real right_coeff = 2. * center_volumes(x, y);

					diag_coeff += right_coeff;

					int ridx = solvable_faces(x + 1, y, 0);
					if (ridx >= 0)
						solver.add_element(idx, ridx, -right_coeff);
					else if (m_colvel_set && ridx == COLLISION)
						solver.add_rhs(idx, right_coeff * m_collision_vel(x + 1, y, 0));
				}
				
				// Top node
				{
					Real top_coeff = node_volumes(x, y + 1);

					diag_coeff += top_coeff;

					// u_yy term
					int tidx = solvable_faces(x, y + 1, 0);
					if (tidx >= 0)
						solver.add_element(idx, tidx, -top_coeff);
					else if (m_colvel_set && tidx == COLLISION)
						solver.add_rhs(idx, top_coeff * m_collision_vel(x, y + 1, 0));

					// v_xy terms
					int tylidx = solvable_faces(x - 1, y + 1, 1);
					if (tylidx >= 0)
						solver.add_element(idx, tylidx, top_coeff);
					else if(m_colvel_set && tylidx == COLLISION)
						solver.add_rhs(idx,  -top_coeff * m_collision_vel(x - 1, y + 1, 1));

					int tyridx = solvable_faces(x, y + 1, 1);
					if (tyridx >= 0)
						solver.add_element(idx, tyridx, -top_coeff);
					else if (m_colvel_set && tyridx == COLLISION)
						solver.add_rhs(idx, top_coeff * m_collision_vel(x, y + 1, 1));	
				}

				// Bottom
				{
					Real bottom_coeff = node_volumes(x, y);
					
					diag_coeff += bottom_coeff;

					// u_yy term
					int bidx = solvable_faces(x, y - 1, 0);
					if (bidx >= 0)
						solver.add_element(idx, bidx, -bottom_coeff);
					else if (m_colvel_set && bidx == COLLISION)
						solver.add_rhs(idx, bottom_coeff * m_collision_vel(x, y - 1, 0));

					// v_xy terms
					int bylidx = solvable_faces(x - 1, y, 1);
					if (bylidx >= 0)
						solver.add_element(idx, bylidx, -bottom_coeff);
					else if (m_colvel_set && bylidx == COLLISION)
						solver.add_rhs(idx, bottom_coeff * m_collision_vel(x - 1, y, 1));

					int byridx = solvable_faces(x, y, 1);
					if (byridx >= 0)
						solver.add_element(idx, byridx, bottom_coeff);
					else if (m_colvel_set && byridx == COLLISION)
						solver.add_rhs(idx, -bottom_coeff * m_collision_vel(x, y, 1));

				}
		
				solver.add_element(idx, idx, diag_coeff);

			}
		}
		
	x_size = m_vel.size(1)[0];
	y_size = m_vel.size(1)[1];
	for (size_t x = 1; x < x_size - 1; ++x)
		for (size_t y = 1; y < y_size - 1; ++y)
		{
			int idx = solvable_faces(x, y, 1);
			if (idx >= 0)
			{
				solver.add_guess(idx, m_vel(x, y, 1));

				Real v_vol = face_volumes(x, y, 1);

				solver.add_rhs(idx, m_vel(x, y, 1) * v_vol);

				Real diag_coeff = v_vol;

				// TODO: do volume-based ghost fluid weights for solid boundaries

				// Bottom cell-centered term
				{
					Real bottom_coeff = 2. * center_volumes(x, y - 1);

					diag_coeff += bottom_coeff;

					int bidx = solvable_faces(x, y - 1, 1);
					if (bidx >= 0)
						solver.add_element(idx, bidx, -bottom_coeff);
					else if (m_colvel_set && bidx == COLLISION)
						solver.add_rhs(idx, bottom_coeff * m_collision_vel(x, y - 1, 1));
				}
				// Top cell-centered term
				{
					Real top_coeff = 2. * center_volumes(x, y);

					diag_coeff += top_coeff;

					int tidx = solvable_faces(x, y + 1, 1);
					if (tidx >= 0)
						solver.add_element(idx, tidx, -top_coeff);
					else if (m_colvel_set && tidx == COLLISION)
						solver.add_rhs(idx, top_coeff * m_collision_vel(x, y + 1, 1));
				}
				// Right node
				{
					Real right_coeff = node_volumes(x + 1, y);

					diag_coeff += right_coeff;

					// v_xx term
					int ridx = solvable_faces(x + 1, y, 1);
					if (ridx >= 0)
						solver.add_element(idx, ridx, -right_coeff);
					else if (m_colvel_set && ridx == COLLISION)
						solver.add_rhs(idx, right_coeff * m_collision_vel(x + 1, y, 1));

					// u_xy terms
					int rybidx = solvable_faces(x + 1, y - 1, 0);
					if (rybidx >= 0)
						solver.add_element(idx, rybidx, right_coeff);
					else if (m_colvel_set && rybidx == COLLISION)
						solver.add_rhs(idx, -right_coeff * m_collision_vel(x + 1, y - 1, 0));

					int rxtidx = solvable_faces(x + 1, y, 0);
					if (rxtidx >= 0)
						solver.add_element(idx, rxtidx, -right_coeff);
					else if (m_colvel_set && rxtidx == COLLISION)
						solver.add_rhs(idx, right_coeff * m_collision_vel(x + 1, y, 0));
				}

				// Left
				{
					Real left_coeff = node_volumes(x, y);

					diag_coeff += left_coeff;

					// v_xx term
					int lidx = solvable_faces(x - 1, y, 1);
					if (lidx >= 0)
						solver.add_element(idx, lidx, -left_coeff);
					else if (m_colvel_set && lidx == COLLISION)
						solver.add_rhs(idx, left_coeff * m_collision_vel(x - 1, y, 1));

					// u_xy terms
					int lxbidx = solvable_faces(x, y - 1, 0);
					if (lxbidx >= 0)
						solver.add_element(idx, lxbidx, -left_coeff);
					else if (m_colvel_set && lxbidx == COLLISION)
						solver.add_rhs(idx, left_coeff * m_collision_vel(x, y - 1, 0));

					int lxtidx = solvable_faces(x, y, 0);
					if (lxtidx >= 0)
						solver.add_element(idx, lxtidx, left_coeff);
					else if (m_colvel_set && lxtidx == COLLISION)
						solver.add_rhs(idx, -left_coeff * m_collision_vel(x, y, 0));

				}

				solver.add_element(idx, idx, diag_coeff);

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
	x_size = m_vel.size(0)[0];
	y_size = m_vel.size(0)[1];
	for (size_t x = 0; x < x_size; ++x)
		for (size_t y = 0; y < y_size; ++y)
		{
			int idx = solvable_faces(x, y, 0);
			if (idx >= 0)
				m_vel(x, y, 0) = solver.sol(idx);
			else if (idx == COLLISION)
			{
				if (m_colvel_set)
					m_vel(x, y, 0) = m_collision_vel(x, y, 0);
				else
					m_vel(x, y, 0) = 0.;
			}
		}

	x_size = m_vel.size(1)[0];
	y_size = m_vel.size(1)[1];
	for (size_t x = 0; x < x_size; ++x)
		for (size_t y = 0; y < y_size; ++y)
		{
			int idx = solvable_faces(x, y, 1);
			if (idx >= 0)
				m_vel(x, y, 1) = solver.sol(idx);
			else if (idx == COLLISION)
			{
				if (m_colvel_set)
					m_vel(x, y, 1) = m_collision_vel(x, y, 1);
				else
					m_vel(x, y, 1) = 0.;
			}

		}
	
	//m_vel.draw_sample_point_vectors(renderer, Vec3f(1, 0, 0), m_dt * m_surface.dx());
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