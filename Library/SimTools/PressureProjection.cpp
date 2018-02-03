#include <vector>
#include <array>
#include <queue>

#include "PressureProjection.h"
#include "Solver.h"

static int UNSOLVED = -1;
static int SOLVED = 0;

static int unvisited = -2;
static int visited = -1;
static int finished = 0;

void PressureProjection::draw_pressure(Renderer& renderer) const
{
	m_pressure.draw_supersampled_values(renderer, .25, 3, 2);
}

void PressureProjection::draw_divergence(Renderer& renderer) const
{
	ScalarGrid<Real> div(m_surface.xform(), m_surface.size(), 0);

	for (int x = 0; x < div.size()[0]; ++x)
		for (int y = 0; y < div.size()[1]; ++y)
		{
			if (m_surface(x, y) < 0)
			{
				for (int dir = 0; dir < 4; ++dir)
				{
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];
					double sign = (dir % 2 == 0) ? -1 : 1;
					div(x, y) = sign * m_vel(fx, fy, dir / 2) / m_surface.dx();
				}
			}
			else div(x, y) = 0;
		}
	Real max = div.maxval();
	Real min = div.minval();
	div.draw_supersampled_values(renderer, .25, 3, 2);
}

void PressureProjection::project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights, const ScalarGrid<Real>& center_weights, Renderer& renderer)
{
	assert(liquid_weights.size(0) == fluid_weights.size(0) &&
			liquid_weights.size(1) == fluid_weights.size(1) &&
			liquid_weights.size(0) == m_vel.size(0) &&
			liquid_weights.size(1) == m_vel.size(1));

	UniformGrid<int> solvable_cells(m_surface.size(), UNSOLVED);

	// This loop is dumb in serial but it's here as a placeholder for parallel later
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			if (m_surface(x, y) < 0.)
			{
				for (int dir = 0; dir < 4; ++dir)
				{
					// A solvable pressure sample point should be inside in the surface
					// and with at least one non-zero cut-cell length (i.e. not extrapolated
					// too far into the collision volume)
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];

					int axis = dir / 2;
					if (fluid_weights(fx, fy, axis) > 0)
						solvable_cells(x, y) = SOLVED;
				}
			}
		}

	int solvecount = 0;
	for (size_t x = 0; x < solvable_cells.size()[0]; ++x)
		for (size_t y = 0; y < solvable_cells.size()[1]; ++y)
		{
			if (solvable_cells(x, y) >= 0)
				solvable_cells(x, y) = solvecount++;
		}


	Solver solver(solvecount, solvecount * 7);

	// Build linear system
	double dx = m_surface.dx();
	Real cell_vol = sqr(dx);
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			int idx = solvable_cells(x, y);
			if (idx >= 0)
			{
				// Build RHS divergence
				for (int dir = 0; dir < 4; ++dir)
				{
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];
					double sign = (dir % 2 == 0) ? 1 : -1;
					double div = sign * m_vel(fx, fy, dir / 2) * fluid_weights(fx, fy, dir / 2) / dx;
					
					solver.add_rhs(idx, div);
				}

				// Build collision vel divergence
				if (m_colvel_set)
				{
					for (int dir = 0; dir < 4; ++dir)
					{
						int fx = x + cell_to_face[dir][0];
						int fy = y + cell_to_face[dir][1];
						double sign = (dir % 2 == 0) ? 1. : -1.;
						double div = sign * m_collision_vel(fx, fy, dir / 2) * (1.0 - fluid_weights(fx, fy, dir / 2)) / dx;

						solver.add_rhs(idx, div);
					}
				}

				/*if(m_divergence)
					solver.add_rhs(idx, (*m_divergence)(x, y));*/

				// Build row
				double middle_coeff = 0;
				for (int dir = 0; dir < 4; ++dir)
				{
					int cx = x + cell_offset[dir][0];
					int cy = y + cell_offset[dir][1];

					// Bounds check. If out-of-bounds, treat like a grid-aligned collision.
					if (cx < 0 || cy < 0 || cx >= m_surface.size()[0] || cy >= m_surface.size()[1]) continue;

					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];

					double coeff = fluid_weights(fx, fy, dir / 2 ) * m_dt / dx / dx;
					
					// If neighbouring cell is solvable, it should have an entry in the system
					int cidx = solvable_cells(cx, cy);
					if (cidx >= 0)
					{
						solver.add_element(idx, cidx, -coeff);
						middle_coeff += coeff;
					}
					else if (fluid_weights(fx, fy, dir / 2) > 0.) // Preventing any complications deep inside collision
					{
						double theta = liquid_weights(fx, fy, dir / 2);
						if (theta < 0.01) theta = 0.01;
						middle_coeff += coeff / theta;

						// Add surface pressure to RHS
						if (m_sp_set)
						{
							Real spress = lerp(m_surface_pressure(cx, cy), m_surface_pressure(x, y), theta);
							solver.add_rhs(idx, spress * m_sp_scale * coeff / theta);
						}

					}					
				}

				if (middle_coeff > 0.)	solver.add_element(idx, idx, middle_coeff);
				else solver.add_element(idx, idx, 1.);
			}
		}

	solver.solve();

	// Load solution into pressure grid
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			int idx = solvable_cells(x, y);
			if (idx >= 0)
			{
				double p = solver.sol(idx);
				m_pressure(x, y) = p;
			}
		}
}

void PressureProjection::apply_solution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights)
{
	assert(vel.size(0) == m_vel.size(0) &&
			vel.size(1) == m_vel.size(1) &&
			liquid_weights.size(0) == m_vel.size(0) &&
			liquid_weights.size(1) == m_vel.size(1) &&
			fluid_weights.size(0) == m_vel.size(0) &&
			fluid_weights.size(1) == m_vel.size(1));
			
	for (int y = 0; y < vel.size(0)[1]; ++y)
	{
		vel(0, y, 0) = 0.;
		vel(vel.size(0)[0] - 1, y, 0) = 0.;
	}

	for (int x = 1; x < vel.size(0)[0] - 1; ++x)
		for (int y = 0; y < vel.size(0)[1]; ++y)
		{
			Vec2i cb = Vec2i(x - 1, y);
			Vec2i cf = Vec2i(x, y);

			if (fluid_weights(x, y, 0) > 0.)
			{
				Real theta = liquid_weights(x, y, 0);
				if (theta > 0.)
				{
					if (theta < 0.01) theta = 0.01;

					Real pb = 0;
					if (m_surface(cb[0], cb[1]) < 0.)
						pb = m_pressure(cb[0], cb[1]);
					else if (m_sp_set)
					{
						pb += lerp(m_surface_pressure(cb[0], cb[1]), m_surface_pressure(cf[0], cf[1]), theta) * m_sp_scale;
					}

					Real pf = 0;
					if (m_surface(cf[0], cf[1]) < 0.)
						pf = m_pressure(cf[0], cf[1]);
					else if (m_sp_set)
					{
						pf += lerp(m_surface_pressure(cf[0], cf[1]), m_surface_pressure(cb[0], cb[1]), theta) * m_sp_scale;
					}
										
					vel(x, y, 0) = m_vel(x, y, 0) - m_dt * (pf - pb) / m_surface.dx() / theta;
					m_valid(x, y, 0) = 1.;
				}
				else vel(x, y, 0) = 0.;
			}
			else vel(x, y, 0) = 0.;
		}

	for (int x = 0; x < vel.size(1)[0]; ++x)
	{
		vel(x, 0, 1) = 0.;
		vel(x, vel.size(1)[1] - 1, 1) = 0.;
	}

	for (int x = 0; x < vel.size(1)[0]; ++x)
		for (int y = 1; y < vel.size(1)[1] - 1; ++y)
		{
			Vec2i cb = Vec2i(x, y - 1);
			Vec2i cf = Vec2i(x, y);

			if (fluid_weights(x, y, 1) > 0.)
			{
				Real theta = liquid_weights(x, y, 1);
				if (theta > 0.)
				{
					if (theta < 0.01) theta = 0.01;

					Real pb = 0;
					if (m_surface(cb[0], cb[1]) < 0.)
						pb = m_pressure(cb[0], cb[1]);
					else if (m_sp_set)
					{
						pb += lerp(m_surface_pressure(cb[0], cb[1]), m_surface_pressure(cf[0], cf[1]), theta) * m_sp_scale;
					}

					Real pf = 0;
					if (m_surface(cf[0], cf[1]) < 0.)
						pf = m_pressure(cf[0], cf[1]);
					else
					{
						if (m_sp_set)
							pf += lerp(m_surface_pressure(cf[0], cf[1]), m_surface_pressure(cb[0], cb[1]), theta) * m_sp_scale;
					}

					vel(x, y, 1) = m_vel(x, y, 1) - m_dt * (pf - pb) / m_surface.dx() / theta;
					m_valid(x, y, 1) = 1.;
				}
				else vel(x, y, 1) = 0.;
			}
			else vel(x, y, 1) = 0.;
		}
}

void PressureProjection::apply_valid(VectorGrid<Real> &valid)
{
	valid = m_valid;
}