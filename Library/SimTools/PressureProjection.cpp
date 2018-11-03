#include <iostream>

#include "PressureProjection.h"
#include "Solver.h"

void PressureProjection::draw_pressure(Renderer& renderer) const
{
	m_pressure.draw_supersampled_values(renderer, .25, 3, 2);
}

void PressureProjection::project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights)
{
	assert(liquid_weights.is_matched(fluid_weights) && liquid_weights.is_matched(m_vel));

	UniformGrid<int> solvable_cells(m_surface.size(), UNSOLVED);

	// This loop is dumb in serial but it's here as a placeholder for parallel later
	Real dx = m_surface.dx();
	int solve_count = 0;
	for_each_voxel_range(Vec2ui(0), solvable_cells.size(), [&](const Vec2ui& cell)
	{
		Real sdf = m_surface(cell);
		bool in_fluid = (sdf < 0.);
		
		// Implicit extrapolation of the fluid into the collision. This is an important piece
		// of Batty et al. 2007. A cell whose center falls inside the solid could still have fluid
		// in the cell. By extrapolating, we are sure to get all partially filled cells.
		if (!in_fluid)
		{
			if (sdf < .5 * dx && m_collision(cell) < 0.) in_fluid = true;
		}

		if (in_fluid)
		{
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				// A solvable pressure sample point should be inside in the surface
				// and with at least one non-zero cut-cell length (i.e. not extrapolated
				// too far into the collision volume)
				Vec2ui face = cell + cell_to_face[dir];

				unsigned axis = dir / 2;
				if (fluid_weights(face, axis) > 0)
				{
					solvable_cells(cell) = solve_count++;
					break;
				}
			}
		}
	});

	Solver<true> solver(solve_count, solve_count * 5);

	// Build linear system
	double inv_dx = 1. / dx;
	for_each_voxel_range(Vec2ui(0), solvable_cells.size(), [&](const Vec2ui& cell)
	{
		int idx = solvable_cells(cell);
		if (idx >= 0)
		{
			// Build RHS divergence
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2ui face = cell + cell_to_face[dir];
				
				unsigned axis = dir / 2;

				Real weight = fluid_weights(face, axis);

				if (weight > 0)
				{
					double sign = (dir % 2 == 0) ? 1 : -1;
					double div = sign * m_vel(face, axis) * weight * inv_dx;

					div += sign * m_collision_vel(face, axis) * (1.0 - weight) * inv_dx;

					solver.add_rhs(idx, div);
				}
			}

			// Build row
			double middle_coeff = 0.;

			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i adjacent_cell = Vec2i(cell) + cell_to_cell[dir];

				unsigned axis = dir / 2;
				
				// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
				if (adjacent_cell[axis] < 0 || adjacent_cell[axis] >= m_surface.size()[axis]) continue;

				Vec2ui face = cell + cell_to_face[dir];
				
				double weight = fluid_weights(face, axis);

				if (weight > 0)
				{
					double coeff = weight * m_dt * sqr(inv_dx);

					// If neighbouring cell is solvable, it should have an entry in the system
					int cidx = solvable_cells(Vec2ui(adjacent_cell));
					if (cidx >= 0)
					{
						solver.add_element(idx, cidx, -coeff);
						middle_coeff += coeff;
					}
					else
					{
						Real theta = liquid_weights(face, axis);

						theta = clamp(theta, MINTHETA, Real(1.));
						middle_coeff += coeff / theta;

						// TODO: add surface tension
					}
				}
			}

			if (middle_coeff > 0.)	solver.add_element(idx, idx, middle_coeff);
			else solver.add_element(idx, idx, 1.);
		}
	});

	bool result = solver.solve();

	if (!result)
	{
		std::cout << "Pressure projection failed to solve" << std::endl;
		return;
	}

	// Load solution into pressure grid
	for_each_voxel_range(Vec2ui(0), solvable_cells.size(), [&](const Vec2ui& cell)
	{
		int idx = solvable_cells(cell);
		if (idx >= 0)
		{
			double p = solver.sol(idx);
			m_pressure(cell) = p;
		}
	});

	// Set valid faces
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2i backward_cell = Vec2i(face) + face_to_cell[axis][0];
			Vec2i forward_cell = Vec2i(face) + face_to_cell[axis][1];

			bool out_of_bounds = (backward_cell[axis] < 0 || forward_cell[axis] >= m_surface.size()[axis]);
			if (!out_of_bounds)
			{
				if (solvable_cells(Vec2ui(backward_cell)) >= 0 || solvable_cells(Vec2ui(forward_cell)) >= 0)
					m_valid(face, axis) = 1;
			}
		});
	}
}

void PressureProjection::apply_solution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights)
{
	assert(liquid_weights.is_matched(fluid_weights) && liquid_weights.is_matched(m_vel));
	
	Real inv_dx = 1 / m_surface.dx();
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui vel_size = vel.size(axis);

		for_each_voxel_range(Vec2ui(0), vel_size, [&](const Vec2ui& face)
		{
			Real temp_vel = 0;
			if (m_valid(face, axis) > 0)
			{
				Real theta = liquid_weights(face, axis);

				if (theta > 0.)
				{
					theta = clamp(theta, MINTHETA, Real(1.));

					Vec2i backward_cell = Vec2i(face) + face_to_cell[axis][0];
					Vec2i forward_cell = Vec2i(face) + face_to_cell[axis][1];

					bool out_of_bounds = (backward_cell[axis] < 0 || forward_cell[axis] >= m_surface.size()[axis]);
					if (!out_of_bounds)
					{
						Real backward_pressure = 0.;

						Vec2ui ubcell(backward_cell);
						if (m_surface(ubcell) < 0.)
							backward_pressure = m_pressure(ubcell);

						Real forward_pressure = 0.;

						Vec2ui ufcell(forward_cell);
						if (m_surface(ufcell) < 0.)
							forward_pressure = m_pressure(ufcell);

						temp_vel = m_vel(face, axis) - m_dt * (forward_pressure - backward_pressure) * inv_dx / theta;
					}
				} 
			}

			vel(face, axis) = temp_vel;
		});
	}
}

void PressureProjection::apply_valid(VectorGrid<Real> &valid)
{
	valid = m_valid;
}