#include "AnalyticalViscositySolver.h"

void AnalyticalViscositySolver::draw_grid(Renderer& renderer) const
{
	m_vel_index.draw_grid(renderer);
}

void AnalyticalViscositySolver::draw_active_velocity(Renderer& renderer) const
{
	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel_index.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				if (m_vel_index(i, j, axis) >= 0)
				{
					Vec2R pos = m_vel_index.idx_to_ws(Vec2R(i, j), axis);

					renderer.add_point(pos, colours[axis], 5);
				}
			}

	}
}

size_t AnalyticalViscositySolver::set_velocity_index()
{
	// Loop over each face. If it's not along the boundary
	// then include it into the system.

	size_t index = 0;

	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2st size = m_vel_index.size(axis);

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				Vec2i face(i, j);

				// Faces along the boundary are removed from the simulation
				if (face[axis] == 0 || face[axis] == size[axis] - 1)
					m_vel_index(face[0], face[1], axis) = -2;
				else
					m_vel_index(face[0], face[1], axis) = index++;
			}
	}
	// Returning index gives the number of velocity positions required for a linear system
	return index;
}