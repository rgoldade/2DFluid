#include "AnalyticalViscositySolver.h"

void AnalyticalViscositySolver::draw_grid(Renderer& renderer) const
{
	m_vel_index.draw_grid(renderer);
}

void AnalyticalViscositySolver::draw_active_velocity(Renderer& renderer) const
{
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel_index.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			if (m_vel_index(face, axis) >= 0)
			{
				Vec2R pos = m_vel_index.idx_to_ws(Vec2R(face), axis);

				renderer.add_point(pos, colours[axis], 5);
			}
		});
	}
}

unsigned AnalyticalViscositySolver::set_velocity_index()
{
	// Loop over each face. If it's not along the boundary
	// then include it into the system.

	unsigned index = 0;

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui size = m_vel_index.size(axis);

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			// Faces along the boundary are removed from the simulation
			if (!(face[axis] == 0 || face[axis] == size[axis] - 1))
				m_vel_index(face[0], face[1], axis) = index++;
		});
	}
	// Returning index gives the number of velocity positions required for a linear system
	return index;
}