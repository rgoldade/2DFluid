#include "AnalyticalViscositySolver.h"

void AnalyticalViscositySolver::drawGrid(Renderer& renderer) const
{
	myVelocityIndex.drawGrid(renderer);
}

void AnalyticalViscositySolver::drawActiveVelocity(Renderer& renderer) const
{
	for (auto axis : { 0, 1 })
	{
		Vec2ui size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			if (myVelocityIndex(face, axis) >= 0)
			{
				Vec2R worldPosition = myVelocityIndex.indexToWorld(Vec2R(face), axis);

				renderer.addPoint(worldPosition, colours[axis], 5);
			}
		});
	}
}

unsigned AnalyticalViscositySolver::setVelocityIndex()
{
	// Loop over each face. If it's not along the boundary
	// then include it into the system.

	unsigned index = 0;

	for (auto axis : { 0, 1 })
	{
		Vec2ui size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			// Faces along the boundary are removed from the simulation
			if (!(face[axis] == 0 || face[axis] == size[axis] - 1))
				myVelocityIndex(face[0], face[1], axis) = index++;
		});
	}
	// Returning index gives the number of velocity positions required for a linear system
	return index;
}