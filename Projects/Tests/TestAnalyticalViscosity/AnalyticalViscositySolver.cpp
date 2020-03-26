#include "AnalyticalViscositySolver.h"

void AnalyticalViscositySolver::drawGrid(Renderer& renderer) const
{
	myVelocityIndex.drawGrid(renderer);
}

void AnalyticalViscositySolver::drawActiveVelocity(Renderer& renderer) const
{
	for (int axis : { 0, 1 })
	{
		Vec2i size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			if (myVelocityIndex(face, axis) >= 0)
			{
				Vec2f worldPosition = myVelocityIndex.indexToWorld(Vec2f(face), axis);

				renderer.addPoint(worldPosition, colours[axis], 5);
			}
		});
	}
}

unsigned AnalyticalViscositySolver::setVelocityIndex()
{
	// Loop over each face. If it's not along the boundary
	// then include it into the system.

	int index = 0;

	for (int axis : { 0, 1 })
	{
		Vec2i size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			// Faces along the boundary are removed from the simulation
			if (!(face[axis] == 0 || face[axis] == size[axis] - 1))
				myVelocityIndex(face[0], face[1], axis) = index++;
		});
	}
	// Returning index gives the number of velocity positions required for a linear system
	return index;
}