#include <iostream>

#include "ViscositySolver.h"

#include "ComputeWeights.h"
#include "Solver.h"
#include "VectorGrid.h"

void ViscositySolver(const Real dt,
	const LevelSet& surface,
	VectorGrid<Real>& velocity,
	const LevelSet& solidSurface,
	const VectorGrid<Real>& solidVelocity,
	const ScalarGrid<Real>& viscosity)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled surface and collision
	assert(surface.isGridMatched(solidSurface));
	assert(surface.isGridMatched(viscosity));

	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled surface and collision
	assert(velocity.size(0)[0] - 1 == surface.size()[0] &&
			velocity.size(0)[1] == surface.size()[1] &&
			velocity.size(1)[0] == surface.size()[0] &&
			velocity.size(1)[1] - 1 == surface.size()[1]);

	assert(solidVelocity.size(0)[0] - 1 == surface.size()[0] &&
			solidVelocity.size(0)[1] == surface.size()[1] &&
			solidVelocity.size(1)[0] == surface.size()[0] &&
			solidVelocity.size(1)[1] - 1 == surface.size()[1]);

	int samples = 3;

	ScalarGrid<Real> centerAreas = computeSuperSampledAreas(surface, ScalarGridSettings::SampleType::CENTER, 3);
	ScalarGrid<Real> nodeAreas = computeSuperSampledAreas(surface, ScalarGridSettings::SampleType::NODE, 3);
	VectorGrid<Real> faceAreas = computeSuperSampledFaceAreas(surface, 3);

	VectorGrid<int> liquidFaces(surface.xform(), surface.size(), ViscosityCellLabels::UNSOLVED_CELL, VectorGridSettings::SampleType::STAGGERED);

	// Build solvable faces. Assumes the grid limits are solid boundaries and left out
	// of the system.

	int liquidDOFCount = 0;

	for (int axis : {0, 1})
	{
		Vec2i size = liquidFaces.size(axis);

		Vec2i start(0); ++start[axis];
		Vec2i end(size); --end[axis];

		forEachVoxelRange(start, end, [&](const Vec2i& face)
		{
			bool inSolve = false;

			for (int direction : {0, 1})
			{
				Vec2i cell = faceToCell(face, axis, direction);
				if (centerAreas(cell) > 0.) inSolve = true;
			}

			if (!inSolve)
			{
				for (int direction : {0, 1})
				{
					Vec2i node = faceToNode(face, axis, direction);
					if (nodeAreas(node) > 0.) inSolve = true;
				}
			}

			if (inSolve)
			{
				if (solidSurface.interp(liquidFaces.indexToWorld(Vec2R(face), axis)) <= 0.)
					liquidFaces(face, axis) = ViscosityCellLabels::SOLID_CELL;
				else
					liquidFaces(face, axis) = liquidDOFCount++;
			}
		});
	}

	// Build a single container of the viscosity weights (liquid volumes, gf weights, viscosity coefficients)
	Real invDx2 = 1. / Util::sqr(surface.dx());
	forEachVoxelRange(Vec2i(0), centerAreas.size(), [&](const Vec2i& cell)
	{
		centerAreas(cell) *= dt * invDx2;
		centerAreas(cell) *= viscosity(cell);
	});
	
	forEachVoxelRange(Vec2i(0), nodeAreas.size(), [&](const Vec2i& node)
	{
		nodeAreas(node) *= dt * invDx2;
		nodeAreas(node) *= viscosity.interp(nodeAreas.indexToWorld(Vec2R(node)));
	});

	Solver<true> solver(liquidDOFCount, liquidDOFCount * 9);

	for (int axis : {0, 1})
	{
		Vec2i faceGridSize = velocity.size(axis);
		forEachVoxelRange(Vec2i(0), faceGridSize, [&](const Vec2i& face)
		{
			int index = liquidFaces(face, axis);
			if (index >= 0)
			{
				Real volume = faceAreas(face, axis);

				// Build RHS with weight velocities
				solver.addToRhs(index, velocity(face, axis) * volume);
				solver.addToGuess(index, velocity(face, axis));

				// Add control volume weight on the diagonal.
				solver.addToElement(index, index, volume);

				// Build cell-centered stresses.
				for (int cellDirection : {0, 1})
				{
					Vec2i cell = faceToCell(face, axis, cellDirection);

					Real coeff = 2. * centerAreas(cell);

					Real centerSign = (cellDirection == 0) ? -1. : 1.;

					for (int faceDirection : {0, 1})
					{
						// Since we've assumed grid boundaries are static solids
						// we don't have any solveable faces with adjacent cells
						// out of the grid bounds. We can skip that check here.

						Vec2i adjacentFace = cellToFace(cell, axis, faceDirection);

						Real faceSign = (faceDirection == 0) ? -1. : 1.;

						int faceIndex = liquidFaces(adjacentFace, axis);
						if (faceIndex >= 0)
							solver.addToElement(index, faceIndex, -centerSign * faceSign * coeff);
						else if (faceIndex == ViscosityCellLabels::SOLID_CELL)
							solver.addToRhs(index, centerSign * faceSign * coeff * solidVelocity(adjacentFace, axis));
					}
				}

				// Build node stresses.
				for (int nodeDirection : {0, 1})
				{
					Vec2i node = faceToNode(face, axis, nodeDirection);

					Real nodeSign = (nodeDirection == 0) ? -1. : 1.;

					Real coeff = nodeAreas(node);

					for (int gradientAxis : {0, 1})
						for (int faceDirection : {0, 1})
						{
							Vec2i adjacentFace = nodeToFace(node, gradientAxis, faceDirection);

							Real faceSign = faceDirection == 0 ? -1. : 1.;

							if (adjacentFace[gradientAxis] >= 0 && adjacentFace[gradientAxis] < faceGridSize[gradientAxis])
							{
								int faceAxis = (gradientAxis + 1) % 2;
								int faceIndex = liquidFaces(adjacentFace, faceAxis);

								if (faceIndex >= 0)
									solver.addToElement(index, faceIndex, -nodeSign * faceSign * coeff);
								else if (faceIndex == ViscosityCellLabels::SOLID_CELL)
									solver.addToRhs(index, nodeSign * faceSign * coeff * solidVelocity(adjacentFace, faceAxis));
							}
						}
				}
			}
		});
	}

	bool solved = solver.solveIterative();

	if (!solved)
	{
		std::cout << "Viscosity failed to solve" << std::endl;
		assert(false);
	}

	// Update velocity
	for (int axis : {0, 1})
	{
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			int index = liquidFaces(face, axis);
			if (index >= 0)
				velocity(face, axis) = solver.solution(index);
			else if (index == ViscosityCellLabels::SOLID_CELL)
				velocity(face, axis) = solidVelocity(face, axis);
		});
	}
}