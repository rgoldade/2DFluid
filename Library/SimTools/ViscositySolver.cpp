#include <iostream>

#include "ViscositySolver.h"

#include "Solver.h"
#include "VectorGrid.h"

void ViscositySolver::solve(const VectorGrid<Real>& faceVolumes,
							ScalarGrid<Real>& centerVolumes,
							ScalarGrid<Real>& nodeVolumes,
							const ScalarGrid<Real>& solidCenterVolumes,
							const ScalarGrid<Real>& solidNodeVolumes)
{
	// Debug check that grids are the same
	assert(myVelocity.isGridMatched(faceVolumes));
	assert(mySurface.isGridMatched(centerVolumes));
	assert(mySurface.isGridMatched(solidCenterVolumes));
	assert(mySurface.size() + Vec2ui(1) == solidNodeVolumes.size());
	assert(mySurface.size() + Vec2ui(1) == nodeVolumes.size());

	VectorGrid<int> liquidFaces(mySurface.xform(), mySurface.size(), UNSOLVED, VectorGridSettings::SampleType::STAGGERED);

	// Build solvable faces. Assumes the grid limits are solid boundaries and left out
	// of the system.

	unsigned liquidDOFCount = 0;

	for (unsigned axis : {0, 1})
	{
		Vec2ui size = liquidFaces.size(axis);

		Vec2ui start(0); ++start[axis];
		Vec2ui end(size); --end[axis];

		forEachVoxelRange(start, end, [&](const Vec2ui& face)
		{
			bool inSolve = false;

			for (unsigned direction : {0, 1})
			{
				Vec2i cell = faceToCell(Vec2i(face), axis, direction);
				if (centerVolumes(Vec2ui(cell)) > 0.) inSolve = true;
			}

			if (!inSolve)
			{
				for (unsigned direction : {0, 1})
				{
					Vec2ui node = faceToNode(face, axis, direction);
					if (nodeVolumes(node) > 0.) inSolve = true;
				}
			}

			if (inSolve)
			{
				if (mySolidSurface.interp(liquidFaces.indexToWorld(Vec2R(face), axis)) <= 0.)
					liquidFaces(face, axis) = SOLIDBOUNDARY;
				else
					liquidFaces(face, axis) = liquidDOFCount++;
			}
		});
	}

	// Build a single container of the viscosity weights (liquid volumes, gf weights, viscosity coefficients)
	Real invDx2 = 1. / Util::sqr(mySurface.dx());
	{
		forEachVoxelRange(Vec2ui(0), centerVolumes.size(), [&](const Vec2ui& cell)
		{
			centerVolumes(cell) *= myDt * invDx2;
			centerVolumes(cell) *= myViscosity(cell);
			centerVolumes(cell) *= Util::clamp(solidCenterVolumes(cell), 0.01, 1.);
		});
	}

	{
		forEachVoxelRange(Vec2ui(0), nodeVolumes.size(), [&](const Vec2ui& node)
		{
			nodeVolumes(node) *= myDt * invDx2;
			nodeVolumes(node) *= myViscosity.interp(nodeVolumes.indexToWorld(Vec2R(node)));
			nodeVolumes(node) *= Util::clamp(solidNodeVolumes(node), 0.01, 1.);
		});
	}

	Solver<true> solver(liquidDOFCount, liquidDOFCount * 9);

	for (unsigned axis : {0, 1})
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int index = liquidFaces(face, axis);
			if (index >= 0)
			{
				Real volume = faceVolumes(face, axis);

				// Build RHS with weight velocities
				solver.addToRhs(index, myVelocity(face, axis) * volume);
				solver.addToGuess(index, myVelocity(face, axis));

				// Add control volume weight on the diagonal.
				solver.addToElement(index, index, volume);

				// Build cell-centered stresses.
				for (unsigned cellDirection : {0, 1})
				{
					Vec2i cell = faceToCell(Vec2i(face), axis, cellDirection);

					Real coeff = 2. * centerVolumes(Vec2ui(cell));

					Real centerSign = (cellDirection == 0) ? -1. : 1.;

					for (unsigned faceDirection : {0, 1})
					{
						// Since we've assumed grid boundaries are static solids
						// we don't have any solveable faces with adjacent cells
						// out of the grid bounds. We can skip that check here.

						Vec2ui adjacentFace = cellToFace(Vec2ui(cell), axis, faceDirection);

						Real faceSign = (faceDirection == 0) ? -1. : 1.;

						int faceIndex = liquidFaces(adjacentFace, axis);
						if (faceIndex >= 0)
							solver.addToElement(index, faceIndex, -centerSign * faceSign * coeff);
						else if (faceIndex == SOLIDBOUNDARY)
							solver.addToRhs(index, centerSign * faceSign * coeff * mySolidVelocity(adjacentFace, axis));
					}
				}

				// Build node stresses.
				for (unsigned nodeDirection : {0, 1})
				{
					Vec2ui node = faceToNode(face, axis, nodeDirection);

					Real nodeSign = (nodeDirection == 0) ? -1. : 1.;

					Real coeff = nodeVolumes(node);

					for (unsigned gradientAxis : {0, 1})
						for (unsigned faceDirection : {0, 1})
						{
							Vec2i adjacentFace = nodeToFace(Vec2i(node), gradientAxis, faceDirection);

							Real faceSign = faceDirection == 0 ? -1. : 1.;

							if (adjacentFace[gradientAxis] >= 0 && adjacentFace[gradientAxis] < size[gradientAxis])
							{
								unsigned faceAxis = (gradientAxis + 1) % 2;
								int faceIndex = liquidFaces(Vec2ui(adjacentFace), faceAxis);

								if (faceIndex >= 0)
									solver.addToElement(index, faceIndex, -nodeSign * faceSign * coeff);
								else if (faceIndex == SOLIDBOUNDARY)
									solver.addToRhs(index, nodeSign * faceSign * coeff * mySolidVelocity(Vec2ui(adjacentFace), faceAxis));
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
	for (unsigned axis : {0, 1})
	{
		Vec2ui size = myVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int index = liquidFaces(face, axis);
			if (index >= 0)
				myVelocity(face, axis) = solver.solution(index);
			else if (index == SOLIDBOUNDARY)
				myVelocity(face, axis) = mySolidVelocity(face, axis);
		});
	}
}

void ViscositySolver::setViscosity(Real mu)
{
	myViscosity = ScalarGrid<Real>(mySurface.xform(), mySurface.size(), mu);
}

void ViscositySolver::setViscosity(const ScalarGrid<Real>& mu)
{
	assert(mu.size() == mySurface.size());
	myViscosity = mu;
}