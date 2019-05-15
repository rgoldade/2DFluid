#include <iostream>

#include "PressureProjection.h"
#include "Solver.h"

void PressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .25, 1, 2);
}

void PressureProjection::project(const VectorGrid<Real>& ghostFluidWeights, const VectorGrid<Real>& cutCellWeights)
{
	assert(ghostFluidWeights.isMatched(cutCellWeights) && ghostFluidWeights.isMatched(myFluidVelocity));

	// This loop is dumb in serial but it's here as a placeholder for parallel later
	Real dx = myFluidSurface.dx();
	int liquidDOFCount = 0;
	forEachVoxelRange(Vec2ui(0), myFluidCellIndex.size(), [&](const Vec2ui& cell)
	{
		if (myFluidSurface(cell) <= 0)
		{
			for (auto axis : { 0,1 })
				for (unsigned direction : {0, 1})
				{
					Vec2ui face = cellToFace(cell, axis, direction);

					if (cutCellWeights(face, axis) > 0)
					{
						myFluidCellIndex(cell) = liquidDOFCount++;
						return;
					}
				}
		}
	});

	Solver<true> solver(liquidDOFCount, liquidDOFCount * 5);

	// Build linear system
	forEachVoxelRange(Vec2ui(0), myFluidCellIndex.size(), [&](const Vec2ui& cell)
	{
		int row = myFluidCellIndex(cell);
		if (row >= 0)
		{
			// Build RHS divergence
			double divergence = 0;
			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2ui face = cellToFace(cell, axis, direction);

					Real weight = cutCellWeights(face, axis);

					double sign = (direction == 0) ? 1 : -1;

					if (weight > 0)
						divergence += sign * myFluidVelocity(face, axis) * weight;
					if (weight < 1.)
						divergence += sign * mySolidVelocity(face, axis) * (1. - weight);
				}

			solver.addRhs(row, divergence);

			// Build row
			double diagonal = 0.;

			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);
			
					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= myFluidSurface.size()[axis]) continue;

					Vec2ui face = cellToFace(cell, axis, direction);
				
					double weight = cutCellWeights(face, axis);

					if (weight > 0)
					{
						// If neighbouring cell is solvable, it should have an entry in the system
						int adjacentRow = myFluidCellIndex(Vec2ui(adjacentCell));
						if (adjacentRow >= 0)
						{
							solver.addElement(row, adjacentRow, -weight);
							diagonal += weight;
						}
						else
						{
							Real theta = ghostFluidWeights(face, axis);

							theta = Util::clamp(theta, MINTHETA, Real(1.));
							diagonal += weight / theta;

							// TODO: add surface tension
						}
					}
				}
			assert(diagonal > 0);
			solver.addElement(row, row, diagonal);
		}
	});

	
	bool result = solver.solveIterative();

	if (!result)
	{
		std::cout << "Pressure projection failed to solve" << std::endl;
		return;
	}

	// Load solution into pressure grid
	forEachVoxelRange(Vec2ui(0), myFluidCellIndex.size(), [&](const Vec2ui& cell)
	{
		int row = myFluidCellIndex(cell);
		if (row >= 0)
			myPressure(cell) = solver.solution(row);
		else
			myPressure(cell) = 0;
	});

	// Set valid faces
	for (unsigned axis : {0, 1})
	{
		Vec2ui size = myFluidVelocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
			Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

			if (!(backwardCell[axis] < 0 || forwardCell[axis] >= myFluidSurface.size()[axis]))
			{
				if ((myFluidCellIndex(Vec2ui(backwardCell)) >= 0 || myFluidCellIndex(Vec2ui(forwardCell)) >= 0) &&
					cutCellWeights(face, axis) > 0)
					myValid(face, axis) = MarkedCells::FINISHED;
				else myValid(face, axis) = MarkedCells::UNVISITED;
			}
			else myValid(face, axis) = MarkedCells::UNVISITED;
		});
	}
}

void PressureProjection::applySolution(VectorGrid<Real>& velocity, const VectorGrid<Real>& liquidWeights)
{
	assert(liquidWeights.isMatched(myFluidVelocity));
	
	for (unsigned axis : {0, 1})
	{
		Vec2ui size = velocity.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			Real localVelocity = 0;
			if (myValid(face, axis) == MarkedCells::FINISHED)
			{
				Real theta = liquidWeights(face, axis);
				theta = Util::clamp(theta, MINTHETA, Real(1.));

				Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
				Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

				if (!(backwardCell[axis] < 0 || forwardCell[axis] >= myFluidSurface.size()[axis]))
				{
					Real gradient = 0;

					if (myFluidCellIndex(Vec2ui(backwardCell)) >= 0)
						gradient -= myPressure(Vec2ui(backwardCell));

					if (myFluidCellIndex(Vec2ui(forwardCell)) >= 0)
						gradient += myPressure(Vec2ui(forwardCell));

					localVelocity = myFluidVelocity(face, axis) - gradient / theta;
				}
			}

			velocity(face, axis) = localVelocity;
		});
	}
}

void PressureProjection::applyValid(VectorGrid<MarkedCells> &valid)
{
	valid = myValid;
}