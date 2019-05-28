#include <iostream>

#include "PressureProjection.h"
#include "Solver.h"

void PressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .25, 1, 2);
}

void PressureProjection::project(const VectorGrid<Real>& ghostFluidWeights, const VectorGrid<Real>& cutCellWeights)
{
	assert(ghostFluidWeights.isGridMatched(cutCellWeights) && ghostFluidWeights.isGridMatched(myVelocity));

	// This loop is dumb in serial but it's here as a placeholder for parallel later
	int liquidDOFCount = 0;
	forEachVoxelRange(Vec2i(0), myFluidCellIndex.size(), [&](const Vec2i& cell)
	{
		if (mySurface(cell) <= 0)
		{
			for (int axis : { 0,1 })
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					if (cutCellWeights(face, axis) > 0)
					{
						myFluidCellIndex(cell) = liquidDOFCount++;
						return;
					}
				}
		}
	});

	Solver<false> solver(liquidDOFCount, liquidDOFCount * 5);

	// Build linear system
	forEachVoxelRange(Vec2i(0), myFluidCellIndex.size(), [&](const Vec2i& cell)
	{
		int fluidIndex = myFluidCellIndex(cell);
		if (fluidIndex >= 0)
		{
			// Build RHS divergence
			double divergence = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					Real weight = cutCellWeights(face, axis);

					double sign = (direction == 0) ? 1 : -1;

					if (weight > 0)
						divergence += sign * myVelocity(face, axis) * weight;
					if (weight < 1.)
						divergence += sign * mySolidVelocity(face, axis) * (1. - weight);
				}

			solver.addToRhs(fluidIndex, divergence);

			// Build row
			double diagonal = 0.;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);
			
					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= mySurface.size()[axis]) continue;

					Vec2i face = cellToFace(cell, axis, direction);
				
					double weight = cutCellWeights(face, axis);

					if (weight > 0)
					{
						// If neighbouring cell is solvable, it should have an entry in the system
						int adjacentFluidIndex = myFluidCellIndex(adjacentCell);
						if (adjacentFluidIndex >= 0)
						{
							solver.addToElement(fluidIndex, adjacentFluidIndex, -weight);
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
			solver.addToElement(fluidIndex, fluidIndex, diagonal);

			// Add initial guess
			if (myUseInitialGuess)
			{
				assert(myInitialGuess != nullptr);
				solver.addToGuess(fluidIndex, (*myInitialGuess)(cell));
			}
		}
	});
	
	bool result = solver.solveIterative(1E-1);

	if (!result)
	{
		std::cout << "Pressure projection failed to solve" << std::endl;
		return;
	}

	// Load solution into pressure grid
	forEachVoxelRange(Vec2i(0), myFluidCellIndex.size(), [&](const Vec2i& cell)
	{
		int fluidIndex = myFluidCellIndex(cell);
		if (fluidIndex >= 0)
			myPressure(cell) = solver.solution(fluidIndex);
		else
			myPressure(cell) = 0;
	});

	// Set valid faces
	for (int axis : {0, 1})
	{
		Vec2i size = myVelocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
			{
				if ((myFluidCellIndex(backwardCell) >= 0 || myFluidCellIndex(forwardCell) >= 0) &&
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
	assert(liquidWeights.isGridMatched(myVelocity));
	
	for (int axis : {0, 1})
	{
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Real localVelocity = 0;
			if (myValid(face, axis) == MarkedCells::FINISHED)
			{
				Real theta = liquidWeights(face, axis);
				theta = Util::clamp(theta, MINTHETA, Real(1.));

				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
				{
					Real gradient = 0;

					if (myFluidCellIndex(Vec2i(backwardCell)) >= 0)
						gradient -= myPressure(Vec2i(backwardCell));

					if (myFluidCellIndex(Vec2i(forwardCell)) >= 0)
						gradient += myPressure(Vec2i(forwardCell));

					localVelocity = myVelocity(face, axis) - gradient / theta;
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