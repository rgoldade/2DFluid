#include "PressureProjection.h"

#include "Solver.h"

void PressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSuperSampledValues(renderer, .25, 1, 2);
}

void PressureProjection::project(VectorGrid<Real>& velocity)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled liquid and solid surfaces.
	assert(velocity.size(0)[0] - 1 == mySurface.size()[0] &&
			velocity.size(0)[1] == mySurface.size()[1] &&
			velocity.size(1)[0] == mySurface.size()[0] &&
			velocity.size(1)[1] - 1 == mySurface.size()[1]);

	assert(velocity.isGridMatched(myCutCellWeights));

	int liquidDOFCount = 0;
	forEachVoxelRange(Vec2i(0), myFluidCellIndex.size(), [&](const Vec2i& cell)
	{
		if (mySurface(cell) <= 0)
		{
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					if (myCutCellWeights(face, axis) > 0)
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

					Real weight = myCutCellWeights(face, axis);

					double sign = (direction == 0) ? 1 : -1;

					if (weight > 0)
						divergence += sign * weight * velocity(face, axis);
					if (weight < 1.)
						divergence += sign * (1. - weight) * mySolidVelocity(face, axis);
				}

			solver.addToRhs(fluidIndex, divergence);

			// Build row
			double diagonal = 0.;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);
			
					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= mySurface.size()[axis])
						continue;

					Vec2i face = cellToFace(cell, axis, direction);
				
					double weight = myCutCellWeights(face, axis);

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
							Real theta = myGhostFluidWeights(face, axis);

							theta = Util::clamp(theta, MINTHETA, Real(1.));
							diagonal += weight / theta;
						}
					}
				}
			assert(diagonal > 0);
			solver.addToElement(fluidIndex, fluidIndex, diagonal);

			// Add initial guess
			if (myUseInitialGuessPressure)
			{
				assert(myInitialGuessPressure != nullptr);
				solver.addToGuess(fluidIndex, (*myInitialGuessPressure)(cell));
			}
		}
	});
	
	bool result = solver.solveIterative(1E-5);

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
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
			{
				if ((myFluidCellIndex(backwardCell) >= 0 || myFluidCellIndex(forwardCell) >= 0) &&
					myCutCellWeights(face, axis) > 0)
					myValidFaces(face, axis) = MarkedCells::FINISHED;
				else myValidFaces(face, axis) = MarkedCells::UNVISITED;
			}
			else myValidFaces(face, axis) = MarkedCells::UNVISITED;
		});
	}

	for (int axis : {0, 1})
	{
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Real localVelocity = 0;
			if (myValidFaces(face, axis) == MarkedCells::FINISHED)
			{
				Real theta = myGhostFluidWeights(face, axis);
				theta = Util::clamp(theta, MINTHETA, Real(1.));

				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
				{
					Real gradient = 0;

					if (myFluidCellIndex(backwardCell) >= 0)
						gradient -= myPressure(backwardCell);

					if (myFluidCellIndex(forwardCell) >= 0)
						gradient += myPressure(forwardCell);

					localVelocity = velocity(face, axis) - gradient / theta;
				}
			}

			velocity(face, axis) = localVelocity;
		});
	}
}