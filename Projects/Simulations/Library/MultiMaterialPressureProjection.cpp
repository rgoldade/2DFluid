#include <limits>

#include "MultiMaterialPressureProjection.h"
#include "Solver.h"

void MultiMaterialPressureProjection::drawPressure(Renderer &renderer) const
{
	myPressure.drawSuperSampledValues(renderer, .5, 3, 1);
}

void MultiMaterialPressureProjection::project(std::vector<VectorGrid<Real>> &fluidVelocities)
{
	assert(fluidVelocities.size() == myFluidSurfaces.size());
	for (int material = 1; material < myMaterialsCount; ++material)
		assert(fluidVelocities[material - 1].isGridMatched(fluidVelocities[material]));

	for (int material = 0; material < myMaterialsCount; ++material)
		assert(fluidVelocities[material].isGridMatched(myMaterialCutCellWeights[material]));

	// Since every surface and every velocity field is matched, we only need to compare
	// on pair of fields to make sure the two sets are matched.
	assert(fluidVelocities[0].size(0)[0] - 1 == myFluidSurfaces[0].size()[0] &&
			fluidVelocities[0].size(0)[1] == myFluidSurfaces[0].size()[1] &&
			fluidVelocities[0].size(1)[0] == myFluidSurfaces[0].size()[0] &&
			fluidVelocities[0].size(1)[1] - 1 == myFluidSurfaces[0].size()[1]);

	Vec2i gridSize = myFluidSurfaces[0].size();

	UniformGrid<int> materialCellLabels = UniformGrid<int>(mySolidSurface.size(), PressureCellLabels::UNSOLVED_CELL);
	UniformGrid<int> solvableCellIndices = UniformGrid<int>(mySolidSurface.size(), PressureCellLabels::UNSOLVED_CELL);

    // Determine which material a voxel falls into
    int liquidDOFCount = 0;
    forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i &cell)
    {
		// Check if the cell has as solid cut-cell weight that is less than unity.
		bool isLiquidCell = false;

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i face = cellToFace(cell, axis, direction);

				if (mySolidCutCellWeights(face, axis) < 1.)
				{
					isLiquidCell = true;
					break;
				}
			}

		if (isLiquidCell)
		{
			Real minDistance = std::numeric_limits<Real>::max();
			int minMaterial = -1;

			for (int material = 0; material < myMaterialsCount; ++material)
			{
				bool validMaterial = false;

				// Check if the cell has a non-zero material fraction on one
				// of the faces.
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i face = cellToFace(cell, axis, direction);

						if (myMaterialCutCellWeights[material](face, axis) > 0.)
						{
							validMaterial = true;
							break;
						}
					}

				// Find the material with the lowest SDF value. This could be outside
				// of a material if the cell is partially inside a solid.
				if (validMaterial)
				{
					Real sdf = myFluidSurfaces[material](cell);

					if (sdf < minDistance)
					{
						minMaterial = material;
						minDistance = sdf;
					}
				}
			}

			assert(minMaterial >= 0);

			if (minDistance > 0) assert(mySolidSurface(cell) <= 0);

			solvableCellIndices(cell) = liquidDOFCount++;
			materialCellLabels(cell) = minMaterial;

		}
		else
			assert(mySolidSurface(cell) <= 0);
    });

    Solver<true> solver(liquidDOFCount, liquidDOFCount * 5);

    forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
    {
		int row = solvableCellIndices(cell);

		if (row >= 0)
		{
			// Build RHS divergence contribution per material.
			double divergence = 0;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					for (int material = 0; material < myMaterialsCount; ++material)
					{
						double weight = myMaterialCutCellWeights[material](face, axis);

						if (weight > 0.)
						{
							// We have a negative sign in the forward direction because
							// we're actually solving with a -1 leading ceofficient.
							double sign = (direction == 0) ? 1 : -1;
							divergence += sign * weight * fluidVelocities[material](face, axis);
						}
					}
				}

			assert(std::isfinite(divergence));
			solver.addToRhs(row, divergence);

			// Build A matrix row
			double diagonal = 0;

			int material = materialCellLabels(cell);
			assert(material >= 0 && material < myMaterialsCount);

			double phi = myFluidSurfaces[material](cell);
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= gridSize[axis]) continue;

					Vec2i face = cellToFace(cell, axis, direction);

					int adjacentRow = solvableCellIndices(adjacentCell);

					double collisionWeight = mySolidCutCellWeights(face, axis);

					if (collisionWeight == 1.)
						continue;

					assert(adjacentRow >= 0);

					// The cut-cell weight for a pressure gradient is only the inverse of
					// the solid weights. This is due to the contribution from each
					// material across the face using the same pressure gradient term.

					double weight = 1. - collisionWeight;

					int adjacentMaterial = materialCellLabels(adjacentCell);

					assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialsCount);

					double density;
					if (adjacentMaterial != material)
					{
						double adjacentPhi = myFluidSurfaces[adjacentMaterial](adjacentCell);

						double theta;
						if (direction == 0)
							theta = fabs(adjacentPhi) / (fabs(adjacentPhi) + fabs(phi));
						else
							theta = fabs(phi) / (fabs(phi) + fabs(adjacentPhi));

						theta = Util::clamp(theta, 0, 1);

						// Build interpolated density
						if (direction == 0)
							density = theta * myFluidDensities[adjacentMaterial] + (1. - theta) * myFluidDensities[material];
						else
							density = (1. - theta) * myFluidDensities[adjacentMaterial] + theta * myFluidDensities[material];

						assert(std::isfinite(density));
						assert(density > 0);
				
					}
					else density = myFluidDensities[material];

					weight /= density;

					solver.addToElement(row, adjacentRow, -weight);
					diagonal += weight;
				}
			solver.addToElement(row, row, diagonal);
		}
    });

	// Project out null space
	double avgRHS = 0;
	for (int row = 0; row < liquidDOFCount; ++row)
		avgRHS += solver.rhs(row);

	avgRHS /= double(liquidDOFCount);

	for (int row = 0; row < liquidDOFCount; ++row)
		solver.addToRhs(row, -avgRHS);

	assert(solver.isSymmetric());
	assert(solver.isFinite());

    bool result = solver.solveIterative();

    if (!result)
    {
		std::cout << "Pressure projection failed to solve" << std::endl;
		return;
    }

    // Load solution into pressure grid
    forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
    {
		int row = solvableCellIndices(cell);
		if (row >= 0)
			myPressure(cell) = solver.solution(row);
    });

    // Set valid faces
	for (int axis : {0, 1})
    {
		forEachVoxelRange(Vec2i(0), fluidVelocities[0].size(axis), [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
			Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] >= gridSize[axis])
				return;

			if ((mySolidCutCellWeights(face, axis) < 1.) &&
				(solvableCellIndices(backwardCell) >= 0 || solvableCellIndices(forwardCell) >= 0))
			{
				assert(solvableCellIndices(backwardCell) >= 0 && solvableCellIndices(forwardCell) >= 0);
				myValidFaces(face, axis) = MarkedCells::FINISHED;
			}
		});
    }

	for (int axis : {0, 1})
    {
		Vec2i velSize = fluidVelocities[0].size(axis);

		forEachVoxelRange(Vec2i(0), velSize, [&](const Vec2i& face)
		{
			if (myValidFaces(face, axis) == MarkedCells::FINISHED)
			{
				Real theta = 0;
				Real phi = 0;
				Real sampleDensity[2];
				int materials[2];

				Real gradient = 0;
				for (int direction : {0, 1})
				{
					Vec2i cell = faceToCell(face, axis, direction);
					int material = materialCellLabels(cell);
					
					materials[direction] = material;
					assert(solvableCellIndices(cell) >= 0);

					phi += fabs(myFluidSurfaces[material](cell));

					if (direction == 0) theta = phi;

					sampleDensity[direction] = myFluidDensities[material];

					Real sign = (direction == 0) ? -1 : 1;
					gradient += sign * myPressure(cell);
				}

				Real density;
				if (materials[0] == materials[1])
				{
					assert(sampleDensity[0] == sampleDensity[1]);
					density = sampleDensity[0];
				}
				else
				{
					theta /= phi;
					theta = Util::clamp(theta, 0, 1);
					density = theta * sampleDensity[0] + (1. - theta) * sampleDensity[1];
				}

				gradient /= density;

				// Update every material's velocity with a non-zero face weight.
				for (int material = 0; material < myMaterialsCount; ++material)
				{
					if (myMaterialCutCellWeights[material](face, axis) > 0.)
					{
						myValidMaterialFaces[material](face, axis) = MarkedCells::FINISHED;
						fluidVelocities[material](face, axis) -= gradient;
					}
					else
						fluidVelocities[material](face, axis) = 0;
				}
			}
			else
			{
				for (int material = 0; material < myMaterialsCount; ++material)
					fluidVelocities[material](face, axis) = 0;
			}
		});
    }
}