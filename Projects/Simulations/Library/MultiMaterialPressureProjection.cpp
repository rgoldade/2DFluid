#include <limits>

#include "MultiMaterialPressureProjection.h"
#include "Solver.h"

void MultiMaterialPressureProjection::drawPressure(Renderer &renderer) const
{
	myPressure.drawSupersampledValues(renderer, .5, 3, 1);
}

void MultiMaterialPressureProjection::project(std::vector<VectorGrid<Real>> &velocityList)
{
	assert(velocityList.size() == mySurfaceList.size());
	for (int material = 1; material < myMaterialsCount; ++material)
		assert(velocityList[material - 1].isGridMatched(velocityList[material]));

	for (int material = 0; material < myMaterialsCount; ++material)
		assert(velocityList[material].isGridMatched(myMaterialCutCellWeightsList[material]));

	// Since every surface and every velocity field is matched, we only need to compare
	// on pair of fields to make sure the two sets are matched.
	assert(velocityList[0].size(0)[0] - 1 == mySurfaceList[0].size()[0] &&
			velocityList[0].size(0)[1] == mySurfaceList[0].size()[1] &&
			velocityList[0].size(1)[0] == mySurfaceList[0].size()[0] &&
			velocityList[0].size(1)[1] - 1 == mySurfaceList[0].size()[1]);

	Vec2i gridSize = mySurfaceList[0].size();

    myMaterialLabels = UniformGrid<int>(mySolidSurface.size(), PressureCellLabels::UNSOLVED_CELL);
    mySolverIndex = UniformGrid<int>(mySolidSurface.size(), PressureCellLabels::UNSOLVED_CELL);

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

						if (myMaterialCutCellWeightsList[material](face, axis) > 0.)
						{
							validMaterial = true;
							break;
						}
					}

				// Find the material with the lowest SDF value. This could be outside
				// of a material if the cell is partially inside a solid.
				if (validMaterial)
				{
					Real sdf = mySurfaceList[material](cell);

					if (sdf < minDistance)
					{
						minMaterial = material;
						minDistance = sdf;
					}
				}
			}

			assert(minMaterial >= 0);

			if (minDistance > 0) assert(mySolidSurface(cell) <= 0);

			mySolverIndex(cell) = liquidDOFCount++;
			myMaterialLabels(cell) = minMaterial;

		}
		else
			assert(mySolidSurface(cell) <= 0);
    });

    Solver<true> solver(liquidDOFCount, liquidDOFCount * 5);

    forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
    {
		int row = mySolverIndex(cell);

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
						double weight = myMaterialCutCellWeightsList[material](face, axis);

						if (weight > 0.)
						{
							// We have a negative sign in the forward direction because
							// we're actually solving with a -1 leading ceofficient.
							double sign = (direction == 0) ? 1 : -1;
							divergence += sign * weight * velocityList[material](face, axis);
						}
					}
				}

			assert(std::isfinite(divergence));
			solver.addToRhs(row, divergence);

			// Build A matrix row
			double diagonal = 0;

			int material = myMaterialLabels(cell);
			assert(material >= 0 && material < myMaterialsCount);

			double phi = mySurfaceList[material](cell);
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= gridSize[axis]) continue;

					Vec2i face = cellToFace(cell, axis, direction);

					int adjacentRow = mySolverIndex(adjacentCell);

					double collisionWeight = mySolidCutCellWeights(face, axis);

					if (collisionWeight == 1.)
						continue;

					assert(adjacentRow >= 0);

					// The cut-cell weight for a pressure gradient is only the inverse of
					// the solid weights. This is due to the contribution from each
					// material across the face using the same pressure gradient term.

					double weight = 1. - collisionWeight;

					int adjacentMaterial = myMaterialLabels(adjacentCell);

					assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialsCount);

					double density;
					if (adjacentMaterial != material)
					{
						double adjacentPhi = mySurfaceList[adjacentMaterial](adjacentCell);

						double theta;
						if (direction == 0)
							theta = fabs(adjacentPhi) / (fabs(adjacentPhi) + fabs(phi));
						else
							theta = fabs(phi) / (fabs(phi) + fabs(adjacentPhi));

						theta = Util::clamp(theta, MINTHETA, Real(1.));

						// Build interpolated density
						if (direction == 0)
							density = theta * myDensityList[adjacentMaterial] + (1. - theta) * myDensityList[material];
						else
							density = (1. - theta) * myDensityList[adjacentMaterial] + theta * myDensityList[material];

						assert(std::isfinite(density));
						assert(density > 0);
				
					}
					else density = myDensityList[material];

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

    bool result = solver.solveDirect();

    if (!result)
    {
		std::cout << "Pressure projection failed to solve" << std::endl;
		return;
    }

    // Load solution into pressure grid
    forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
    {
		int row = mySolverIndex(cell);
		if (row >= 0)
			myPressure(cell) = solver.solution(row);
    });

    // Set valid faces
	for (int axis : {0, 1})
    {
		forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
			Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] >= gridSize[axis])
				return;

			if ((mySolidCutCellWeights(face, axis) < 1.) &&
				(mySolverIndex(backwardCell) >= 0 || mySolverIndex(forwardCell) >= 0))
			{
				assert(mySolverIndex(backwardCell) >= 0 && mySolverIndex(forwardCell) >= 0);
				myValidFaces(face, axis) = MarkedCells::FINISHED;
			}
		});
    }

	for (int axis : {0, 1})
    {
		Vec2i velSize = velocityList[0].size(axis);

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
					int material = myMaterialLabels(cell);
					
					materials[direction] = material;
					assert(mySolverIndex(cell) >= 0);

					phi += fabs(mySurfaceList[material](cell));

					if (direction == 0) theta = phi;

					sampleDensity[direction] = myDensityList[material];

					Real sign = (direction == 0) ? -1 : 1;
					gradient += sign * myPressure(cell);
				}

				Real density;
				if (materials[0] == materials[1])
				{
					assert(sampleDensity[0] == sampleDensity[1]);
					density = sampleDensity[0];
					theta = 1;
				}
				else
				{
					theta /= phi;
					theta = Util::clamp(theta, MINTHETA, Real(1.));

					density = theta * sampleDensity[0] + (1. - theta) * sampleDensity[1];
				}

				gradient /= density;

				// Update every material's velocity with a non-zero face weight.
				for (int material = 0; material < myMaterialsCount; ++material)
				{
					if (myMaterialCutCellWeightsList[material](face, axis) > 0.)
					{
						myValidMaterialFacesList[material](face, axis) = MarkedCells::FINISHED;
						velocityList[material](face, axis) -= gradient;
					}
					else
						velocityList[material](face, axis) = 0;
				}
			}
			else
			{
				for (int material = 0; material < myMaterialsCount; ++material)
					velocityList[material](face, axis) = 0;
			}
		});
    }
}