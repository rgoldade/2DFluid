#include <limits>

#include "MultiMaterialPressureProjection.h"
#include "Solver.h"

void MultiMaterialPressureProjection::drawPressure(Renderer &renderer) const
{
	myPressure.drawSupersampledValues(renderer, .5, 3, 1);
}

void MultiMaterialPressureProjection::project(const std::vector<VectorGrid<Real>> &materialCutCellWeights,
												const VectorGrid<Real> &collisionCutCellWeights)
{
    assert(materialCutCellWeights.size() == mySurfaceList.size());
	for (unsigned material = 0; material < myMaterialsCount; ++material)
		assert(materialCutCellWeights[material].isGridMatched(myVelocityList[material]));

    const Vec2ui gridSize = mySurfaceList[0].size();

    myMaterialLabels = UniformGrid<int>(mySolidSurface.size(), UNSOLVED);
    mySolverIndex = UniformGrid<int>(mySolidSurface.size(), UNSOLVED);

    // Determine which material a voxel falls into
    int liquidDOFCount = 0;
    forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui &cell)
    {
		// Check if the cell has as solid cut-cell weight that is less than unity.
		bool isLiquidCell = false;

		for (auto axis : { 0,1 })
			for (auto direction : { 0,1 })
			{
				Vec2ui face = cellToFace(cell, axis, direction);

				if (collisionCutCellWeights(face, axis) < 1.)
				{
					isLiquidCell = true;
					break;
				}
			}

		if (isLiquidCell)
		{
			Real minDistance = std::numeric_limits<Real>::max();
			int minMaterial = -1;

			for (unsigned material = 0; material < myMaterialsCount; ++material)
			{
				bool validMaterial = false;

				// Check if the cell has a non-zero material fraction on one
				// of the faces.
				for (auto axis : { 0,1 })
					for (auto direction : { 0,1 })
					{
						Vec2ui face = cellToFace(cell, axis, direction);

						if (materialCutCellWeights[material](face, axis) > 0.)
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

    forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
    {
		int row = mySolverIndex(cell);

		if (row >= 0)
		{
			// Build RHS divergence contribution per material.
			double divergence = 0;

			for (auto axis : { 0,1 })
				for (auto direction : { 0,1 })
				{
					Vec2ui face = cellToFace(cell, axis, direction);

					for (unsigned material = 0; material < myMaterialsCount; ++material)
					{
						double weight = materialCutCellWeights[material](face, axis);

						if (weight > 0.)
						{
							// We have a negative sign in the forward direction because
							// we're actually solving with a -1 leading ceofficient.
							double sign = (direction == 0) ? 1 : -1;
							divergence += sign * weight * myVelocityList[material](face, axis);
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
			for (auto axis : { 0,1 })
				for (auto direction : { 0,1 })
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

					// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= gridSize[axis]) continue;

					Vec2ui face = cellToFace(cell, axis, direction);

					int adjacentRow = mySolverIndex(Vec2ui(adjacentCell));

					double collisionWeight = collisionCutCellWeights(face, axis);

					if (collisionWeight == 1.)
						continue;

					assert(adjacentRow >= 0);

					// The cut-cell weight for a pressure gradient is only the inverse of
					// the solid weights. This is due to the contribution from each
					// material across the face using the same pressure gradient term.

					double weight = 1. - collisionWeight;

					int adjacentMaterial = myMaterialLabels(Vec2ui(adjacentCell));

					assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialsCount);

					double density;
					if (adjacentMaterial != material)
					{
						double adjacentPhi = mySurfaceList[adjacentMaterial](Vec2ui(adjacentCell));

						double theta;
						if (direction == 0)
							theta = fabs(adjacentPhi) / (fabs(adjacentPhi) + fabs(phi));
						else
							theta = fabs(phi) / (fabs(phi) + fabs(adjacentPhi));

						theta = Util::clamp(theta, MINTHETA, 1.);

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
    forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
    {
		int row = mySolverIndex(cell);
		if (row >= 0)
			myPressure(cell) = solver.solution(row);
    });

    // Set valid faces
	for (auto axis : { 0,1 })
    {
		forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& face)
		{
			Vec2i backward_cell = faceToCell(Vec2i(face), axis, 0);
			Vec2i forward_cell = faceToCell(Vec2i(face), axis, 1);

			if (backward_cell[axis] < 0 || forward_cell[axis] >= gridSize[axis])
				return;

			if ((collisionCutCellWeights(face, axis) < 1.) &&
				(mySolverIndex(Vec2ui(backward_cell)) >= 0 || mySolverIndex(Vec2ui(forward_cell)) >= 0))
			{
				assert(mySolverIndex(Vec2ui(backward_cell)) >= 0 && mySolverIndex(Vec2ui(forward_cell)) >= 0);
				myValid(face, axis) = 1;
			}
		});
    }
}

void MultiMaterialPressureProjection::applySolution(std::vector<VectorGrid<Real>> &velocity,
													const std::vector<VectorGrid<Real>> &materialCutCellWeights) const
{
    for (unsigned material = 0; material < myMaterialsCount; ++material)
		assert(myVelocityList[material].isGridMatched(velocity[material]));

    for (unsigned axis = 0; axis < 2; ++axis)
    {
		Vec2ui velSize = velocity[0].size(axis);

		forEachVoxelRange(Vec2ui(0), velSize, [&](const Vec2ui& face)
		{
			if (myValid(face, axis) > 0)
			{
				Real theta = 0;
				Real phi = 0;
				Real sampleDensity[2];
				int materials[2];

				Real gradient = 0;
				for (auto direction : { 0,1 })
				{
					Vec2i cell = faceToCell(Vec2i(face), axis, direction);
					int material = myMaterialLabels(Vec2ui(cell));
					materials[direction] = material;
					assert(mySolverIndex(Vec2ui(cell)) >= 0);

					phi += fabs(mySurfaceList[material](Vec2ui(cell)));

					if (direction == 0) theta = phi;

					sampleDensity[direction] = myDensityList[material];

					Real sign = (direction == 0) ? -1 : 1;
					gradient += sign * myPressure(Vec2ui(cell));
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
					theta = Util::clamp(theta, MINTHETA, 1.);

					density = theta * sampleDensity[0] + (1. - theta) * sampleDensity[1];
				}

				gradient /= density;

				// Update every material's velocity with a non-zero face weight.
				for (unsigned material = 0; material < myMaterialsCount; ++material)
				{
					if (materialCutCellWeights[material](face, axis) > 0.)
						velocity[material](face, axis) -= gradient;
					else
						velocity[material](face, axis) = 0;
				}
			}
			else
			{
				for (unsigned material = 0; material < myMaterialsCount; ++material)
					velocity[material](face, axis) = 0;
			}
		});
    }
}