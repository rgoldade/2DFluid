#include "MultiMaterialPressureProjection.h"

#include <limits>

#include <iostream>

#include "tbb/tbb.h"

#include "ConjugateGradientSolver.h"
#include "GeometricMultigridPoissonSolver.h"

namespace FluidSim2D::RegularGridSim
{

MultiMaterialPressureProjection::MultiMaterialPressureProjection(const std::vector<LevelSet>& surfaces,
																	const std::vector<float>& densities,
																	const LevelSet& solidSurface)
	: myFluidSurfaces(surfaces)
	, myFluidDensities(densities)
	, mySolidSurface(solidSurface)
	, myMaterialCount(surfaces.size())
{
	assert(myFluidSurfaces.size() == myFluidDensities.size());

	for (int material = 1; material < myMaterialCount; ++material)
		assert(myFluidSurfaces[material - 1].isGridMatched(myFluidSurfaces[material]));

	assert(myFluidSurfaces[0].isGridMatched(mySolidSurface));

	myPressure = ScalarGrid<float>(mySolidSurface.xform(), mySolidSurface.size(), 0);

	myValidMaterialFaces.resize(myMaterialCount);
	for (int material = 0; material < myMaterialCount; ++material)
		myValidMaterialFaces[material] = VectorGrid<VisitedCellLabels>(mySolidSurface.xform(), mySolidSurface.size(), VisitedCellLabels::UNVISITED_CELL, VectorGridSettings::SampleType::STAGGERED);

	// Compute cut-cell weights
	mySolidCutCellWeights = computeCutCellWeights(solidSurface);

	myMaterialCutCellWeights.resize(myMaterialCount);

	for (int material = 0; material < myMaterialCount; ++material)
		myMaterialCutCellWeights[material] = computeCutCellWeights(surfaces[material]);

	// Now normalize the weights, removing the solid boundary contribution first.
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, mySolidCutCellWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = mySolidCutCellWeights.grid(axis).unflatten(faceIndex);

				float weight = 1;
				weight -= mySolidCutCellWeights(face, axis);
				weight = clamp(weight, float(0.), weight);

				if (weight > 0)
				{
					float accumulatedWeight = 0;
					for (int material = 0; material < myMaterialCount; ++material)
						accumulatedWeight += myMaterialCutCellWeights[material](face, axis);

					if (accumulatedWeight > 0)
					{
						weight /= accumulatedWeight;

						for (int material = 0; material < myMaterialCount; ++material)
							myMaterialCutCellWeights[material](face, axis) *= weight;
					}
				}
				else
				{
					for (int material = 0; material < myMaterialCount; ++material)
						myMaterialCutCellWeights[material](face, axis) = 0;
				}

				// Debug check
				float totalWeight = mySolidCutCellWeights(face, axis);

				for (int material = 0; material < myMaterialCount; ++material)
					totalWeight += myMaterialCutCellWeights[material](face, axis);

				if (totalWeight == 0)
				{
					// If there is a zero total weight it is likely due to a fluid-fluid boundary
					// falling exactly across a grid face. There should never be a zero weight
					// along a fluid-solid boundary.

					std::vector<int> faceAlignedSurfaces;

					int otherAxis = (axis + 1) % 2;

					Vec2f offset(0); offset[otherAxis] = .5;

					for (int material = 0; material < myMaterialCount; ++material)
					{
						Vec2f backwardNodePoint = myMaterialCutCellWeights[material].indexToWorld(Vec2f(face) - offset, axis);
						Vec2f forwardNodePoint = myMaterialCutCellWeights[material].indexToWorld(Vec2f(face) + offset, axis);

						float weight = lengthFraction(surfaces[material].biLerp(backwardNodePoint), surfaces[material].biLerp(forwardNodePoint));

						if (weight == 0)
							faceAlignedSurfaces.push_back(material);
					}

					if (!(faceAlignedSurfaces.size() > 1))
					{
						std::cout << "Zero weight problems!!" << std::endl;
						exit(-1);
					}
					assert(faceAlignedSurfaces.size() > 1);

					myMaterialCutCellWeights[faceAlignedSurfaces[0]](face, axis) = 1.;
				}
			}
		});
	}
}

void MultiMaterialPressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .5, 3, 1);
}

void MultiMaterialPressureProjection::copyToPreconditionerGrids(std::vector<UniformGrid<SolveReal>> &mgSourceGrid,
																UniformGrid<SolveReal> &smootherSourceGrid,
																const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
																const UniformGrid<int>& materialCellLabels,
																const UniformGrid<int>& solvableCellIndices,
																const Vector &sourceVector,
																const std::vector<Vec2i> &mgExpandedOffset) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = materialCellLabels.unflatten(cellIndex);
				int material = materialCellLabels(cell);

				if (material >= 0)
				{
					Vec2i expandedCell = cell + mgExpandedOffset[material];

					assert(mgDomainCellLabels[material](expandedCell) == MGCellLabels::INTERIOR_CELL ||
						mgDomainCellLabels[material](expandedCell) == MGCellLabels::BOUNDARY_CELL);

					int index = solvableCellIndices(cell);
					assert(index >= 0);

					smootherSourceGrid(cell) = sourceVector(index);

					mgSourceGrid[material](expandedCell) = myFluidDensities[material] * sourceVector(index);
				}
				else
				{
					assert(solvableCellIndices(cell) == UNSOLVED_CELL);
					assert(materialCellLabels(cell) == UNSOLVED_CELL);

					for (int material = 0; material < myMaterialCount; ++material)
					{
						Vec2i expandedCell = cell + mgExpandedOffset[material];
						assert(mgDomainCellLabels[material](expandedCell) == MGCellLabels::EXTERIOR_CELL);
					}
				}
			}
		});
}

void MultiMaterialPressureProjection::applyBoundarySmoothing(std::vector<SolveReal>& tempDestinationVector,
	const std::vector<Vec2i>& boundarySmootherCells,
	const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
	const UniformGrid<int>& materialCellLabels,
	const UniformGrid<SolveReal>& smootherDestinationGrid,
	const UniformGrid<SolveReal>& smootherSourceGrid,
	const std::vector<Vec2i>& mgExpandedOffset) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, boundarySmootherCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundarySmootherCells[cellIndex];

				SolveReal laplacian = 0;
				SolveReal diagonal = 0;

				int material = materialCellLabels(cell);
				assert(material >= 0);

				assert(mgDomainCellLabels[material](cell + mgExpandedOffset[material]) == MGCellLabels::INTERIOR_CELL ||
					mgDomainCellLabels[material](cell + mgExpandedOffset[material]) == MGCellLabels::BOUNDARY_CELL);

				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i face = cellToFace(cell, axis, direction);

						SolveReal solidWeight = mySolidCutCellWeights(face, axis);

						if (solidWeight == 1.)
							continue;

						SolveReal weight = 1. - solidWeight;

						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						int adjacentMaterial = materialCellLabels(adjacentCell);

						assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialCount);

						SolveReal density;
						if (adjacentMaterial != material)
						{
							SolveReal phi = myFluidSurfaces[material](cell);
							SolveReal adjacentPhi = myFluidSurfaces[adjacentMaterial](adjacentCell);

							SolveReal theta;
							if (direction == 0)
								theta = std::fabs(adjacentPhi) / (std::fabs(adjacentPhi) + std::fabs(phi));
							else
								theta = std::fabs(phi) / (std::fabs(phi) + std::fabs(adjacentPhi));

							theta = clamp(theta, SolveReal(0), SolveReal(1));

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

						laplacian -= weight * smootherDestinationGrid(adjacentCell);
						diagonal += weight;
					}

				laplacian += diagonal * smootherDestinationGrid(cell);

				SolveReal residual = smootherSourceGrid(cell) - laplacian;
				residual /= diagonal;

				constexpr SolveReal dampedWeight = 2. / 3.;
				tempDestinationVector[cellIndex] = smootherDestinationGrid(cell) + dampedWeight * residual;
			}
		});
}

void MultiMaterialPressureProjection::updateDestinationGrid(UniformGrid<SolveReal>& smootherDestinationGrid,
	const UniformGrid<int>& materialCellLabels,
	const std::vector<Vec2i>& boundarySmootherCells,
	const std::vector<SolveReal>& tempDestinationVector) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, boundarySmootherCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundarySmootherCells[cellIndex];
				assert(materialCellLabels(cell) >= 0);

				smootherDestinationGrid(cell) = tempDestinationVector[cellIndex];
			}
		});
}

void MultiMaterialPressureProjection::applyDirichletToMG(std::vector<UniformGrid<SolveReal>>& mgSourceGrid,
	std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
	const UniformGrid<SolveReal>& smootherDestinationGrid,
	const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
	const UniformGrid<int>& materialCellLabels,
	const UniformGrid<int>& solvableCellIndices,
	const Vector& sourceVector,
	const std::vector<Vec2i>& boundarySmootherCells,
	const std::vector<Vec2i>& mgExpandedOffset) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, boundarySmootherCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundarySmootherCells[cellIndex];

				int material = materialCellLabels(cell);
				assert(material >= 0);

				Vec2i expandedCell = cell + mgExpandedOffset[material];

				int index = solvableCellIndices(cell);
				assert(index >= 0);

				mgDestinationGrid[material](expandedCell) = smootherDestinationGrid(cell);

				if (mgDomainCellLabels[material](expandedCell) == MGCellLabels::BOUNDARY_CELL)
				{
					SolveReal laplacian = 0;

					bool hasMultiMaterialBoundary = false;
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal solidWeight = mySolidCutCellWeights(face, axis);

							if (solidWeight == 1.)
								continue;

							SolveReal weight = 1. - solidWeight;

							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							int adjacentMaterial = materialCellLabels(adjacentCell);

							assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialCount);

							if (adjacentMaterial != material)
							{
								SolveReal phi = myFluidSurfaces[material](cell);
								SolveReal adjacentPhi = myFluidSurfaces[adjacentMaterial](adjacentCell);

								SolveReal theta;
								if (direction == 0)
									theta = std::fabs(adjacentPhi) / (std::fabs(adjacentPhi) + std::fabs(phi));
								else
									theta = std::fabs(phi) / (std::fabs(phi) + std::fabs(adjacentPhi));

								theta = clamp(theta, SolveReal(0), SolveReal(1));

								// Build interpolated density
								SolveReal density;
								if (direction == 0)
									density = theta * myFluidDensities[adjacentMaterial] + (1. - theta) * myFluidDensities[material];
								else
									density = (1. - theta) * myFluidDensities[adjacentMaterial] + theta * myFluidDensities[material];

								assert(std::isfinite(density));
								assert(density > 0);

								hasMultiMaterialBoundary = true;

								weight *= myFluidDensities[material] / density;

								laplacian -= weight * smootherDestinationGrid(adjacentCell);
							}
						}

					mgSourceGrid[material](expandedCell) = myFluidDensities[material] * sourceVector(index) - laplacian;
				}
			}
		});
}

void MultiMaterialPressureProjection::updateSmootherGrid(UniformGrid<SolveReal>& smootherDestinationGrid,
	const std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
	const UniformGrid<int>& materialCellLabels,
	const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
	const std::vector<Vec2i>& mgExpandedOffset) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = materialCellLabels.unflatten(cellIndex);

				int material = materialCellLabels(cell);

				if (material >= 0)
				{
					Vec2i expandedCell = cell + mgExpandedOffset[material];

					assert(mgDomainCellLabels[material](expandedCell) == MGCellLabels::INTERIOR_CELL ||
						mgDomainCellLabels[material](expandedCell) == MGCellLabels::BOUNDARY_CELL);

					smootherDestinationGrid(cell) = mgDestinationGrid[material](expandedCell);
				}

			}
		});
}

void MultiMaterialPressureProjection::copyMGSolutionToVector(Vector& destinationVector,
	const UniformGrid<int>& materialCellLabels,
	const UniformGrid<int>& solvableCellIndices,
	const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
	const std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
	const std::vector<Vec2i>& mgExpandedOffset) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = materialCellLabels.unflatten(cellIndex);
				int material = materialCellLabels(cell);

				if (material >= 0)
				{
					Vec2i expandedCell = cell + mgExpandedOffset[material];

					assert(mgDomainCellLabels[material](expandedCell) == MGCellLabels::INTERIOR_CELL ||
						mgDomainCellLabels[material](expandedCell) == MGCellLabels::BOUNDARY_CELL);

					int index = solvableCellIndices(cell);
					assert(index >= 0);

					destinationVector(index) = mgDestinationGrid[material](expandedCell);
				}
			}
		});
}

void MultiMaterialPressureProjection::copyBoundarySolutionToVector(Vector& destinationVector,
	const UniformGrid<int>& materialCellLabels,
	const UniformGrid<int>& solvableCellIndices,
	const UniformGrid<SolveReal>& smootherDestinationGrid,
	const std::vector<Vec2i>& boundarySmootherCells) const
{
	tbb::parallel_for(tbb::blocked_range<int>(0, boundarySmootherCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundarySmootherCells[cellIndex];

				int material = materialCellLabels(cell);
				assert(material >= 0);

				int index = solvableCellIndices(cell);

				destinationVector(index) = smootherDestinationGrid(cell);
			}
		});
}

void MultiMaterialPressureProjection::project(std::vector<VectorGrid<float>>& fluidVelocities)
{
	assert(fluidVelocities.size() == myFluidSurfaces.size());

	for (int material = 1; material < myMaterialCount; ++material)
		assert(fluidVelocities[material - 1].isGridMatched(fluidVelocities[material]));

	for (int material = 0; material < myMaterialCount; ++material)
		assert(fluidVelocities[material].isGridMatched(myMaterialCutCellWeights[material]));

	// Since every surface and every velocity field is matched, we only need to compare
	// on pair of fields to make sure the two sets are matched.
	assert(fluidVelocities[0].size(0)[0] - 1 == myFluidSurfaces[0].size()[0] &&
			fluidVelocities[0].size(0)[1] == myFluidSurfaces[0].size()[1] &&
			fluidVelocities[0].size(1)[0] == myFluidSurfaces[0].size()[0] &&
			fluidVelocities[0].size(1)[1] - 1 == myFluidSurfaces[0].size()[1]);

	UniformGrid<int> materialCellLabels = UniformGrid<int>(mySolidSurface.size(), UNSOLVED_CELL);
	UniformGrid<int> solvableCellIndices = UniformGrid<int>(mySolidSurface.size(), UNSOLVED_CELL);

	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = materialCellLabels.unflatten(cellIndex);

			// Check if the cell has as solid cut-cell weight that is less than unity.
			bool isFluidCell = false;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					if (mySolidCutCellWeights(face, axis) < 1.)
					{
						isFluidCell = true;
						break;
					}
				}

			if (isFluidCell)
			{
				float minDistance = std::numeric_limits<float>::max();
				int minMaterial = -1;

				for (int material = 0; material < myMaterialCount; ++material)
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
						float sdf = myFluidSurfaces[material](cell);

						if (sdf < minDistance)
						{
							minMaterial = material;
							minDistance = sdf;
						}
					}
				}

				assert(minMaterial >= 0);

				if (minDistance > 0) assert(mySolidSurface(cell) <= 0);

				materialCellLabels(cell) = minMaterial;
			}
			else
				assert(mySolidSurface(cell) <= 0);
		}
	});

	int solvableCellCount = 0;
	forEachVoxelRange(Vec2i(0), materialCellLabels.size(), [&](const Vec2i& cell)
	{
		if (materialCellLabels(cell) >= 0)
			solvableCellIndices(cell) = solvableCellCount++;
	});

	Vector rhsVector = Vector::Zero(solvableCellCount);

	Eigen::SparseMatrix<SolveReal, Eigen::RowMajor> sparseMatrix(solvableCellCount, solvableCellCount);
	Eigen::SparseMatrix<SolveReal, Eigen::RowMajor> diagonalSparseMatrix(solvableCellCount, solvableCellCount);

	{
		tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<SolveReal>>> parallelSparseMatrixElements;
		tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<SolveReal>>> parallelDiagonalSparseMatrixElements;

		tbb::parallel_for(tbb::blocked_range<int>(0, solvableCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			auto& localSparseMatrixElements = parallelSparseMatrixElements.local();
			auto& localDiagonalMatrixElements = parallelDiagonalSparseMatrixElements.local();

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = solvableCellIndices.unflatten(cellIndex);

				int liquidIndex = solvableCellIndices(cell);

				if (liquidIndex >= 0)
				{
					// Build RHS divergence contribution per material.
					SolveReal divergence = 0;

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);

							if (mySolidCutCellWeights(face, axis) < 1)
							{
								SolveReal sign = (direction == 0) ? 1 : -1;

								for (int material = 0; material < myMaterialCount; ++material)
								{
									SolveReal weight = myMaterialCutCellWeights[material](face, axis);

									if (weight > 0.)
									{
										// We have a negative sign in the forward direction because
										// we're actually solving with a -1 leading ceofficient.

										divergence += sign * weight * fluidVelocities[material](face, axis);
									}
								}
							}
						}

					assert(std::isfinite(divergence));
					
					rhsVector(liquidIndex) = divergence;

					SolveReal diagonal = 0;

					int material = materialCellLabels(cell);
					assert(material >= 0 && material < myMaterialCount);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
							if (adjacentCell[axis] < 0 || adjacentCell[axis] >= materialCellLabels.size()[axis])
								continue;

							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal solidWeight = mySolidCutCellWeights(face, axis);

							if (solidWeight == 1.)
								continue;

							int adjacentIndex = solvableCellIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							// The cut-cell weight for a pressure gradient is only the inverse of
							// the solid weights. This is due to the contribution from each
							// material across the face using the same pressure gradient term.

							SolveReal weight = 1. - solidWeight;

							int adjacentMaterial = materialCellLabels(adjacentCell);

							assert(adjacentMaterial >= 0 && adjacentMaterial < myMaterialCount);

							SolveReal density;
							if (adjacentMaterial != material)
							{
								SolveReal phi = myFluidSurfaces[material](cell);
								SolveReal adjacentPhi = myFluidSurfaces[adjacentMaterial](adjacentCell);

								SolveReal theta;
								if (direction == 0)
									theta = std::fabs(adjacentPhi) / (std::fabs(adjacentPhi) + std::fabs(phi));
								else
									theta = std::fabs(phi) / (std::fabs(phi) + std::fabs(adjacentPhi));

								theta = clamp(theta, SolveReal(0), SolveReal(1));

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

							localSparseMatrixElements.emplace_back(liquidIndex, adjacentIndex, -weight);
							diagonal += weight;
						}

					assert(diagonal > 0);
					localSparseMatrixElements.emplace_back(liquidIndex, liquidIndex, diagonal);

					localDiagonalMatrixElements.emplace_back(liquidIndex, liquidIndex, 1. / diagonal);
				}
				else assert(materialCellLabels(cell) == UNSOLVED_CELL);
			}
		});

		std::vector<Eigen::Triplet<SolveReal>> sparseMatrixElements;
		mergeLocalThreadVectors(sparseMatrixElements, parallelSparseMatrixElements);
		sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());
		sparseMatrix.makeCompressed();

		std::vector<Eigen::Triplet<SolveReal>> diagonalSparseMatrixElements;
		mergeLocalThreadVectors(diagonalSparseMatrixElements, parallelDiagonalSparseMatrixElements);
		diagonalSparseMatrix.setFromTriplets(diagonalSparseMatrixElements.begin(), diagonalSparseMatrixElements.end());
		diagonalSparseMatrix.makeCompressed();
	}

	// Project out null space
	{
		tbb::enumerable_thread_specific<SolveReal> parallelAccumulatedDivergence(0);
		tbb::parallel_for(tbb::blocked_range<int>(0, rhsVector.rows(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				SolveReal& localAccumulatedDivergence = parallelAccumulatedDivergence.local();

				for (int row = range.begin(); row != range.end(); ++row)
					localAccumulatedDivergence += rhsVector(row);
			});

		SolveReal accumulatedDivergence = 0;
		parallelAccumulatedDivergence.combine_each([&](SolveReal localAccumulatedDivergence)
			{
				accumulatedDivergence += localAccumulatedDivergence;
			});

		SolveReal averageDivergence = accumulatedDivergence / SolveReal(solvableCellCount);

		tbb::parallel_for(tbb::blocked_range<int>(0, rhsVector.rows(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int row = range.begin(); row != range.end(); ++row)
					rhsVector(row) -= averageDivergence;
			});
	}

	Vector solutionVector = Vector::Zero(solvableCellCount);

	if (false/*myUseInitialGuessPressure*/)
	{
		assert(myInitialGuessPressure != nullptr);
		tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = materialCellLabels.unflatten(cellIndex);

					if (materialCellLabels(cell) >= 0)
					{
						int index = solvableCellIndices(cell);
						assert(index >= 0);

						solutionVector(index) = (*myInitialGuessPressure)(cell);
					}
				}
			});
	}

	bool printResidual = true;
	auto ResidualPrinter = [&](const Vector& residualVector, int iteration)
	{
		if (printResidual)
		{
			UniformGrid<float> residualGrid(materialCellLabels.size(), 0);

			tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
					{
						Vec2i cell = materialCellLabels.unflatten(flatIndex);

						if (materialCellLabels(cell) >= 0)
						{
							int index = solvableCellIndices(cell);
							assert(index >= 0);

							residualGrid(cell) = residualVector(index);
						}
					}
				});

			residualGrid.printAsOBJ("multimaterialResidual" + std::to_string(iteration));
		}
	};

	bool useMultigridPreconditioer = true;
	if (useMultigridPreconditioer)
	{
		// Build MG grid per material
		using StoreReal = double;

		std::vector<UniformGrid<MGCellLabels>> mgDomainCellLabels(myMaterialCount);
		std::vector<VectorGrid<StoreReal>> mgBoundaryWeights;

		std::vector<Vec2i> mgExpandedOffset(myMaterialCount);
		std::vector<int> mgLevels(myMaterialCount);

		{
			std::vector<UniformGrid<MGCellLabels>> baseDomainCellLabels;

			for (int material = 0; material < myMaterialCount; ++material)
			{
				baseDomainCellLabels.emplace_back(materialCellLabels.size(), MGCellLabels::EXTERIOR_CELL);

				tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
					{
						for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
						{
							Vec2i cell = materialCellLabels.unflatten(cellIndex);

							int localMaterial = materialCellLabels(cell);
							if (localMaterial >= 0)
							{
								assert(solvableCellIndices(cell) >= 0);
								if (localMaterial == material)
									baseDomainCellLabels[material](cell) = MGCellLabels::INTERIOR_CELL;
								else
									baseDomainCellLabels[material](cell) = MGCellLabels::DIRICHLET_CELL;

							}
							else
							{
								assert(baseDomainCellLabels[material](cell) == MGCellLabels::EXTERIOR_CELL);
								assert(solvableCellIndices(cell) == UNSOLVED_CELL);
							}
						}
					});
			}

			std::vector<VectorGrid<StoreReal>> poissonFaceWeights(myMaterialCount);

			for (int material = 0; material < myMaterialCount; ++material)
			{
				poissonFaceWeights[material] = VectorGrid<StoreReal>(myFluidSurfaces[0].xform(), myFluidSurfaces[0].size(), 0, VectorGridSettings::SampleType::STAGGERED);

				for (int axis : {0, 1})
				{
					tbb::parallel_for(tbb::blocked_range<int>(0, poissonFaceWeights[material].grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
						{
							for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
							{
								Vec2i face = poissonFaceWeights[material].grid(axis).unflatten(faceIndex);

								SolveReal solidWeight = mySolidCutCellWeights(face, axis);

								if (solidWeight == 1.)
									continue;

								SolveReal weight = 1. - solidWeight;

								if (weight > 0)
								{
									Vec2i backwardCell = faceToCell(face, axis, 0);
									Vec2i forwardCell = faceToCell(face, axis, 1);

									auto backwardLabel = baseDomainCellLabels[material](backwardCell);
									auto forwardLabel = baseDomainCellLabels[material](forwardCell);

									if ((backwardLabel == MGCellLabels::INTERIOR_CELL && forwardLabel == MGCellLabels::DIRICHLET_CELL) ||
										(backwardLabel == MGCellLabels::DIRICHLET_CELL && forwardLabel == MGCellLabels::INTERIOR_CELL))
									{
										int backwardMaterial = materialCellLabels(backwardCell);
										int forwardMaterial = materialCellLabels(forwardCell);

										assert(backwardMaterial >= 0 && forwardMaterial >= 0);
										assert((backwardMaterial == material && forwardMaterial != material) ||
												(backwardMaterial != material && forwardMaterial == material));

										SolveReal backwardPhi = myFluidSurfaces[backwardMaterial](backwardCell);
										SolveReal forwardPhi = myFluidSurfaces[forwardMaterial](forwardCell);

										SolveReal theta = std::fabs(backwardPhi) / (std::fabs(forwardPhi) + std::fabs(backwardPhi));
										theta = clamp(theta, SolveReal(0), SolveReal(1));

										int adjacentMaterial = (backwardMaterial != material) ? backwardMaterial : forwardMaterial;

										SolveReal backwardDensity = myFluidDensities[backwardMaterial];
										SolveReal forwardDensity = myFluidDensities[forwardMaterial];

										weight *= myFluidDensities[material] / (theta * backwardDensity + (1. - theta) * forwardDensity);
									}
								}

								poissonFaceWeights[material](face, axis) = weight;
							}
						});
				}
			}

			for (int material = 0; material < myMaterialCount; ++material)
			{
				std::pair<Vec2i, int> mgSettings = GeometricMultigridOperators::buildExpandedDomainLabels(mgDomainCellLabels[material], baseDomainCellLabels[material]);
				mgExpandedOffset[material] = mgSettings.first;
				mgLevels[material] = mgSettings.second;
			}

			for (int material = 0; material < myMaterialCount; ++material)
			{
				mgBoundaryWeights.emplace_back(poissonFaceWeights[material].xform(), mgDomainCellLabels[material].size(), 0, VectorGridSettings::SampleType::STAGGERED);

				for (int axis : {0, 1})
					GeometricMultigridOperators::buildExpandedBoundaryWeights(mgBoundaryWeights[material], poissonFaceWeights[material], mgDomainCellLabels[material], mgExpandedOffset[material], axis);

				GeometricMultigridOperators::setBoundaryDomainLabels(mgDomainCellLabels[material], mgBoundaryWeights[material]);

				assert(GeometricMultigridOperators::unitTestBoundaryCells(mgDomainCellLabels[material], &mgBoundaryWeights[material]));
				assert(GeometricMultigridOperators::unitTestExteriorCells(mgDomainCellLabels[material]));
			}
		}

		std::vector<UniformGrid<StoreReal>> mgSourceGrid;
		std::vector<UniformGrid<StoreReal>> mgDestinationGrid;

		for (int material = 0; material < myMaterialCount; ++material)
		{
			mgSourceGrid.emplace_back(mgDomainCellLabels[material].size());
			mgDestinationGrid.emplace_back(mgDomainCellLabels[material].size());
		}

		// Build list of cells at multi-material boundaries
		std::vector<Vec2i> boundarySmootherCells;

		{
			int smootherBandSize = 6;

			UniformGrid<VisitedCellLabels> boundaryCellMarkers(myFluidSurfaces[0].size(), VisitedCellLabels::UNVISITED_CELL);

			// Build initial layer
			tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelBoundaryCells;

			tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					auto& localBoundaryCells = parallelBoundaryCells.local();
					for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
					{
						Vec2i cell = materialCellLabels.unflatten(cellIndex);
						int material = materialCellLabels(cell);
						if (material >= 0)
						{
							assert(solvableCellIndices(cell) >= 0);

							assert(mgDomainCellLabels[material](cell + mgExpandedOffset[material]) == MGCellLabels::INTERIOR_CELL ||
									mgDomainCellLabels[material](cell + mgExpandedOffset[material]) == MGCellLabels::BOUNDARY_CELL);

							bool isBoundaryCell = false;
							for (int axis : {0, 1})
								for (int direction : {0, 1})
								{
									Vec2i face = cellToFace(cell, axis, direction);

									if (mySolidCutCellWeights(face, axis) < 1)
									{
										Vec2i adjacentCell = cellToCell(cell, axis, direction);

										if (adjacentCell[axis] < 0 || adjacentCell[axis] >= materialCellLabels.size()[axis])
											continue;

										if (materialCellLabels(adjacentCell) != material)
										{
											assert(materialCellLabels(adjacentCell) >= 0);
											isBoundaryCell = true;
										}
									}
								}

							if (isBoundaryCell)
								localBoundaryCells.push_back(cell);
						}
					}
				});

			std::vector<Vec2i> tempBoundaryCells;
			for (int layer = 0; layer < smootherBandSize; ++layer)
			{
				// Compile parallel build bubble list
				tempBoundaryCells.clear();
				mergeLocalThreadVectors(tempBoundaryCells, parallelBoundaryCells);

				// Set visited boundary cells
				tbb::parallel_for(tbb::blocked_range<int>(0, tempBoundaryCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
					{
						for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
						{
							Vec2i cell = tempBoundaryCells[cellIndex];
							boundaryCellMarkers(cell) = VisitedCellLabels::VISITED_CELL;
						}
					});

				if (layer < smootherBandSize - 1)
				{
					parallelBoundaryCells.clear();

					tbb::parallel_sort(tempBoundaryCells.begin(), tempBoundaryCells.end(), [&](const Vec2i& vec0, const Vec2i& vec1)
						{
							if (vec0[0] < vec1[0]) return true;
							else if (vec0[0] == vec1[0] && vec0[1] < vec1[1]) return true;
							return false;
						});

					tbb::parallel_for(tbb::blocked_range<int>(0, tempBoundaryCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
						{
							auto& localBoundaryCells = parallelBoundaryCells.local();

							int cellIndex = range.begin();

							// Advanced past duplicates
							if (cellIndex > 0)
							{
								while (cellIndex != range.end() && tempBoundaryCells[cellIndex - 1] == tempBoundaryCells[cellIndex])
									++cellIndex;
							}

							Vec2i oldCell(-1);
							for (; cellIndex != range.end(); ++cellIndex)
							{
								Vec2i cell = tempBoundaryCells[cellIndex];

								if (cell == oldCell)
									continue;
								else
									oldCell = cell;

								assert(materialCellLabels(cell) >= 0);
								assert(boundaryCellMarkers(cell) == VisitedCellLabels::VISITED_CELL);

								for (int axis : {0, 1})
									for (int direction : {0, 1})
									{
										Vec2i adjacentCell = cellToCell(cell, axis, direction);

										if (adjacentCell[axis] < 0 || adjacentCell[axis] >= materialCellLabels.size()[axis])
											continue;

										Vec2i face = cellToFace(cell, axis, direction);

										if (mySolidCutCellWeights(face, axis) < 1)
										{
											if (materialCellLabels(adjacentCell) >= 0 &&
												boundaryCellMarkers(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
											{
												localBoundaryCells.push_back(adjacentCell);
											}
										}
									}
							}
						});
				}
			}

			// Build list of boundary cells
			parallelBoundaryCells.clear();
			tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					auto& localBoundaryCells = parallelBoundaryCells.local();
					for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
					{
						Vec2i cell = materialCellLabels.unflatten(cellIndex);
						if (boundaryCellMarkers(cell) == VisitedCellLabels::VISITED_CELL)
						{
							assert(materialCellLabels(cell) >= 0);
							localBoundaryCells.push_back(cell);
						}
					}
				});

			mergeLocalThreadVectors(boundarySmootherCells, parallelBoundaryCells);
		}

		std::vector<std::unique_ptr<GeometricMultigridPoissonSolver>> mgPreconditioners(myMaterialCount);
		
		for (int material = 0; material < myMaterialCount; ++material)
			mgPreconditioners[material] = std::make_unique<GeometricMultigridPoissonSolver>(mgDomainCellLabels[material], mgBoundaryWeights[material], mgLevels[material], 1. /* unit dx */);

		UniformGrid<SolveReal> smootherDestinationGrid(materialCellLabels.size());
		UniformGrid<SolveReal> smootherSourceGrid(materialCellLabels.size());

		auto MultigridPreconditioner = [&](const Vector& sourceVector)
		{
			Vector destinationVector = Vector::Zero(solvableCellCount);

			assert(sourceVector.rows() == solvableCellCount);

			for (int material = 0; material < myMaterialCount; ++material)
			{
				mgSourceGrid[material].reset(0);
				mgDestinationGrid[material].reset(0);
			}

			smootherDestinationGrid.reset(0);
			smootherSourceGrid.reset(0);

			copyToPreconditionerGrids(mgSourceGrid, smootherSourceGrid, mgDomainCellLabels,
										materialCellLabels, solvableCellIndices, sourceVector,
										mgExpandedOffset);

			// Apply smoother to cells along multi-material boundaries
			std::vector<SolveReal> tempDestinationVector;

			int smootherIterations = 20;

			for (int smootherIteration = 0; smootherIteration < smootherIterations; ++smootherIteration)
			{
				tempDestinationVector.clear();
				tempDestinationVector.resize(boundarySmootherCells.size(), 0);

				applyBoundarySmoothing(tempDestinationVector, boundarySmootherCells, mgDomainCellLabels, materialCellLabels, smootherDestinationGrid, smootherSourceGrid, mgExpandedOffset);

				updateDestinationGrid(smootherDestinationGrid, materialCellLabels, boundarySmootherCells, tempDestinationVector);
			}

			for (int mgIterations = 0; mgIterations < 1; ++mgIterations)
			{
				// Apply Dirichlet condition to MG
				applyDirichletToMG(mgSourceGrid, mgDestinationGrid, smootherDestinationGrid, mgDomainCellLabels, materialCellLabels, solvableCellIndices, sourceVector, boundarySmootherCells, mgExpandedOffset);

				for (int material = 0; material < myMaterialCount; ++material)
					mgPreconditioners[material]->applyMGVCycle(mgDestinationGrid[material], mgSourceGrid[material], true);

				// Transfer MG results back to smoother grid
				// TODO: re-write to only copy near boundaries
				updateSmootherGrid(smootherDestinationGrid, mgDestinationGrid, materialCellLabels, mgDomainCellLabels, mgExpandedOffset);

				// Apply smoother along multi-material boundaries
				for (int smootherIteration = 0; smootherIteration < smootherIterations; ++smootherIteration)
				{
					tempDestinationVector.clear();
					tempDestinationVector.resize(boundarySmootherCells.size(), 0);

					applyBoundarySmoothing(tempDestinationVector, boundarySmootherCells, mgDomainCellLabels, materialCellLabels, smootherDestinationGrid, smootherSourceGrid, mgExpandedOffset);

					updateDestinationGrid(smootherDestinationGrid, materialCellLabels, boundarySmootherCells, tempDestinationVector);
				}
			}

			// Copy MG solutions to destination grid
			copyMGSolutionToVector(destinationVector, materialCellLabels, solvableCellIndices, mgDomainCellLabels, mgDestinationGrid, mgExpandedOffset);

			// Copy boundary smoother solutions to destination grid
			copyBoundarySolutionToVector(destinationVector, materialCellLabels, solvableCellIndices, smootherDestinationGrid, boundarySmootherCells);

			return destinationVector;
		};

		auto MatrixVectorMultiply = [&](const Vector& source)
		{
			return sparseMatrix * source;
		};

		solutionVector = Vector::Zero(solvableCellCount);
		
		solveConjugateGradient<SolveReal>(solutionVector, rhsVector, MatrixVectorMultiply, MultigridPreconditioner, ResidualPrinter, 1E-5, 2000, false);
	}
	else
	{
		auto MatrixVectorMultiply = [&](const Vector& source)
		{
			return sparseMatrix * source;
		};

		auto DiagonalPreconditioner = [&](const Vector& source)
		{
			return diagonalSparseMatrix * source;
		};

		solveConjugateGradient<SolveReal>(solutionVector, rhsVector, MatrixVectorMultiply, DiagonalPreconditioner, ResidualPrinter, 1E-5, 2000, false);
	}

	// Copy resulting vector to pressure grid
	tbb::parallel_for(tbb::blocked_range<int>(0, solvableCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = solvableCellIndices.unflatten(cellIndex);

			int liquidIndex = solvableCellIndices(cell);

			if (liquidIndex >= 0)
			{
				assert(materialCellLabels(cell) >= 0);
				myPressure(cell) = solutionVector(liquidIndex);
			}
			else (assert(materialCellLabels(cell) == UNSOLVED_CELL));
		}
	});

	// Set valid faces
	VectorGrid<VisitedCellLabels> validFaces(mySolidSurface.xform(), mySolidSurface.size(), VisitedCellLabels::UNVISITED_CELL, VectorGridSettings::SampleType::STAGGERED);
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = validFaces.grid(axis).unflatten(faceIndex);

				if (mySolidCutCellWeights(face, axis) < 1.)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					if (backwardCell[axis] < 0 || forwardCell[axis] >= solvableCellIndices.size()[axis])
						return;

					assert(solvableCellIndices(backwardCell) >= 0 && solvableCellIndices(forwardCell) >= 0);
					validFaces(face, axis) = VisitedCellLabels::FINISHED_CELL;
				}
			}
		});
	}

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = validFaces.grid(axis).unflatten(faceIndex);

				if (validFaces(face, axis) == VisitedCellLabels::FINISHED_CELL)
				{
					assert(mySolidCutCellWeights(face, axis) < 1);

					SolveReal phi[2];
					SolveReal sampleDensity[2];
					int materials[2];

					SolveReal gradient = 0;
					for (int direction : {0, 1})
					{
						Vec2i cell = faceToCell(face, axis, direction);
						int material = materialCellLabels(cell);

						materials[direction] = material;
						assert(solvableCellIndices(cell) >= 0);

						phi[direction] = myFluidSurfaces[material](cell);

						sampleDensity[direction] = myFluidDensities[material];

						SolveReal sign = (direction == 0) ? -1 : 1;
						gradient += sign * myPressure(cell);
					}

					SolveReal density;
					if (materials[0] == materials[1])
					{
						assert(sampleDensity[0] == sampleDensity[1]);
						density = sampleDensity[0];
					}
					else
					{
						SolveReal theta = std::fabs(phi[0]) / (std::fabs(phi[1]) + std::fabs(phi[0]));
						theta = clamp(theta, SolveReal(0), SolveReal(1));
						density = theta * sampleDensity[0] + (1. - theta) * sampleDensity[1];
					}

					gradient /= density;

					// Update every material's velocity with a non-zero face weight.
					for (int material = 0; material < myMaterialCount; ++material)
					{
						if (myMaterialCutCellWeights[material](face, axis) > 0.)
						{
							myValidMaterialFaces[material](face, axis) = VisitedCellLabels::FINISHED_CELL;
							fluidVelocities[material](face, axis) -= gradient;
						}
						else
							fluidVelocities[material](face, axis) = 0;
					}
				}
				else
				{
					for (int material = 0; material < myMaterialCount; ++material)
						fluidVelocities[material](face, axis) = 0;
				}
			}
		});
	}

	//
	// Debug test to verify divergence free constraints are met
	//

	{
		tbb::enumerable_thread_specific<SolveReal> parallelMaxDivergence(0);
		tbb::enumerable_thread_specific<SolveReal> parallelAccumulatedDivergence(0);
		tbb::enumerable_thread_specific<SolveReal> parallelCellCount(0);

		tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto& localMaxDivergence = parallelMaxDivergence.local();
				auto& localAccumulatedDivergence = parallelAccumulatedDivergence.local();
				auto& localCellCount = parallelCellCount.local();

				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = materialCellLabels.unflatten(cellIndex);

					if (materialCellLabels(cell) >= 0)
					{
						assert(solvableCellIndices(cell) >= 0);

						SolveReal divergence = 0;
						for (int axis : {0, 1})
							for (int direction : {0, 1})
							{
								Vec2i face = cellToFace(cell, axis, direction);
								SolveReal sign = (direction == 0) ? -1 : 1;

								for (int material = 0; material < myMaterialCount; ++material)
								{
									SolveReal weight = myMaterialCutCellWeights[material](face, axis);

									if (weight > 0)
										divergence += sign * weight * fluidVelocities[material](face, axis);
								}
							}

						localAccumulatedDivergence += std::fabs(divergence);
						localMaxDivergence = std::max(localMaxDivergence, std::fabs(divergence));
						++localCellCount;
					}
					else assert(solvableCellIndices(cell) == UNSOLVED_CELL);
				}
			});

		SolveReal accumulatedDivergence = 0;
		SolveReal maxDivergence;
		SolveReal cellCount = 0;

		parallelMaxDivergence.combine_each([&](SolveReal localMaxDivergence)
			{
				maxDivergence = std::max(localMaxDivergence, maxDivergence);
			});

		parallelAccumulatedDivergence.combine_each([&](SolveReal localAccumulatedDivergence)
			{
				accumulatedDivergence += localAccumulatedDivergence;
			});

		parallelCellCount.combine_each([&](SolveReal localCellCount)
			{
				cellCount += localCellCount;
			});

		assert(cellCount == solvableCellCount);

		std::cout << "Accumulated divergence: " << accumulatedDivergence << std::endl;
		std::cout << "Average divergence: " << accumulatedDivergence / SolveReal(cellCount) << std::endl;
		std::cout << "Max divergence: " << maxDivergence << std::endl;
	}

}

}