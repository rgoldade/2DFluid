#include "GeometricMGOperations.h"

#include "tbb/tbb.h"

using namespace tbb;

Vec2i getChildCell(Vec2i cell, const int childIndex)
{
	assert(childIndex < 4);

	cell *= 2;
	for (int axis : {0, 1})
	{
		if (childIndex & (1 << axis))
			++cell[axis];
	}

	return cell;
}

void GeometricMGOperations::dampedJacobiPoissonSmoother(UniformGrid<Real> &solution,
														const UniformGrid<Real> &rhs,
														const UniformGrid<CellLabels> &cellLabels,
														const Real dx)
{
	assert(solution.size() == rhs.size() &
			solution.size() == cellLabels.size());

	// TODO: fix MG code to remove dependency on dx

	Real gridScalar = 1. / Util::sqr(dx);

	UniformGrid<Real> tempSolution = solution;

	int totalVoxels = solution.size()[0] * solution.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = solution.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real count = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= tempSolution(adjacentCell);
							++count;
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							++count;
					}

				laplacian += count * tempSolution(cell);
				laplacian *= gridScalar;
				Real residual = rhs(cell) - laplacian;
				residual /= (count * gridScalar);

				solution(cell) += 2. / 3. * residual;
			}
		}
	});
}

void GeometricMGOperations::dampedJacobiWeightedPoissonSmoother(UniformGrid<Real> &solution,
																const UniformGrid<Real> &rhs,
																const UniformGrid<CellLabels> &cellLabels,
																const VectorGrid<Real> &gradientWeights,
																const Real dx)
{
	assert(solution.size() == rhs.size() &
			solution.size() == cellLabels.size());

	Real gridScalar = 1. / Util::sqr(dx);

	UniformGrid<Real> tempSolution = solution;

	int totalVoxels = solution.size()[0] * solution.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = solution.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real diagonal = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						Vec2i face = cellToFace(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= gradientWeights(face, axis) * tempSolution(adjacentCell);
							diagonal += gradientWeights(face, axis);
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							diagonal += gradientWeights(face, axis);
						else assert(gradientWeights(face, axis) == 0);
					}

				laplacian += diagonal * tempSolution(cell);
				laplacian *= gridScalar;
				Real residual = rhs(cell) - laplacian;
				residual /= (diagonal * gridScalar);

				solution(cell) += 2. / 3. * residual;
			}
		}
	});
}

void GeometricMGOperations::dampedJacobiPoissonSmoother(UniformGrid<Real> &solution,
														const UniformGrid<Real> &rhs,
														const UniformGrid<CellLabels> &cellLabels,
														const std::vector<Vec2i> &boundaryCells,
														const Real dx)
{
	assert(solution.size() == rhs.size() &
		solution.size() == cellLabels.size());

	Real gridScalar = 1. / Util::sqr(dx);

	std::vector<Real> tempSolution(boundaryCells.size(), 0);

	parallel_for(blocked_range<int>(0, boundaryCells.size(), tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];

			assert(cellLabels(cell) == CellLabels::INTERIOR);

			Real laplacian = 0;
			Real count = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.size()[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
					{
						laplacian -= solution(adjacentCell);
						++count;
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
						++count;
				}

			laplacian += count * solution(cell);
			laplacian *= gridScalar;
			Real residual = rhs(cell) - laplacian;
			residual /= (count * gridScalar);

			tempSolution[cellIndex] = solution(cell) + 2. / 3. * residual;
		}
	});

	parallel_for(blocked_range<int>(0, boundaryCells.size(), tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];
			solution(cell) = tempSolution[cellIndex];
		}
	});
}

void GeometricMGOperations::dampedJacobiWeightedPoissonSmoother(UniformGrid<Real> &solution,
																const UniformGrid<Real> &rhs,
																const UniformGrid<CellLabels> &cellLabels,
																const std::vector<Vec2i> &boundaryCells,
																const VectorGrid<Real> &gradientWeights,
																const Real dx)
{
	assert(solution.size() == rhs.size() &
		solution.size() == cellLabels.size());

	Real gridScalar = 1. / Util::sqr(dx);

	std::vector<Real> tempSolution(boundaryCells.size(), 0);

	parallel_for(blocked_range<int>(0, boundaryCells.size(), tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];

			assert(cellLabels(cell) == CellLabels::INTERIOR);

			Real laplacian = 0;
			Real diagonal = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);
					Vec2i face = cellToFace(cell, axis, direction);

					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.size()[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
					{
						laplacian -= gradientWeights(face, axis) * solution(adjacentCell);
						diagonal += gradientWeights(face, axis);
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
						diagonal += gradientWeights(face, axis);
					else assert(gradientWeights(face, axis) == 0);
				}

			laplacian += diagonal * solution(cell);
			laplacian *= gridScalar;
			Real residual = rhs(cell) - laplacian;
			residual /= (diagonal * gridScalar);

			tempSolution[cellIndex] = solution(cell) + 2. / 3. * residual;
		}
	});

	parallel_for(blocked_range<int>(0, boundaryCells.size(), tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];
			solution(cell) = tempSolution[cellIndex];
		}
	});
}

void GeometricMGOperations::applyPoissonMatrix(UniformGrid<Real> &destination,
												const UniformGrid<Real> &source,
												const UniformGrid<CellLabels> &cellLabels,
												const Real dx)
{
	assert(destination.size() == source.size() &&
			source.size() == cellLabels.size());

	Real gridScalar = 1. / Util::sqr(dx);

	int totalVoxels = destination.size()[0] * destination.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destination.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real count = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destination.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= source(adjacentCell);
							++count;
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							++count;
					}

				laplacian += count * source(cell);
				laplacian *= gridScalar;
				destination(cell) = laplacian;
			}
		}
	});
}

void GeometricMGOperations::applyWeightedPoissonMatrix(UniformGrid<Real> &destination,
														const UniformGrid<Real> &source,
														const UniformGrid<CellLabels> &cellLabels,
														const VectorGrid<Real> &gradientWeights,
														const Real dx)
{
	assert(destination.size() == source.size() &&
		source.size() == cellLabels.size());

	assert(gradientWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
		gradientWeights.size(0)[0] == destination.size()[0] + 1 &&
		gradientWeights.size(0)[1] == destination.size()[1] &&
		gradientWeights.size(1)[0] == destination.size()[0] &&
		gradientWeights.size(1)[1] == destination.size()[1] + 1);

	Real gridScalar = 1. / Util::sqr(dx);

	int totalVoxels = destination.size()[0] * destination.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destination.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real diagonal = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						Vec2i face = cellToFace(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destination.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= gradientWeights(face, axis) * source(adjacentCell);
							diagonal += gradientWeights(face, axis);
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							diagonal += gradientWeights(face, axis);
						else assert(gradientWeights(face, axis) == 0);
					}

				laplacian += diagonal * source(cell);
				laplacian *= gridScalar;
				destination(cell) = laplacian;
			}
		}
	});
}

void GeometricMGOperations::computePoissonResidual(UniformGrid<Real> &residual,
													const UniformGrid<Real> &solution,
													const UniformGrid<Real> &rhs,
													const UniformGrid<CellLabels> &cellLabels,
													const Real dx)
{
	assert(residual.size() == solution.size() &&
			residual.size() == rhs.size() &
			residual.size() == cellLabels.size());

	Real gridScalar = 1. / Util::sqr(dx);

	int totalVoxels = residual.size()[0] * residual.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = residual.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real count = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < residual.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= solution(adjacentCell);
							++count;
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							++count;
					}

				laplacian += count * solution(cell);
				laplacian *= gridScalar;
				residual(cell) = rhs(cell) - laplacian;
			}
		}
	});
}

void GeometricMGOperations::computeWeightedPoissonResidual(UniformGrid<Real> &residual,
															const UniformGrid<Real> &solution,
															const UniformGrid<Real> &rhs,
															const UniformGrid<CellLabels> &cellLabels,
															const VectorGrid<Real> &gradientWeights,
															const Real dx)
{
	assert(residual.size() == solution.size() &&
		residual.size() == rhs.size() &
		residual.size() == cellLabels.size());

	assert(gradientWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
		gradientWeights.size(0)[0] == residual.size()[0] + 1 &&
		gradientWeights.size(0)[1] == residual.size()[1] &&
		gradientWeights.size(1)[0] == residual.size()[0] &&
		gradientWeights.size(1)[1] == residual.size()[1] + 1);

	Real gridScalar = 1. / Util::sqr(dx);

	int totalVoxels = residual.size()[0] * residual.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = residual.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
			{
				Real laplacian = 0;
				Real diagonal = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						Vec2i face = cellToFace(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < residual.size()[axis]);

						if (cellLabels(adjacentCell) == CellLabels::INTERIOR)
						{
							laplacian -= gradientWeights(face, axis) * solution(adjacentCell);
							diagonal += gradientWeights(face, axis);
						}
						else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET)
							diagonal += gradientWeights(face, axis);
						else assert(gradientWeights(face, axis) == 0);
					}

				laplacian += diagonal * solution(cell);
				laplacian *= gridScalar;
				residual(cell) = rhs(cell) - laplacian;
			}
		}
	});
}

static Real restrictionWeights[4][4] = {{1. / 64., 3. / 64., 3. / 64., 1. / 64.},
										{3. / 64., 9. / 64., 9. / 64., 3. / 64.},
										{3. / 64., 9. / 64., 9. / 64., 3. / 64.},
										{1. / 64., 3. / 64., 3. / 64., 1. / 64.} };

void GeometricMGOperations::downsample(UniformGrid<Real> &destinationGrid,
										const UniformGrid<Real> &sourceGrid,
										const UniformGrid<CellLabels> &destinationCellLabels,
										const UniformGrid<CellLabels> &sourceCellLabels)
{
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(2 * destinationGrid.size() == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	assert(destinationGrid.size()[0] % 2 == 0 &&
			destinationGrid.size()[1] % 2 == 0 &&
			sourceGrid.size()[0] % 2 == 0 &&
			sourceGrid.size()[1] % 2 == 0);

	int totalVoxels = destinationCellLabels.size()[0] * destinationCellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(flatIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR)
			{
				// Iterator over source cells	
				Real sampleValue = 0;

				Vec2i startCell = 2 * cell - Vec2i(1);
				forEachVoxelRange(Vec2i(0), Vec2i(4), [&](const Vec2i &sampleIndex)
				{
					Vec2i sampleCell = startCell + sampleIndex;

					assert(sampleCell[0] >= 0 && sampleCell[0] < sourceGrid.size()[0] &&
						sampleCell[1] >= 0 && sampleCell[1] < sourceGrid.size()[1]);

					if (sourceCellLabels(sampleCell) == CellLabels::INTERIOR)
						sampleValue += restrictionWeights[sampleIndex[0]][sampleIndex[1]] * sourceGrid(sampleCell);
				});

				destinationGrid(cell) = sampleValue;
			}
		}
	});
}

void GeometricMGOperations::upsample(UniformGrid<Real> &destinationGrid,
										const UniformGrid<Real> &sourceGrid,
										const UniformGrid<CellLabels> &destinationCellLabels,
										const UniformGrid<CellLabels> &sourceCellLabels)
{
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(destinationGrid.size() / 2 == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	assert(destinationGrid.size()[0] % 2 == 0 &&
			destinationGrid.size()[1] % 2 == 0 &&
			sourceGrid.size()[0] % 2 == 0 &&
			sourceGrid.size()[1] % 2 == 0);

	int totalVoxels = destinationCellLabels.size()[0] * destinationCellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(flatIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR)
			{
				// Iterator over source cells	
				Real sampleValue = 0;

				Vec2R samplePoint = .5 * (Vec2R(cell) + Vec2R(.5)) - Vec2R(.5);

				Vec2i startCell = Vec2i(samplePoint);

				Vec2R interpWeight = samplePoint - Vec2R(startCell);

				// Hard code interpolation
				Real v00 = (sourceCellLabels(startCell) == CellLabels::INTERIOR) ? sourceGrid(startCell) : 0;
				Real v01 = (sourceCellLabels(startCell + Vec2i(0, 1)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(0, 1)) : 0;
				Real v10 = (sourceCellLabels(startCell + Vec2i(1, 0)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(1, 0)) : 0;
				Real v11 = (sourceCellLabels(startCell + Vec2i(1, 1)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(1, 1)) : 0;

				destinationGrid(cell) = Util::bilerp(v00, v10, v01, v11, interpWeight[0], interpWeight[1]);
			}
		}
	});
}

void GeometricMGOperations::upsampleAndAdd(UniformGrid<Real> &destinationGrid,
											const UniformGrid<Real> &sourceGrid,
											const UniformGrid<CellLabels> &destinationCellLabels,
											const UniformGrid<CellLabels> &sourceCellLabels)
{
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(destinationGrid.size() / 2 == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	assert(destinationGrid.size()[0] % 2 == 0 &&
		destinationGrid.size()[1] % 2 == 0 &&
		sourceGrid.size()[0] % 2 == 0 &&
		sourceGrid.size()[1] % 2 == 0);

	int totalVoxels = destinationCellLabels.size()[0] * destinationCellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(flatIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR)
			{
				// Iterator over source cells	
				Real sampleValue = 0;

				Vec2R samplePoint = .5 * (Vec2R(cell) + Vec2R(.5)) - Vec2R(.5);

				Vec2i startCell = Vec2i(samplePoint);

				Vec2R interpWeight = samplePoint - Vec2R(startCell);

				// Hard code interpolation
				Real v00 = (sourceCellLabels(startCell) == CellLabels::INTERIOR) ? sourceGrid(startCell) : 0;
				Real v01 = (sourceCellLabels(startCell + Vec2i(0, 1)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(0, 1)) : 0;
				Real v10 = (sourceCellLabels(startCell + Vec2i(1, 0)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(1, 0)) : 0;
				Real v11 = (sourceCellLabels(startCell + Vec2i(1, 1)) == CellLabels::INTERIOR) ? sourceGrid(startCell + Vec2i(1, 1)) : 0;

				destinationGrid(cell) += Util::bilerp(v00, v10, v01, v11, interpWeight[0], interpWeight[1]);
			}
		}
	});
}

UniformGrid<GeometricMGOperations::CellLabels> GeometricMGOperations::buildCoarseCellLabels(const UniformGrid<CellLabels> &sourceCellLabels)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 &&
			sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<CellLabels> destinationCellLabels(sourceCellLabels.size() / 2, CellLabels::EXTERIOR);

	int totalVoxels = destinationCellLabels.size()[0] * destinationCellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(flatIndex);

			// Iterate over the destination cell's children.
			Vec2i childCell = 2 * cell;

			bool hasDirichletChild = false;
			bool hasInteriorChild = false;

			for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
			{
				Vec2i localCell = cellToNode(childCell, cellIndex);

				if (sourceCellLabels(localCell) == CellLabels::DIRICHLET)
				{
					hasDirichletChild = true;
					break;
				}
				else if (sourceCellLabels(localCell) == CellLabels::INTERIOR)
					hasInteriorChild = true;
			}

			if (hasDirichletChild)
				destinationCellLabels(cell) = CellLabels::DIRICHLET;
			else if (hasInteriorChild)
				destinationCellLabels(cell) = CellLabels::INTERIOR;
			else
				destinationCellLabels(cell) = CellLabels::EXTERIOR;
		}
	});

	return destinationCellLabels;
}

UniformGrid<GeometricMGOperations::CellLabels> GeometricMGOperations::buildBoundaryCellLabels(const UniformGrid<CellLabels> &sourceCellLabels,
																								int boundaryWidth)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 &&
		sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<CellLabels> boundaryCellLabels(sourceCellLabels.size(), CellLabels::EXTERIOR);
	UniformGrid<MarkedCells> visitedCells(sourceCellLabels.size(), MarkedCells::UNVISITED);

	int totalVoxels = sourceCellLabels.size()[0] * sourceCellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = sourceCellLabels.unflatten(flatIndex);

			if (sourceCellLabels(cell) == CellLabels::INTERIOR)
			{
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (sourceCellLabels(adjacentCell) == CellLabels::DIRICHLET ||
							sourceCellLabels(adjacentCell) == CellLabels::EXTERIOR)
						{
							boundaryCellLabels(cell) = CellLabels::BOUNDARY;
							visitedCells(cell) = MarkedCells::VISITED;
						}
					}
			}
		}
	});

	for (int layer = 1; layer < boundaryWidth; ++layer)
	{
		parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = sourceCellLabels.unflatten(flatIndex);

				if (visitedCells(cell) == MarkedCells::VISITED)
					visitedCells(cell) = MarkedCells::FINISHED;
			}
		});

		parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = sourceCellLabels.unflatten(flatIndex);

				if (boundaryCellLabels(cell) == CellLabels::BOUNDARY &&
					visitedCells(cell) == MarkedCells::FINISHED)
				{
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (sourceCellLabels(adjacentCell) == CellLabels::INTERIOR &&
								visitedCells(adjacentCell) == MarkedCells::UNVISITED)
							{
								boundaryCellLabels(adjacentCell) = CellLabels::BOUNDARY;
								visitedCells(adjacentCell) = MarkedCells::VISITED;
							}
						}
				}
			}
		});
	}

	return boundaryCellLabels;
}

std::vector<Vec2i> GeometricMGOperations::buildBoundaryCells(const UniformGrid<CellLabels> &sourceCellLabels,
																int boundaryWidth)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 &&
			sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<MarkedCells> visitedCells(sourceCellLabels.size(), MarkedCells::UNVISITED);

	using ParallelBoundaryCellListType = enumerable_thread_specific<std::vector<Vec2i>>;

	ParallelBoundaryCellListType parallelBoundaryCellList;

	int totalVoxels = sourceCellLabels.size()[0] * sourceCellLabels.size()[1];

	// Build initial layer
	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		auto &localBoundaryCellList = parallelBoundaryCellList.local();
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = sourceCellLabels.unflatten(flatIndex);

			if (sourceCellLabels(cell) == CellLabels::INTERIOR)
			{
				bool isBoundary = false;
				for (int axis = 0; axis < 2 && !isBoundary; ++axis)
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (sourceCellLabels(adjacentCell) == CellLabels::DIRICHLET ||
							sourceCellLabels(adjacentCell) == CellLabels::EXTERIOR)
						{
							isBoundary = true;
							break;
						}
					}

				if (isBoundary)
					localBoundaryCellList.push_back(cell);
			}
		}
	});

	std::vector<std::vector<Vec2i>> perLayerboundaryCells(boundaryWidth);

	// Pre-allocate memory
	int cellCount = 0;
	parallelBoundaryCellList.combine_each([&cellCount](const std::vector<Vec2i> &localList)
	{
		cellCount += localList.size();
	});

	perLayerboundaryCells[0].reserve(cellCount);

	// Insert cells
	parallelBoundaryCellList.combine_each([&perLayerboundaryCells](const std::vector<Vec2i> &localList)
	{
		perLayerboundaryCells[0].insert(perLayerboundaryCells[0].end(),
										localList.begin(),
										localList.end());
	});
	
	auto vecCompare = [](const Vec2i &vec0, const Vec2i &vec1)
	{
		if (vec0[0] < vec1[0])
			return true;
		else if (vec0[0] == vec1[0] &&
			vec0[1] < vec1[1])
			return true;

		return false;
	};

	parallel_sort(perLayerboundaryCells[0].begin(), perLayerboundaryCells[0].end(), vecCompare);

	std::vector<Vec2i> preSortedTempList;
	for (int layer = 1; layer < boundaryWidth; ++layer)
	{
		// Set cells to visited
		int localCellCount = perLayerboundaryCells[layer - 1].size();
		parallel_for(blocked_range<int>(0, localCellCount, tbbGrainSize), [&](const blocked_range<int> &range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = perLayerboundaryCells[layer - 1][cellIndex];

				visitedCells(cell) = MarkedCells::FINISHED;
			}
		});

		parallelBoundaryCellList.clear();

		parallel_for(blocked_range<int>(0, localCellCount, tbbGrainSize), [&](const blocked_range<int> &range)
		{
			auto &localBoundaryCellList = parallelBoundaryCellList.local();
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = perLayerboundaryCells[layer - 1][cellIndex];

				assert(visitedCells(cell) == MarkedCells::FINISHED);

				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (sourceCellLabels(adjacentCell) == CellLabels::INTERIOR &&
							visitedCells(adjacentCell) == MarkedCells::UNVISITED)
						{
							localBoundaryCellList.push_back(adjacentCell);
						}
					}
			}
		});

		//
		// Collect new layer
		//

		// Pre-allocate memory
		cellCount = 0;
		parallelBoundaryCellList.combine_each([&cellCount](const std::vector<Vec2i> &localList) { cellCount += localList.size(); });

		preSortedTempList.clear();
		preSortedTempList.reserve(cellCount);
		// Insert cells into a temporary list
		parallelBoundaryCellList.combine_each([&preSortedTempList](const std::vector<Vec2i> &localList)
		{
			preSortedTempList.insert(preSortedTempList.end(), localList.begin(), localList.end());
		});

		// Sort temporary list and copy without duplicates to the final list
		parallel_sort(preSortedTempList.begin(), preSortedTempList.end(), vecCompare);

		parallelBoundaryCellList.clear();
		parallel_for(blocked_range<int>(0, preSortedTempList.size(), tbbGrainSize), [&](const blocked_range<int> &range)
		{
			auto &localBoundaryCellList = parallelBoundaryCellList.local();

			const int endCell = range.end();
			int startCell = range.begin();

			if (startCell > 0 && preSortedTempList[startCell] == preSortedTempList[startCell - 1])
			{
				Vec2i localCell = preSortedTempList[startCell];

				while (startCell != endCell && localCell == preSortedTempList[startCell])
					++startCell;

				if (startCell == endCell)
					return;
			}

			Vec2i oldCell(-1);

			for (int cellIndex = startCell; cellIndex < endCell; ++cellIndex)
			{
				Vec2i newCell = preSortedTempList[cellIndex];
				if (oldCell != newCell)
				{
					localBoundaryCellList.push_back(newCell);
					oldCell = newCell;
				}
			}
		});

		// Collect final list
		cellCount = 0;
		parallelBoundaryCellList.combine_each([&cellCount](const std::vector<Vec2i> &localList) { cellCount += localList.size(); });

		assert(perLayerboundaryCells[layer].empty());
		perLayerboundaryCells[layer].reserve(cellCount);
		// Insert cells into a temporary list
		parallelBoundaryCellList.combine_each([&perLayerboundaryCells, &layer](const std::vector<Vec2i> &localList)
		{
			perLayerboundaryCells[layer].insert(perLayerboundaryCells[layer].end(), localList.begin(), localList.end());
		});

		// Sort temporary list and copy without duplicates to the final list
		parallel_sort(perLayerboundaryCells[layer].begin(), perLayerboundaryCells[layer].end(), vecCompare);
	}

	std::vector<Vec2i> finalBoundaryLayerCells;

	int finalCellCount = 0;
	for (int layer = 0; layer < boundaryWidth; ++layer)
		finalCellCount += perLayerboundaryCells[layer].size();

	finalBoundaryLayerCells.reserve(finalCellCount);
	for (int layer = 0; layer < boundaryWidth; ++layer)
		finalBoundaryLayerCells.insert(finalBoundaryLayerCells.end(),
										perLayerboundaryCells[layer].begin(),
										perLayerboundaryCells[layer].end());

	parallel_sort(finalBoundaryLayerCells.begin(), finalBoundaryLayerCells.end(), vecCompare);

	return finalBoundaryLayerCells;
}

bool GeometricMGOperations::unitTestCoarsening(const UniformGrid<CellLabels> &coarseCellLabels,
												const UniformGrid<CellLabels> &fineCellLabels)
{
	// The coarse cell grid must be exactly have the size of the fine cell grid.
	if (2 * coarseCellLabels.size() != fineCellLabels.size())
		return false;

	if (coarseCellLabels.size()[0] % 2 != 0 ||
		coarseCellLabels.size()[1] % 2 != 0 ||
		fineCellLabels.size()[0] % 2 != 0 ||
		fineCellLabels.size()[1] % 2 != 0)
		return false;

	{
		bool testPassed = true;
		forEachVoxelRange(Vec2i(0), fineCellLabels.size(), [&](const Vec2i &fineCell)
		{
			Vec2i coarseCell = fineCell / 2;
			if (!testPassed) return;
			// If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
			if (fineCellLabels(fineCell) == CellLabels::DIRICHLET)
			{
				if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET)
					testPassed = false;
			}
			else if (fineCellLabels(fineCell) == CellLabels::INTERIOR)
			{
				// If the fine cell is interior, the coarse cell can be either
				// interior or Dirichlet (if a sibling cell is Dirichlet).
				if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR)
					testPassed = false;
			}
		});

		if(!testPassed) return false;
	}
	{
		bool testPassed = true;
		forEachVoxelRange(Vec2i(0), coarseCellLabels.size(), [&](const Vec2i &coarseCell)
		{
			if (!testPassed) return;

			bool foundDirichletChild = false;
			bool foundInteriorChild = false;
			bool foundExteriorChild = false;
			for (int childIndex = 0; childIndex < 4; ++childIndex)
			{
				Vec2i fineCell = getChildCell(coarseCell, childIndex);

				auto fineLabel = fineCellLabels(fineCell);

				if (fineLabel == CellLabels::DIRICHLET)
					foundDirichletChild = true;
				else if (fineLabel == CellLabels::INTERIOR)
					foundInteriorChild = true;
				else if (fineLabel == CellLabels::EXTERIOR)
					foundExteriorChild = true;
			}

			auto coarseLabel = coarseCellLabels(coarseCell);
			if (coarseLabel == CellLabels::DIRICHLET)
			{
				if (!foundDirichletChild)
					testPassed = false;
			}
			else if (coarseLabel == CellLabels::INTERIOR)
			{
				if (foundDirichletChild || !foundInteriorChild)
					testPassed = false;
			}
			else if (coarseLabel == CellLabels::EXTERIOR)
			{
				if (foundDirichletChild || foundInteriorChild ||
					!foundExteriorChild)
					testPassed = false;
			}
		});

		if(!testPassed) return false;
	}

	return true;
}

double GeometricMGOperations::dotProduct(const UniformGrid<Real> &aVectorGrid,
											const UniformGrid<Real> &bVectorGrid,
											const UniformGrid<CellLabels> &cellLabels)
{
	double dotValue = 0;
	forEachVoxelRange(Vec2i(0), cellLabels.size(), [&](const Vec2i &cell)
	{
		if (cellLabels(cell) == CellLabels::INTERIOR)
			dotValue += aVectorGrid(cell) * bVectorGrid(cell);
	});

	return dotValue;
}

void GeometricMGOperations::addToVector(UniformGrid<Real> &destination,
										const UniformGrid<Real> &source,
										const UniformGrid<CellLabels> &cellLabels,
										const Real scale)
{
	assert(destination.size() == source.size() &&
		source.size() == cellLabels.size());

	int totalVoxels = destination.size()[0] * destination.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destination.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
				destination(cell) += scale * source(cell);
		}
	});
}

void GeometricMGOperations::addToVector(UniformGrid<Real> &destination,
										const UniformGrid<Real> &source,
										const UniformGrid<Real> &scaledSource,
										const UniformGrid<CellLabels> &cellLabels,
										const Real scale)
{
	assert(destination.size() == source.size() &&
			source.size() == scaledSource.size() &&
			scaledSource.size() == cellLabels.size());

	int totalVoxels = destination.size()[0] * destination.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = destination.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
				destination(cell) = source(cell) + scale * scaledSource(cell);
		}
	});
}

double GeometricMGOperations::l2Norm(const UniformGrid<Real> &vectorGrid,
									const UniformGrid<CellLabels> &cellLabels)
{
	double squaredNorm = 0;
	forEachVoxelRange(Vec2i(0), cellLabels.size(), [&](const Vec2i &cell)
	{
		if (cellLabels(cell) == CellLabels::INTERIOR)
			squaredNorm += Util::sqr(vectorGrid(cell));
	});

	return std::sqrt(squaredNorm);
}

double GeometricMGOperations::lInfinityNorm(const UniformGrid<Real> &vectorGrid,
											const UniformGrid<CellLabels> &cellLabels)
{
	using ParallelReal = enumerable_thread_specific<Real>;

	ParallelReal parallelReal(Real(0));

	int totalVoxels = cellLabels.size()[0] * cellLabels.size()[1];

	parallel_for(blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const blocked_range<int> &range)
	{
		Real &localMaxError = parallelReal.local();

		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = cellLabels.unflatten(flatIndex);

			if (cellLabels(cell) == CellLabels::INTERIOR)
				localMaxError += std::max(fabs(vectorGrid(cell)), localMaxError);
		}
	});

	Real maxError = 0;

	parallelReal.combine_each([&maxError](const Real localMaxError)
	{
		maxError = std::max(maxError, localMaxError);
	});

	return maxError;
}