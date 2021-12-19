#include "GeometricMultigridOperators.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

namespace FluidSim2D::GeometricMultigridOperators
{

std::pair<double, double> computeLaplacian(const UniformGrid<double>& source,
	const UniformGrid<CellLabels>& domainCellLabels,
	const Vec2i& cell,
	const VectorGrid<double>* boundaryWeights = nullptr)
{
	assert(source.size() == domainCellLabels.size());

	if (boundaryWeights != nullptr)
	{
		assert(boundaryWeights->sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			boundaryWeights->size(0)[0] == source.size()[0] + 1 &&
			boundaryWeights->size(0)[1] == source.size()[1] &&
			boundaryWeights->size(1)[0] == source.size()[0] &&
			boundaryWeights->size(1)[1] == source.size()[1] + 1);
	}

	double laplacian = 0;
	double diagonal = 0;
	if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	{
		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < source.size()[axis]);

				assert(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

				if (boundaryWeights != nullptr)
				{
					Vec2i face = cellToFace(cell, axis, direction);
					assert((*boundaryWeights)(face, axis) == 1);
				}

				laplacian -= double(source(adjacentCell));
			}

		diagonal = 4;
	}
	else
	{
		assert(domainCellLabels(cell) == CellLabels::BOUNDARY_CELL);

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);
				Vec2i face = cellToFace(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < source.size()[axis]);

				auto adjacentLabel = domainCellLabels(adjacentCell);
				if (adjacentLabel == CellLabels::INTERIOR_CELL)
				{
					if (boundaryWeights != nullptr)
						assert((*boundaryWeights)(face, axis) == 1);

					laplacian -= double(source(adjacentCell));
					++diagonal;
				}
				else if (adjacentLabel == CellLabels::BOUNDARY_CELL)
				{
					double solutionValue = source(adjacentCell);

					if (boundaryWeights != nullptr)
					{
						double weight = (*boundaryWeights)(face, axis);

						laplacian -= weight * solutionValue;
						diagonal += weight;
					}
					else
					{
						laplacian -= solutionValue;
						++diagonal;
					}
				}
				else if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
				{
					if (boundaryWeights != nullptr)
						diagonal += (*boundaryWeights)(face, axis);
					else ++diagonal;
				}
				else
				{
					if (boundaryWeights != nullptr)
						assert((*boundaryWeights)(face, axis) == 0);
				}
			}
	}

	laplacian += diagonal * double(source(cell));
	return std::pair<double, double>(laplacian, diagonal);
}

void interiorJacobiPoissonSmoother(UniformGrid<double>& solution,
	const UniformGrid<double>& rhs,
	const UniformGrid<CellLabels>& domainCellLabels,
	const double dx,
	const VectorGrid<double>* boundaryWeights)

{
	assert(solution.size() == rhs.size() && solution.size() == domainCellLabels.size());

	UniformGrid<double> tempSolution = solution;

	const double gridScalar = 1. / std::pow(dx, 2);
	constexpr double dampedWeight = 2. / 3.;

	tbb::parallel_for(tbb::blocked_range<int>(0, solution.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			auto label = domainCellLabels(cell);
			if (label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL)
			{
				std::pair<double, double> laplacianResults = computeLaplacian(tempSolution, domainCellLabels, cell, boundaryWeights);

				double laplacian = laplacianResults.first;
				double diagonal = laplacianResults.second;

				if (label == CellLabels::INTERIOR_CELL)
					assert(diagonal == 4);
				else
					assert(diagonal > 0);

				double residual = double(rhs(cell)) - gridScalar * laplacian;
				residual /= (diagonal * gridScalar);

				solution(cell) += dampedWeight * residual;
			}
		}
	});
}

void boundaryJacobiPoissonSmoother(UniformGrid<double>& solution,
	const UniformGrid<double>& rhs,
	const UniformGrid<CellLabels>& domainCellLabels,
	const VecVec2i& boundaryCells,
	const double dx,
	const VectorGrid<double>* boundaryWeights)
{
	assert(solution.size() == rhs.size() && solution.size() == domainCellLabels.size());

	if (boundaryWeights != nullptr)
	{
		assert(boundaryWeights->sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			boundaryWeights->size(0)[0] == solution.size()[0] + 1 &&
			boundaryWeights->size(0)[1] == solution.size()[1] &&
			boundaryWeights->size(1)[0] == solution.size()[0] &&
			boundaryWeights->size(1)[1] == solution.size()[1] + 1);
	}

	const double gridScalar = 1. / std::pow(dx, 2);
	constexpr double dampedWeight = 2. / 3.;

	std::vector<double> tempSolution(boundaryCells.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, int(boundaryCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];

			auto label = domainCellLabels(cell);

			assert(label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL);
				
			std::pair<double, double> laplacianResult = computeLaplacian(solution, domainCellLabels, cell, boundaryWeights);

			double laplacian = laplacianResult.first;
			double diagonal = laplacianResult.second;

			if (label == CellLabels::INTERIOR_CELL)
				assert(diagonal == 4.);
			else
				assert(diagonal > 0);

			double residual = double(rhs(cell)) - gridScalar * laplacian;
			residual /= (diagonal * gridScalar);

			tempSolution[cellIndex] = double(solution(cell)) + dampedWeight * residual;
		}
	});

	tbb::parallel_for(tbb::blocked_range<int>(0, int(boundaryCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];
			solution(cell) = tempSolution[cellIndex];
		}
	});
}

void applyPoissonMatrix(UniformGrid<double>& destination,
	const UniformGrid<double>& source,
	const UniformGrid<CellLabels>& domainCellLabels,
	const double dx,
	const VectorGrid<double>* boundaryWeights)
{
	assert(destination.size() == source.size() && source.size() == domainCellLabels.size());

	if (boundaryWeights != nullptr)
	{
		assert(boundaryWeights->sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			boundaryWeights->size(0)[0] == destination.size()[0] + 1 &&
			boundaryWeights->size(0)[1] == destination.size()[1] &&
			boundaryWeights->size(1)[0] == destination.size()[0] &&
			boundaryWeights->size(1)[1] == destination.size()[1] + 1);
	}

	const double gridScalar = 1. / std::pow(dx, 2);

	tbb::parallel_for(tbb::blocked_range<int>(0, destination.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destination.unflatten(cellIndex);

			auto label = domainCellLabels(cell);
			if (label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL)
			{
				std::pair<double, double> laplacianResult = computeLaplacian(source, domainCellLabels, cell, boundaryWeights);

				double laplacian = laplacianResult.first;
				double diagonal = laplacianResult.second;

				if (label == CellLabels::INTERIOR_CELL)
					assert(diagonal == 4);
				else
					assert(diagonal > 0);

				destination(cell) = gridScalar * laplacian;
			}
		}
	});
}

void computePoissonResidual(UniformGrid<double>& residual,
	const UniformGrid<double>& solution,
	const UniformGrid<double>& rhs,
	const UniformGrid<CellLabels>& domainCellLabels,
	const double dx,
	const VectorGrid<double>* boundaryWeights)
{
	assert(residual.size() == solution.size() &&
		residual.size() == rhs.size() &&
		residual.size() == domainCellLabels.size());

	residual.reset(0);

	applyPoissonMatrix(residual, solution, domainCellLabels, dx, boundaryWeights);
	addVectors(residual, rhs, residual, domainCellLabels, -1);
}

void downsample(UniformGrid<double>& destinationGrid,
	const UniformGrid<double>& sourceGrid,
	const UniformGrid<CellLabels>& destinationCellLabels,
	const UniformGrid<CellLabels>& sourceCellLabels)
{
	constexpr double restrictionWeights[4] = { 1. / 8., 3. / 8., 3. / 8., 1. / 8. };

	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert((2 * destinationGrid.size()).eval() == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	for (int axis : {0, 1})
	{
		assert(destinationGrid.size()[axis] % 2 == 0);
		assert(sourceGrid.size()[axis] % 2 == 0);
	}

	destinationGrid.reset(0);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);
			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				destinationCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(destinationGrid(cell) == 0);

				// Iterator over source cells
				double sampleValue = 0;

				Vec2i startCell = 2 * cell - Vec2i::Ones();
				forEachVoxelRange(Vec2i::Zero(), Vec2i(4, 4), [&](const Vec2i& sampleIndex)
					{
						Vec2i sampleCell = startCell + sampleIndex;

						assert(sampleCell[0] >= 0 && sampleCell[0] < sourceGrid.size()[0] &&
							sampleCell[1] >= 0 && sampleCell[1] < sourceGrid.size()[1]);

						if (sourceCellLabels(sampleCell) == CellLabels::INTERIOR_CELL ||
							sourceCellLabels(sampleCell) == CellLabels::BOUNDARY_CELL)
							sampleValue += restrictionWeights[sampleIndex[0]] * restrictionWeights[sampleIndex[1]] * double(sourceGrid(sampleCell));
						else
							assert(sourceGrid(sampleCell) == 0);
					});

				destinationGrid(cell) = sampleValue;
			}
		}
	});
}

void upsampleAndAdd(UniformGrid<double>& destinationGrid,
	const UniformGrid<double>& sourceGrid,
	const UniformGrid<CellLabels>& destinationCellLabels,
	const UniformGrid<CellLabels>& sourceCellLabels)
{
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert((destinationGrid.size() / 2).eval() == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	assert(destinationGrid.size()[0] % 2 == 0 &&
		destinationGrid.size()[1] % 2 == 0 &&
		sourceGrid.size()[0] % 2 == 0 &&
		sourceGrid.size()[1] % 2 == 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				destinationCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				// Iterator over source cells	
				Vec2d samplePoint = .5 * (cell.cast<double>() + Vec2d(.5, .5)) - Vec2d(.5, .5);

				Vec2i startCell = samplePoint.cast<int>();

				Vec2d interpWeight = samplePoint - startCell.cast<double>();

				double values[2][2];
				for (int xOffset : {0, 1})
					for (int yOffset : {0, 1})
					{
						Vec2i fineCell = startCell + Vec2i(xOffset, yOffset);
						if (sourceCellLabels(fineCell) != CellLabels::INTERIOR_CELL && sourceCellLabels(fineCell) != CellLabels::BOUNDARY_CELL)
						{
							assert(sourceGrid(startCell + Vec2i(xOffset, yOffset)) == 0);
							values[xOffset][yOffset] = 0;
						}
						else
							values[xOffset][yOffset] = sourceGrid(fineCell);
					}

				destinationGrid(cell) += bilerp(values[0][0], values[1][0], values[0][1], values[1][1], interpWeight[0], interpWeight[1]);
			}
		}
	});
}

double dotProduct(const UniformGrid<double>& vectorGridA,
	const UniformGrid<double>& vectorGridB,
	const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGridA.size() == vectorGridB.size() && vectorGridB.size() == domainCellLabels.size());

	double result = tbb::parallel_deterministic_reduce(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), double(0), 
	[&](const tbb::blocked_range<int>& range, double result) -> double
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				result += vectorGridA(cell) * vectorGridB(cell);
		}

		return result;
	},
	[](double a, double b) -> double
	{
		return a + b;
	});

	return result;
}

void addToVector(UniformGrid<double>& destination,
	const UniformGrid<double>& source,
	const UniformGrid<CellLabels>& domainCellLabels,
	const double scale)
{
	assert(destination.size() == source.size() && source.size() == domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, destination.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destination.unflatten(cellIndex);
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				destination(cell) += scale * source(cell);
		}
	});
}

void addVectors(UniformGrid<double>& destination,
	const UniformGrid<double>& source,
	const UniformGrid<double>& scaledSource,
	const UniformGrid<CellLabels>& domainCellLabels,
	const double scale)
{
	assert(destination.size() == source.size() &&
		source.size() == scaledSource.size() &&
		scaledSource.size() == domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, destination.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destination.unflatten(cellIndex);
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				destination(cell) = source(cell) + scale * scaledSource(cell);
		}
	});
}

double squaredl2Norm(const UniformGrid<double>& vectorGrid,
	const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGrid.size() == domainCellLabels.size());

	double squaredNorm = tbb::parallel_deterministic_reduce(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), double(0),
	[&](const tbb::blocked_range<int>& range, double squaredNorm) -> double
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				squaredNorm += std::pow(vectorGrid(cell), 2);
		}

		return squaredNorm;
	},
	[](double a, double b) -> double
	{
		return a + b;
	});

	return squaredNorm;
}

double lInfinityNorm(const UniformGrid<double>& vectorGrid,
	const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGrid.size() == domainCellLabels.size());

	double infNorm = tbb::parallel_reduce(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), double(0),
	[&](const tbb::blocked_range<int>& range, double infNorm) -> double
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				infNorm = std::max(std::fabs(vectorGrid(cell)), infNorm);
		}

		return infNorm;
	},
	[](double a, double b) -> double
	{
		return std::max(a, b);
	});

	return infNorm;
}

void setBoundaryDomainLabels(UniformGrid<CellLabels>& domainCellLabels,
	const VectorGrid<double>& boundaryWeights)
{
	assert(boundaryWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
		boundaryWeights.size(0)[0] == domainCellLabels.size()[0] + 1 &&
		boundaryWeights.size(0)[1] == domainCellLabels.size()[1] &&
		boundaryWeights.size(1)[0] == domainCellLabels.size()[0] &&
		boundaryWeights.size(1)[1] == domainCellLabels.size()[1] + 1);

	tbb::enumerable_thread_specific<VecVec2i> parallelBoundaryCells;

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localBoundaryCells = parallelBoundaryCells.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				bool isBoundary = false;
				for (int axis = 0; axis < 2 && !isBoundary; ++axis)
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.size()[axis]);

						auto adjacentLabel = domainCellLabels(adjacentCell);
						if (adjacentLabel == CellLabels::EXTERIOR_CELL ||
							adjacentLabel == CellLabels::DIRICHLET_CELL)
						{
							isBoundary = true;
							break;
						}

						Vec2i face = cellToFace(cell, axis, direction);

						if (boundaryWeights(face, axis) != 1)
						{
							isBoundary = true;
							break;
						}
					}

				if (isBoundary)
					localBoundaryCells.push_back(cell);
			}
			else assert(domainCellLabels(cell) != CellLabels::BOUNDARY_CELL);
		}
	});

	// Combine parallel list of boundary cells
	VecVec2i boundaryCells;
	mergeLocalThreadVectors(boundaryCells, parallelBoundaryCells);

	tbb::parallel_for(tbb::blocked_range<int>(0, int(boundaryCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];

			assert(domainCellLabels(cell) == CellLabels::INTERIOR_CELL);

			domainCellLabels(cell) = CellLabels::BOUNDARY_CELL;
		}
	});
}

void buildExpandedBoundaryWeights(VectorGrid<double>& expandedBoundaryWeights,
	const VectorGrid<double>& baseBoundaryWeights,
	const UniformGrid<CellLabels>& expandedDomainCellLabels,
	const Vec2i& exteriorOffset,
	int axis)
{
	assert(expandedBoundaryWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
		baseBoundaryWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED);

	assert(expandedBoundaryWeights.size(axis)[0] >= baseBoundaryWeights.size(axis)[0] + exteriorOffset[0] &&
		expandedBoundaryWeights.size(axis)[1] >= baseBoundaryWeights.size(axis)[1] + exteriorOffset[1]);

	expandedBoundaryWeights.grid(axis).reset(0);

	tbb::parallel_for(tbb::blocked_range<int>(0, baseBoundaryWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
		{
			Vec2i face = baseBoundaryWeights.grid(axis).unflatten(faceIndex);

			if (baseBoundaryWeights(face, axis) > 0)
			{
				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				assert(expandedDomainCellLabels(backwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL &&
					expandedDomainCellLabels(forwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL);

				Vec2i expandedFace = face + exteriorOffset;

				expandedBoundaryWeights(expandedFace, axis) = baseBoundaryWeights(face, axis);
			}
		}
	});
}

Vec2i getChildCell(const Vec2i& cell, const int childIndex)
{
	assert(childIndex < 4);

	Vec2i childCell(cell);
	childCell *= 2;
	for (int axis : {0, 1})
	{
		if (childIndex & (1 << axis))
			++childCell[axis];
	}

	return childCell;
}

UniformGrid<CellLabels> buildCoarseCellLabels(const UniformGrid<CellLabels>& sourceCellLabels)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 && sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<CellLabels> destinationCellLabels(sourceCellLabels.size() / 2, CellLabels::EXTERIOR_CELL);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			// Iterate over the destination cell's children.
			Vec2i childCell = 2 * cell;

			bool hasDirichletChild = false;
			bool hasInteriorChild = false;

			for (int adjacentCellIndex = 0; adjacentCellIndex < 4; ++adjacentCellIndex)
			{
				Vec2i localCell = cellToNode(childCell, adjacentCellIndex);

				if (sourceCellLabels(localCell) == CellLabels::DIRICHLET_CELL)
				{
					hasDirichletChild = true;
					break;
				}
				else if (sourceCellLabels(localCell) == CellLabels::INTERIOR_CELL ||
					sourceCellLabels(localCell) == CellLabels::BOUNDARY_CELL)
					hasInteriorChild = true;
			}

			if (hasDirichletChild)
				destinationCellLabels(cell) = CellLabels::DIRICHLET_CELL;
			else if (hasInteriorChild)
				destinationCellLabels(cell) = CellLabels::INTERIOR_CELL;
			else
				assert(destinationCellLabels(cell) == CellLabels::EXTERIOR_CELL);
		}
	});

	// Set boundary cells
	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				bool hasBoundary = false;

				for (int axis = 0; axis < 2 && !hasBoundary; ++axis)
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destinationCellLabels.size()[axis]);

						if (destinationCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL ||
							destinationCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
						{
							hasBoundary = true;
							break;
						}
					}

				if (hasBoundary)
					destinationCellLabels(cell) = CellLabels::BOUNDARY_CELL;
			}
		}
	});

	return destinationCellLabels;
}

VecVec2i buildBoundaryCells(const UniformGrid<CellLabels>& sourceCellLabels,
										int boundaryWidth)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 && sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<VisitedCellLabels> visitedCells(sourceCellLabels.size(), VisitedCellLabels::UNVISITED_CELL);

	VecVec2i boundaryCells;	
	tbb::enumerable_thread_specific<VecVec2i> parallelBoundaryCells;

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, sourceCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localBoundaryCells = parallelBoundaryCells.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = sourceCellLabels.unflatten(cellIndex);

			if (sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				localBoundaryCells.push_back(cell);
		}
	});

	mergeLocalThreadVectors(boundaryCells, parallelBoundaryCells);
	
	for (int layer = 1; layer < boundaryWidth; ++layer)
	{
		// Set cells to visited
		tbb::parallel_for(tbb::blocked_range<int>(0, int(boundaryCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundaryCells[cellIndex];
				visitedCells(cell) = VisitedCellLabels::FINISHED_CELL;
			}
		});

		if (layer < boundaryWidth - 1)
		{
			parallelBoundaryCells.clear();

			tbb::parallel_for(tbb::blocked_range<int>(0, int(boundaryCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto &localBoundaryCells = parallelBoundaryCells.local();

				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = boundaryCells[cellIndex];

					assert(visitedCells(cell) == VisitedCellLabels::FINISHED_CELL);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (sourceCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL &&
								visitedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
							{
								localBoundaryCells.push_back(adjacentCell);
							}
						}
				}
			});
			//
			// Collect new layer
			//

			boundaryCells.clear();
			mergeLocalThreadVectors(boundaryCells, parallelBoundaryCells);
		}
	}

	parallelBoundaryCells.clear();

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, sourceCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto &localBoundaryCells = parallelBoundaryCells.local();
		
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = visitedCells.unflatten(cellIndex);

			if (visitedCells(cell) == VisitedCellLabels::FINISHED_CELL)
			{
				assert(sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL ||
						sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);

				localBoundaryCells.push_back(cell);
			}
		}
	});

	boundaryCells.clear();
	mergeLocalThreadVectors(boundaryCells, parallelBoundaryCells);


	tbb::parallel_sort(boundaryCells.begin(), boundaryCells.end(), [](const Vec2i& vec0, const Vec2i& vec1)
	{
		return std::tie(vec0[0], vec0[1]) < std::tie(vec1[0], vec1[1]);
	});

	return boundaryCells;
}

std::pair<Vec2i, int> buildExpandedDomainLabels(UniformGrid<CellLabels>& expandedDomainCellLabels,
												const UniformGrid<CellLabels>& baseDomainCellLabels)
{
	// Build domain labels with the appropriate padding to apply
	// geometric multigrid directly without a wasteful transfer
	// for each v-cycle.

	// Cap MG levels at 4 voxels in the smallest dimension
	double minLog = std::min(std::log2(baseDomainCellLabels.size()[0]),
								std::log2(baseDomainCellLabels.size()[1]));

	int mgLevels = int(std::ceil(minLog) - std::log2(2));

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = int(std::pow(2, mgLevels - 1));

	Vec2i expandedGridSize = baseDomainCellLabels.size() + 2 * Vec2i(exteriorPadding, exteriorPadding);

	for (int axis : {0, 1})
	{
		double logSize = std::log2(expandedGridSize[axis]);
		logSize = std::ceil(logSize);

		expandedGridSize[axis] = int(std::exp2(logSize));
	}

	Vec2i exteriorOffset(exteriorPadding, exteriorPadding);

	expandedDomainCellLabels.resize(expandedGridSize, CellLabels::EXTERIOR_CELL);

	// Copy initial domain labels to interior domain labels with padding
	tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = baseDomainCellLabels.unflatten(cellIndex);
			auto baseLabel = baseDomainCellLabels(cell);
			if (baseLabel != CellLabels::EXTERIOR_CELL)
			{
				Vec2i expandedCell = cell + exteriorOffset;
				expandedDomainCellLabels(expandedCell) = baseLabel;
			}
		}
	});

	return std::pair<Vec2i, int>(exteriorOffset, mgLevels);
}

bool unitTestCoarsening(const UniformGrid<CellLabels>& coarseCellLabels,
						const UniformGrid<CellLabels>& fineCellLabels)
{
	// The coarse cell grid must be exactly have the size of the fine cell grid.
	if ((2 * coarseCellLabels.size()).eval() != fineCellLabels.size())
		return false;

	if (coarseCellLabels.size()[0] % 2 != 0 ||
		coarseCellLabels.size()[1] % 2 != 0 ||
		fineCellLabels.size()[0] % 2 != 0 ||
		fineCellLabels.size()[1] % 2 != 0)
		return false;

	{
		bool testPassed = true;

		tbb::parallel_for(tbb::blocked_range<int>(0, fineCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			if (!testPassed) return;

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i fineCell = fineCellLabels.unflatten(cellIndex);
				Vec2i coarseCell = fineCell / 2;

				// If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
				if (fineCellLabels(fineCell) == CellLabels::DIRICHLET_CELL)
				{
					if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET_CELL)
					{
						testPassed = false;
						return;
					}
				}
				else if (fineCellLabels(fineCell) == CellLabels::INTERIOR_CELL ||
					fineCellLabels(fineCell) == CellLabels::BOUNDARY_CELL)
				{
					// If the fine cell is interior, the coarse cell can be either
					// interior or Dirichlet (if a sibling cell is Dirichlet).
					if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR_CELL)
					{
						testPassed = false;
						return;
					}
				}
			}
		});

		if(!testPassed) return false;
	}
	{
		bool testPassed = true;

		tbb::parallel_for(tbb::blocked_range<int>(0, coarseCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			if (!testPassed) return;

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coarseCell = coarseCellLabels.unflatten(cellIndex);

				bool foundDirichletChild = false;
				bool foundInteriorChild = false;
				bool foundExteriorChild = false;

				for (int childIndex = 0; childIndex < 4; ++childIndex)
				{
					Vec2i fineCell = getChildCell(coarseCell, childIndex);

					auto fineLabel = fineCellLabels(fineCell);

					if (fineLabel == CellLabels::DIRICHLET_CELL)
						foundDirichletChild = true;
					else if (fineLabel == CellLabels::INTERIOR_CELL ||
						fineLabel == CellLabels::BOUNDARY_CELL)
						foundInteriorChild = true;
					else if (fineLabel == CellLabels::EXTERIOR_CELL)
						foundExteriorChild = true;
				}

				auto coarseLabel = coarseCellLabels(coarseCell);
				if (coarseLabel == CellLabels::DIRICHLET_CELL)
				{
					if (!foundDirichletChild)
						testPassed = false;
				}
				else if (coarseLabel == CellLabels::INTERIOR_CELL ||
					coarseLabel == CellLabels::BOUNDARY_CELL)
				{
					if (foundDirichletChild || !foundInteriorChild)
						testPassed = false;
				}
				else if (coarseLabel == CellLabels::EXTERIOR_CELL)
				{
					if (foundDirichletChild || foundInteriorChild || !foundExteriorChild)
						testPassed = false;
				}
			}
		});
		
		if(!testPassed) return false;
	}

	return true;
}

bool unitTestExteriorCells(const UniformGrid<CellLabels>& cellLabels)
{
	Vec2i startCell = Vec2i::Zero();
	Vec2i endCell = cellLabels.size();

	bool exteriorCellTestPassed = true;
	for (int axis : {0, 1})
		for (int direction : {0, 1})
		{
			Vec2i localStartCell = startCell;
			Vec2i localEndCell = endCell;

			if (direction == 0)
				localEndCell[axis] = 1;
			else
				localStartCell[axis] = endCell[axis] - 1;

			forEachVoxelRange(localStartCell, localEndCell, [&](const Vec2i& cell)
			{
				if (!exteriorCellTestPassed) return;

				if (cellLabels(cell) != CellLabels::EXTERIOR_CELL)
					exteriorCellTestPassed = false;
			});
		}

	return exteriorCellTestPassed;
}

bool unitTestBoundaryCells(const UniformGrid<CellLabels>& cellLabels,
	const VectorGrid<double>* boundaryWeights)
{
	if (boundaryWeights != nullptr)
	{
		assert(boundaryWeights->sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			boundaryWeights->size(0)[0] == cellLabels.size()[0] + 1 &&
			boundaryWeights->size(0)[1] == cellLabels.size()[1] &&
			boundaryWeights->size(1)[0] == cellLabels.size()[0] &&
			boundaryWeights->size(1)[1] == cellLabels.size()[1] + 1);
	}

	bool boundaryCellTestPassed = true;

	forEachVoxelRange(Vec2i::Zero(), cellLabels.size(), [&](const Vec2i& cell)
	{
		if (!boundaryCellTestPassed) return;

		if (cellLabels(cell) == CellLabels::INTERIOR_CELL)
		{
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					if (!(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
						cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL))
					{
						boundaryCellTestPassed = false;
						return;
					}
				}
		}
		else if (cellLabels(cell) == CellLabels::BOUNDARY_CELL)
		{
			bool hasValidBoundary = false;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					if (!(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
						cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL))
						hasValidBoundary = true;
					else if (boundaryWeights != nullptr)
					{
						Vec2i face = cellToFace(cell, axis, direction);
						if ((*boundaryWeights)(face, axis) != 1 && cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
							hasValidBoundary = true;
					}
				}

			if (!hasValidBoundary)
			{
				boundaryCellTestPassed = false;
				return;
			}
		}
	});

	return boundaryCellTestPassed;
}

}