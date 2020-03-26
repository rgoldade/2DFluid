#ifndef GEOMETRIC_MULTIGRID_OPERATORS_H
#define GEOMETRIC_MULTIGRID_OPERATORS_H

#include <utility>

#include "tbb/tbb.h"

#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim2D::SimTools::GeometricMultigridOperators
{

using namespace Utilities;

enum class CellLabels { INTERIOR_CELL, EXTERIOR_CELL, DIRICHLET_CELL, BOUNDARY_CELL };

template<typename SolveReal, typename StoreReal>
void interiorJacobiPoissonSmoother(UniformGrid<StoreReal>& solution,
									const UniformGrid<StoreReal>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const SolveReal dx,
									const VectorGrid<StoreReal>* boundaryWeights = nullptr);

template<typename SolveReal, typename StoreReal>
void boundaryJacobiPoissonSmoother(UniformGrid<StoreReal>& solution,
									const UniformGrid<StoreReal>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const std::vector<Vec2i>& boundaryCells,
									const SolveReal dx,
									const VectorGrid<StoreReal>* boundaryWeights = nullptr);

template<typename SolveReal, typename StoreReal>
void applyPoissonMatrix(UniformGrid<StoreReal>& destination,
						const UniformGrid<StoreReal>& source,
						const UniformGrid<CellLabels>& domainCellLabels,
						const SolveReal dx,
						const VectorGrid<StoreReal>* boundaryWeights = nullptr);

template<typename SolveReal, typename StoreReal>
void computePoissonResidual(UniformGrid<StoreReal>& residual,
							const UniformGrid<StoreReal>& solution,
							const UniformGrid<StoreReal>& rhs,
							const UniformGrid<CellLabels>& domainCellLabels,
							const SolveReal dx,
							const VectorGrid<StoreReal>* boundaryWeights = nullptr);

template<typename SolveReal, typename StoreReal>
void downsample(UniformGrid<StoreReal>& destinationGrid,
				const UniformGrid<StoreReal>& sourceGrid,
				const UniformGrid<CellLabels>& destinationCellLabels,
				const UniformGrid<CellLabels>& sourceCellLabels);

template<typename SolveReal, typename StoreReal>
void upsampleAndAdd(UniformGrid<StoreReal>& destinationGrid,
					const UniformGrid<StoreReal>& sourceGrid,
					const UniformGrid<CellLabels>& destinationCellLabels,
					const UniformGrid<CellLabels>& sourceCellLabels);

template<typename SolveReal, typename StoreReal>
SolveReal dotProduct(const UniformGrid<StoreReal>& vectorA,
						const UniformGrid<StoreReal>& vectorB,
						const UniformGrid<CellLabels>& domainCellLabels);

template<typename SolveReal, typename StoreReal>
void addToVector(UniformGrid<StoreReal>& destination,
					const UniformGrid<StoreReal>& source,
					const UniformGrid<CellLabels>& domainCellLabels,
					const SolveReal scale);

template<typename SolveReal, typename StoreReal>
void addVectors(UniformGrid<StoreReal>& destination,
				const UniformGrid<StoreReal>& source,
				const UniformGrid<StoreReal>& scaledSource,
				const UniformGrid<CellLabels>& domainCellLabels,
				const SolveReal scale);

template<typename SolveReal, typename StoreReal>
SolveReal squaredl2Norm(const UniformGrid<StoreReal>& vectorGrid,
						const UniformGrid<CellLabels>& domainCellLabels);

template<typename SolveReal, typename StoreReal>
SolveReal lInfinityNorm(const UniformGrid<StoreReal>& vectorGrid,
						const UniformGrid<CellLabels>& domainCellLabels);

UniformGrid<CellLabels> buildCoarseCellLabels(const UniformGrid<CellLabels>& sourceCellLabels);

std::vector<Vec2i> buildBoundaryCells(const UniformGrid<CellLabels>& sourceCellLabels, int boundaryWidth);

std::pair<Vec2i, int> buildExpandedDomainLabels(UniformGrid<CellLabels>& expandedDomainCellLabels,
												const UniformGrid<CellLabels>& baseDomainCellLabels);

template<typename CustomLabel,
			typename IsExteriorFunctor,
			typename IsDirichletFunctor,
			typename IsInteriorFunctor>
std::pair<Vec2i, int> buildCustomExpandedDomainLabels(UniformGrid<CellLabels>& expandedDomainCellLabels,
														const UniformGrid<CustomLabel>& baseCustomLabels,
														const IsExteriorFunctor& isExteriorFunctor,
														const IsDirichletFunctor& isDirichletFunctor,
														const IsInteriorFunctor& isInteriorFunctor);

template<typename StoreReal>
void buildExpandedBoundaryWeights(VectorGrid<StoreReal>& expandedBoundaryWeights,
									const VectorGrid<StoreReal>& baseBoundaryWeights,
									const UniformGrid<CellLabels>& expandedDomainCellLabels,
									const Vec2i& exteriorOffset,
									const int axis);

template<typename StoreReal>
void setBoundaryDomainLabels(UniformGrid<CellLabels>& sourceCellLabels,
								const VectorGrid<StoreReal>& boundaryWeights);

bool unitTestCoarsening(const UniformGrid<CellLabels>& coarseCellLabels,
						const UniformGrid<CellLabels>& fineCellLabels);

template<typename StoreReal>
bool unitTestBoundaryCells(const UniformGrid<CellLabels>& domainCellLabels, const VectorGrid<StoreReal>* boundaryWeights = nullptr);

bool unitTestExteriorCells(const UniformGrid<CellLabels>& domainCellLabels);

// Implementation of template functions

template<typename SolveReal, typename StoreReal>
std::pair<SolveReal, SolveReal> computeLaplacian(const UniformGrid<StoreReal>& source,
													const UniformGrid<CellLabels>& domainCellLabels,
													const Vec2i& cell,
													const VectorGrid<StoreReal>* boundaryWeights = nullptr)
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

	SolveReal laplacian = 0;
	SolveReal diagonal = 0;
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

				laplacian -= SolveReal(source(adjacentCell));
			}

		diagonal = 4.;
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

					laplacian -= SolveReal(source(adjacentCell));
					++diagonal;
				}
				else if (adjacentLabel == CellLabels::BOUNDARY_CELL)
				{
					SolveReal solutionValue = source(adjacentCell);

					if (boundaryWeights != nullptr)
					{
						SolveReal weight = (*boundaryWeights)(face, axis);

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

	laplacian += diagonal * SolveReal(source(cell));
	return std::pair<SolveReal, SolveReal>(laplacian, diagonal);
}

template<typename SolveReal, typename StoreReal>
void interiorJacobiPoissonSmoother(UniformGrid<StoreReal>& solution,
									const UniformGrid<StoreReal>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const SolveReal dx,
									const VectorGrid<StoreReal>* boundaryWeights)

{
	assert(solution.size() == rhs.size() && solution.size() == domainCellLabels.size());

	UniformGrid<StoreReal> tempSolution = solution;

	const SolveReal gridScalar = 1. / sqr(dx);
	constexpr SolveReal dampedWeight = 2. / 3.;

	tbb::parallel_for(tbb::blocked_range<int>(0, solution.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);
				
			auto label = domainCellLabels(cell);
			if (label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL)
			{
				std::pair<SolveReal, SolveReal> laplacianResults = computeLaplacian<SolveReal>(tempSolution, domainCellLabels, cell, boundaryWeights);

				SolveReal laplacian = laplacianResults.first;
				SolveReal diagonal = laplacianResults.second;
					
				if (label == CellLabels::INTERIOR_CELL)
					assert(diagonal == 4.);
				else
					assert(diagonal > 0);

				SolveReal residual = SolveReal(rhs(cell)) - gridScalar * laplacian;
				residual /= (diagonal * gridScalar);

				solution(cell) += dampedWeight * residual;
			}
		}
	});
}

template<typename SolveReal, typename StoreReal>
void boundaryJacobiPoissonSmoother(UniformGrid<StoreReal>& solution,
									const UniformGrid<StoreReal>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const std::vector<Vec2i>& boundaryCells,
									const SolveReal dx,
									const VectorGrid<StoreReal>* boundaryWeights)
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

	const SolveReal gridScalar = 1. / sqr(dx);
	constexpr SolveReal dampedWeight = 2. / 3.;

	std::vector<StoreReal> tempSolution(boundaryCells.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, boundaryCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];

			auto label = domainCellLabels(cell);

			assert(label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL);
				
			std::pair<SolveReal, SolveReal> laplacianResult = computeLaplacian<SolveReal>(solution, domainCellLabels, cell, boundaryWeights);

			SolveReal laplacian = laplacianResult.first;
			SolveReal diagonal = laplacianResult.second;

			if (label == CellLabels::INTERIOR_CELL)
				assert(diagonal == 4.);
			else
				assert(diagonal > 0);

			SolveReal residual = SolveReal(rhs(cell)) - gridScalar * laplacian;
			residual /= (diagonal * gridScalar);

			tempSolution[cellIndex] = SolveReal(solution(cell)) + dampedWeight * residual;
		}
	});

	tbb::parallel_for(tbb::blocked_range<int>(0, boundaryCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCells[cellIndex];
			solution(cell) = tempSolution[cellIndex];
		}
	});
}


template<typename SolveReal, typename StoreReal>
void applyPoissonMatrix(UniformGrid<StoreReal>& destination,
						const UniformGrid<StoreReal>& source,
						const UniformGrid<CellLabels>& domainCellLabels,
						const SolveReal dx,
						const VectorGrid<StoreReal>* boundaryWeights)
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

	const SolveReal gridScalar = 1. / sqr(dx);

	tbb::parallel_for(tbb::blocked_range<int>(0, destination.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destination.unflatten(cellIndex);

			auto label = domainCellLabels(cell);
			if (label == CellLabels::INTERIOR_CELL || label == CellLabels::BOUNDARY_CELL)
			{
				std::pair<SolveReal, SolveReal> laplacianResult = computeLaplacian<SolveReal>(source, domainCellLabels, cell, boundaryWeights);

				SolveReal laplacian = laplacianResult.first;
				SolveReal diagonal = laplacianResult.second;

				if (label == CellLabels::INTERIOR_CELL)
					assert(diagonal == 4.);
				else
					assert(diagonal > 0);

				destination(cell) = gridScalar * laplacian;
			}
		}
	});
}

template<typename SolveReal, typename StoreReal>
void computePoissonResidual(UniformGrid<StoreReal>& residual,
							const UniformGrid<StoreReal>& solution,
							const UniformGrid<StoreReal>& rhs,
							const UniformGrid<CellLabels>& domainCellLabels,
							const SolveReal dx,
							const VectorGrid<StoreReal>* boundaryWeights)
{
	assert(residual.size() == solution.size() &&
			residual.size() == rhs.size() &&
			residual.size() == domainCellLabels.size());

	residual.reset(0);

	applyPoissonMatrix<SolveReal>(residual, solution, domainCellLabels, dx, boundaryWeights);
	addVectors<SolveReal>(residual, rhs, residual, domainCellLabels, -1);
}

template<typename SolveReal, typename StoreReal>
void downsample(UniformGrid<StoreReal>& destinationGrid,
				const UniformGrid<StoreReal>& sourceGrid,
				const UniformGrid<CellLabels>& destinationCellLabels,
				const UniformGrid<CellLabels>& sourceCellLabels)
{
	constexpr SolveReal restrictionWeights[4] = { 1. / 8., 3. / 8., 3. / 8., 1. / 8. };

	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(2 * destinationGrid.size() == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	for (int axis : {0, 1})
	{
		assert(destinationGrid.size()[axis] % 2 == 0);
		assert(sourceGrid.size()[axis] % 2 == 0);
	}

	destinationGrid.reset(0);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				destinationCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(destinationGrid(cell) == 0);

				// Iterator over source cells
				SolveReal sampleValue = 0;

				Vec2i startCell = 2 * cell - Vec2i(1);
				forEachVoxelRange(Vec2i(0), Vec2i(4), [&](const Vec2i& sampleIndex)
				{
					Vec2i sampleCell = startCell + sampleIndex;

					assert(sampleCell[0] >= 0 && sampleCell[0] < sourceGrid.size()[0] &&
							sampleCell[1] >= 0 && sampleCell[1] < sourceGrid.size()[1]);

					if (sourceCellLabels(sampleCell) == CellLabels::INTERIOR_CELL ||
						sourceCellLabels(sampleCell) == CellLabels::BOUNDARY_CELL)
						sampleValue += restrictionWeights[sampleIndex[0]] * restrictionWeights[sampleIndex[1]] * SolveReal(sourceGrid(sampleCell));
					else
						assert(sourceGrid(sampleCell) == 0);
				});

				destinationGrid(cell) = sampleValue;
			}
		}
	});
}

template<typename SolveReal, typename StoreReal>
void upsampleAndAdd(UniformGrid<StoreReal>& destinationGrid,
					const UniformGrid<StoreReal>& sourceGrid,
					const UniformGrid<CellLabels>& destinationCellLabels,
					const UniformGrid<CellLabels>& sourceCellLabels)
{
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(destinationGrid.size() / 2 == sourceGrid.size());
	assert(destinationGrid.size() == destinationCellLabels.size());
	assert(sourceGrid.size() == sourceCellLabels.size());

	assert(destinationGrid.size()[0] % 2 == 0 &&
			destinationGrid.size()[1] % 2 == 0 &&
			sourceGrid.size()[0] % 2 == 0 &&
			sourceGrid.size()[1] % 2 == 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				destinationCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				// Iterator over source cells	
				SolveReal sampleValue = 0;

				Vec<2, SolveReal> samplePoint = .5 * (Vec<2, SolveReal>(cell) + Vec<2, SolveReal>(.5)) - Vec<2, SolveReal>(.5);

				Vec2i startCell = Vec2i(samplePoint);

				Vec<2, SolveReal> interpWeight = samplePoint - Vec<2, SolveReal>(startCell);

				SolveReal values[2][2];
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

template<typename SolveReal, typename StoreReal>
SolveReal dotProduct(const UniformGrid<StoreReal>& vectorGridA,
						const UniformGrid<StoreReal>& vectorGridB,
						const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGridA.size() == vectorGridB.size() && vectorGridB.size() == domainCellLabels.size());

	tbb::enumerable_thread_specific<SolveReal> parallelReal(SolveReal(0));

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		SolveReal& localDotProduct = parallelReal.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				localDotProduct += SolveReal(vectorGridA(cell)) * SolveReal(vectorGridB(cell));
		}
	});

	SolveReal dotProduct = 0;

	parallelReal.combine_each([&dotProduct](const SolveReal localDotProduct)
	{
		dotProduct += localDotProduct;
	});

	return dotProduct;
}

template<typename SolveReal, typename StoreReal>
void addToVector(UniformGrid<StoreReal>& destination,
					const UniformGrid<StoreReal>& source,
					const UniformGrid<CellLabels>& domainCellLabels,
					const SolveReal scale)
{
	assert(destination.size() == source.size() && source.size() == domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, destination.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destination.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				destination(cell) += scale * SolveReal(source(cell));
		}
	});
}

template<typename SolveReal, typename StoreReal>
void addVectors(UniformGrid<StoreReal>& destination,
				const UniformGrid<StoreReal>& source,
				const UniformGrid<StoreReal>& scaledSource,
				const UniformGrid<CellLabels>& domainCellLabels,
				const SolveReal scale)
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
				destination(cell) = SolveReal(source(cell)) + scale * SolveReal(scaledSource(cell));
		}
	});
}

template<typename SolveReal, typename StoreReal>
SolveReal squaredl2Norm(const UniformGrid<StoreReal>& vectorGrid,
	const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGrid.size() == domainCellLabels.size());

	tbb::enumerable_thread_specific<SolveReal> parallelReal(SolveReal(0));

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		SolveReal& localSqrNorm = parallelReal.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				localSqrNorm += sqr(SolveReal(vectorGrid(cell)));
		}
	});

	SolveReal squaredNorm = 0;

	parallelReal.combine_each([&squaredNorm](const SolveReal localSqrNorm)
	{
		squaredNorm += localSqrNorm;
	});

	return squaredNorm;
}

template<typename SolveReal, typename StoreReal>
SolveReal lInfinityNorm(const UniformGrid<StoreReal>& vectorGrid,
						const UniformGrid<CellLabels>& domainCellLabels)
{
	assert(vectorGrid.size() == domainCellLabels.size());

	tbb::enumerable_thread_specific<SolveReal> parallelReal(SolveReal(0));

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		SolveReal& localMaxError = parallelReal.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				localMaxError = std::max(std::fabs(SolveReal(vectorGrid(cell))), localMaxError);
		}
	});

	SolveReal maxError = 0;

	parallelReal.combine_each([&maxError](const SolveReal localMaxError)
	{
		maxError = std::max(maxError, localMaxError);
	});

	return maxError;
}

template<typename CustomLabel,
			typename IsExteriorFunctor,
			typename IsDirichletFunctor,
			typename IsInteriorFunctor>
std::pair<Vec2i, int> buildCustomExpandedDomainLabels(UniformGrid<CellLabels>& expandedDomainCellLabels,
														const UniformGrid<CustomLabel>& baseCustomLabels,
														const IsExteriorFunctor& isExteriorFunctor,
														const IsDirichletFunctor& isDirichletFunctor,
														const IsInteriorFunctor& isInteriorFunctor)
{
	// Build domain labels with the appropriate padding to apply
	// geometric multigrid directly without a wasteful transfer
	// for each v-cycle.

	// Cap MG levels at 4 voxels in the smallest dimension
	float minLog = std::min(std::log2(float(baseCustomLabels.size()[0])),
							std::log2(float(baseCustomLabels.size()[1])));

	int mgLevels = std::ceil(minLog) - std::log2(float(2));

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, mgLevels - 1);

	Vec2i expandedGridSize = baseCustomLabels.size() + 2 * Vec2i(exteriorPadding);

	for (int axis : {0, 1})
	{
		float logSize = std::log2(float(expandedGridSize[axis]));
		logSize = std::ceil(logSize);

		expandedGridSize[axis] = std::exp2(logSize);
	}

	Vec2i exteriorOffset = Vec2i(exteriorPadding);

	expandedDomainCellLabels.resize(expandedGridSize, CellLabels::EXTERIOR_CELL);

	// Copy initial domain labels to interior domain labels with padding
	tbb::parallel_for(tbb::blocked_range<int>(0, baseCustomLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = baseCustomLabels.unflatten(cellIndex);

			auto baseLabel = baseCustomLabels(cell);
			if (!isExteriorFunctor(baseLabel))
			{
				Vec2i expandedCell = cell + exteriorOffset;
				if (isInteriorFunctor(baseLabel))
					expandedDomainCellLabels(expandedCell) = CellLabels::INTERIOR_CELL;
				else
				{
					assert(isDirichletFunctor(baseLabel));
					expandedDomainCellLabels(expandedCell) = CellLabels::DIRICHLET_CELL;
				}
			}
		}
	});

	return std::pair<Vec2i, int>(exteriorOffset, mgLevels);
}

template<typename StoreReal>
void setBoundaryDomainLabels(UniformGrid<CellLabels>& domainCellLabels,
								const VectorGrid<StoreReal>& boundaryWeights)
{
	assert(boundaryWeights.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			boundaryWeights.size(0)[0] == domainCellLabels.size()[0] + 1 &&
			boundaryWeights.size(0)[1] == domainCellLabels.size()[1] &&
			boundaryWeights.size(1)[0] == domainCellLabels.size()[0] &&
			boundaryWeights.size(1)[1] == domainCellLabels.size()[1] + 1);

	tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelBoundaryCellList;

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localBoundaryCellList = parallelBoundaryCellList.local();

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
					localBoundaryCellList.push_back(cell);
			}
			else assert(domainCellLabels(cell) != CellLabels::BOUNDARY_CELL);
		}
	});

	// Combine parallel list of boundary cells
	std::vector<Vec2i> boundaryCellList;
	mergeLocalThreadVectors(boundaryCellList, parallelBoundaryCellList);

	tbb::parallel_for(tbb::blocked_range<int>(0, boundaryCellList.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = boundaryCellList[cellIndex];

			assert(domainCellLabels(cell) == CellLabels::INTERIOR_CELL);

			domainCellLabels(cell) = CellLabels::BOUNDARY_CELL;
		}
	});
}

template<typename StoreReal>
void buildExpandedBoundaryWeights(VectorGrid<StoreReal>& expandedBoundaryWeights,
									const VectorGrid<StoreReal>& baseBoundaryWeights,
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

template<typename StoreReal>
bool unitTestBoundaryCells(const UniformGrid<CellLabels>& cellLabels,
							const VectorGrid<StoreReal>* boundaryWeights)
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

	tbb::parallel_for(tbb::blocked_range<int>(0, cellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		if (!boundaryCellTestPassed) return;

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = cellLabels.unflatten(cellIndex);

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
		}
	});

	return boundaryCellTestPassed;
}

}
#endif