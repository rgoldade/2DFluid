#ifndef GEOMETRIC_MULTIGRID_OPERATORS_H
#define GEOMETRIC_MULTIGRID_OPERATORS_H

#include <utility>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim2D::GeometricMultigridOperators
{

enum class CellLabels { INTERIOR_CELL, EXTERIOR_CELL, DIRICHLET_CELL, BOUNDARY_CELL };

void interiorJacobiPoissonSmoother(UniformGrid<double>& solution,
									const UniformGrid<double>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const double dx,
									const VectorGrid<double>* boundaryWeights = nullptr);

void boundaryJacobiPoissonSmoother(UniformGrid<double>& solution,
									const UniformGrid<double>& rhs,
									const UniformGrid<CellLabels>& domainCellLabels,
									const VecVec2i& boundaryCells,
									const double dx,
									const VectorGrid<double>* boundaryWeights = nullptr);

void applyPoissonMatrix(UniformGrid<double>& destination,
						const UniformGrid<double>& source,
						const UniformGrid<CellLabels>& domainCellLabels,
						const double dx,
						const VectorGrid<double>* boundaryWeights = nullptr);

void computePoissonResidual(UniformGrid<double>& residual,
							const UniformGrid<double>& solution,
							const UniformGrid<double>& rhs,
							const UniformGrid<CellLabels>& domainCellLabels,
							const double dx,
							const VectorGrid<double>* boundaryWeights = nullptr);

void downsample(UniformGrid<double>& destinationGrid,
				const UniformGrid<double>& sourceGrid,
				const UniformGrid<CellLabels>& destinationCellLabels,
				const UniformGrid<CellLabels>& sourceCellLabels);

void upsampleAndAdd(UniformGrid<double>& destinationGrid,
					const UniformGrid<double>& sourceGrid,
					const UniformGrid<CellLabels>& destinationCellLabels,
					const UniformGrid<CellLabels>& sourceCellLabels);

double dotProduct(const UniformGrid<double>& vectorA,
						const UniformGrid<double>& vectorB,
						const UniformGrid<CellLabels>& domainCellLabels);

void addToVector(UniformGrid<double>& destination,
					const UniformGrid<double>& source,
					const UniformGrid<CellLabels>& domainCellLabels,
					const double scale);

void addVectors(UniformGrid<double>& destination,
				const UniformGrid<double>& source,
				const UniformGrid<double>& scaledSource,
				const UniformGrid<CellLabels>& domainCellLabels,
				const double scale);

double squaredl2Norm(const UniformGrid<double>& vectorGrid,
						const UniformGrid<CellLabels>& domainCellLabels);

double lInfinityNorm(const UniformGrid<double>& vectorGrid,
						const UniformGrid<CellLabels>& domainCellLabels);

Vec2i getChildCell(const Vec2i& cell, const int childIndex);

UniformGrid<CellLabels> buildCoarseCellLabels(const UniformGrid<CellLabels>& sourceCellLabels);

VecVec2i buildBoundaryCells(const UniformGrid<CellLabels>& sourceCellLabels, int boundaryWidth);

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

void buildExpandedBoundaryWeights(VectorGrid<double>& expandedBoundaryWeights,
									const VectorGrid<double>& baseBoundaryWeights,
									const UniformGrid<CellLabels>& expandedDomainCellLabels,
									const Vec2i& exteriorOffset,
									const int axis);

void setBoundaryDomainLabels(UniformGrid<CellLabels>& sourceCellLabels,
								const VectorGrid<double>& boundaryWeights);

bool unitTestCoarsening(const UniformGrid<CellLabels>& coarseCellLabels,
						const UniformGrid<CellLabels>& fineCellLabels);

bool unitTestBoundaryCells(const UniformGrid<CellLabels>& domainCellLabels, const VectorGrid<double>* boundaryWeights = nullptr);

bool unitTestExteriorCells(const UniformGrid<CellLabels>& domainCellLabels);

// Implementation of template functions

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
	double minLog = std::min(std::log2(double(baseCustomLabels.size()[0])),
							std::log2(double(baseCustomLabels.size()[1])));

	int mgLevels = std::ceil(minLog) - std::log2(double(2));

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, mgLevels - 1);

	Vec2i expandedGridSize = baseCustomLabels.size() + 2 * Vec2i(exteriorPadding);

	for (int axis : {0, 1})
	{
		double logSize = std::log2(double(expandedGridSize[axis]));
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

}
#endif