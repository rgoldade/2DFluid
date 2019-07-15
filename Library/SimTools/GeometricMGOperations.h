#ifndef GEOMETRIC_MG_OPERATIONS_H
#define GEOMETRIC_MG_OPERATIONS_H

#include "Common.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

namespace GeometricMGOperations
{
enum class CellLabels {INTERIOR, EXTERIOR, DIRICHLET, BOUNDARY};

void dampedJacobiPoissonSmoother(UniformGrid<Real> &solution,
									const UniformGrid<Real> &rhs,
									const UniformGrid<CellLabels> &cellLabels,
									const Real dx);

void dampedJacobiWeightedPoissonSmoother(UniformGrid<Real> &solution,
											const UniformGrid<Real> &rhs,
											const UniformGrid<CellLabels> &cellLabels,
											const VectorGrid<Real> &gradientWeights,
											const Real dx);

void dampedJacobiPoissonSmoother(UniformGrid<Real> &solution,
	const UniformGrid<Real> &rhs,
	const UniformGrid<CellLabels> &cellLabels,
	const std::vector<Vec2i> &boundaryCells,
	const Real dx);

void dampedJacobiWeightedPoissonSmoother(UniformGrid<Real> &solution,
	const UniformGrid<Real> &rhs,
	const UniformGrid<CellLabels> &cellLabels,
	const std::vector<Vec2i> &boundaryCells,
	const VectorGrid<Real> &gradientWeights,
	const Real dx);

void computePoissonResidual(UniformGrid<Real> &residual,
							const UniformGrid<Real> &solution,
							const UniformGrid<Real> &rhs,
							const UniformGrid<CellLabels> &cellLabels,
							const Real dx);

void computeWeightedPoissonResidual(UniformGrid<Real> &residual,
									const UniformGrid<Real> &solution,
									const UniformGrid<Real> &rhs,
									const UniformGrid<CellLabels> &cellLabels,
									const VectorGrid<Real> &gradientWeights,
									const Real dx);

void applyPoissonMatrix(UniformGrid<Real> &destination,
						const UniformGrid<Real> &source,
						const UniformGrid<CellLabels> &cellLabels,
						const Real dx);

void applyWeightedPoissonMatrix(UniformGrid<Real> &destination,
								const UniformGrid<Real> &source,
								const UniformGrid<CellLabels> &cellLabels,
								const VectorGrid<Real> &gradientWeights,
								const Real dx);

void downsample(UniformGrid<Real> &destinationGrid,
				const UniformGrid<Real> &sourceGrid,
				const UniformGrid<CellLabels> &destinationCellLabels,
				const UniformGrid<CellLabels> &sourceCellLabels);

void upsample(UniformGrid<Real> &destinationGrid,
				const UniformGrid<Real> &sourceGrid,
				const UniformGrid<CellLabels> &destinationCellLabels,
				const UniformGrid<CellLabels> &sourceCellLabels);

void upsampleAndAdd(UniformGrid<Real> &destinationGrid,
					const UniformGrid<Real> &sourceGrid,
					const UniformGrid<CellLabels> &destinationCellLabels,
					const UniformGrid<CellLabels> &sourceCellLabels);

std::vector<Vec2i> buildBoundaryCells(const UniformGrid<CellLabels> &sourceCellLabels, int boundaryWidth);

UniformGrid<CellLabels> buildBoundaryCellLabels(const UniformGrid<CellLabels> &sourceCellLabels, int boundaryWidth);
UniformGrid<CellLabels> buildCoarseCellLabels(const UniformGrid<CellLabels> &sourceCellLabels);

bool unitTestCoarsening(const UniformGrid<CellLabels> &coarseCellLabels,
						const UniformGrid<CellLabels> &fineCellLabels);

double dotProduct(const UniformGrid<Real> &grid0,
					const UniformGrid<Real> &grid1,
					const UniformGrid<CellLabels> &cellLabels);

void addToVector(UniformGrid<Real> &destination,
					const UniformGrid<Real> &source,
					const UniformGrid<CellLabels> &cellLabels,
					const Real scale);

void addToVector(UniformGrid<Real> &destination,
					const UniformGrid<Real> &source,
					const UniformGrid<Real> &scaledSource,
					const UniformGrid<CellLabels> &cellLabels,
					const Real scale);

double l2Norm(const UniformGrid<Real> &vectorGrid,
				const UniformGrid<CellLabels> &cellLabels);

double lInfinityNorm(const UniformGrid<Real> &vectorGrid,
						const UniformGrid<CellLabels> &cellLabels);
}

#endif