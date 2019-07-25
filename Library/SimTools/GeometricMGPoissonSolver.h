#ifndef GEOMETRIC_MG_POISSONSOLVER_H
#define GEOMETRIC_MG_POISSONSOLVER_H

#include "Eigen/Sparse"

#include "Common.h"
#include "GeometricMGOperations.h"
#include "UniformGrid.h"
#include "VectorGrid.h"

using namespace GeometricMGOperations;

class GeometricMGPoissonSolver
{
public:

	GeometricMGPoissonSolver(const UniformGrid<CellLabels> &initialDomainLabels,
								const int mgLevels,
								const Real dx)
		: myDoApplyGradientWeights(false)
	{
		assert(mgLevels > 0 && dx > 0);

		// Add the necessary exterior cells so that after coarsening to the top level
		// there is still a single layer of exterior cells
		int exteriorPadding = pow(2, mgLevels - 1);

		Vec2i expandedResolution = initialDomainLabels.size() + 2 * Vec2i(exteriorPadding);

		// Expand the domain to be a power of 2.
		for (int axis : {0, 1})
		{
			Real logSize = std::log2(Real(expandedResolution[axis]));
			logSize = std::ceil(logSize);

			expandedResolution[axis] = std::exp2(logSize);
		}

		myExteriorOffset = Vec2i(exteriorPadding);

		myMGLevels = mgLevels;

		// Clamp top level to the highest possible coarsening given the resolution.
		for (int axis : {0, 1})
		{
			int topLevel = int(log2(expandedResolution[axis]));
			if (topLevel < myMGLevels) myMGLevels = topLevel;
		}

		// Build finest level domain labels with the necessary padding to maintain a 
		// 1-band ring of exterior cells at the coarsest level.
		myDomainLabels.resize(myMGLevels);

		myDomainLabels[0] = UniformGrid<CellLabels>(expandedResolution, CellLabels::EXTERIOR);
		
		// Copy over initial domain
		forEachVoxelRange(Vec2i(0), initialDomainLabels.size(), [&](const Vec2i &cell)
		{
			Vec2i expandedCell = cell + myExteriorOffset;
			myDomainLabels[0](expandedCell) = initialDomainLabels(cell);
		});

		auto checkInteriorCell = [](const UniformGrid<CellLabels> &testGrid) -> bool
		{
			bool hasInteriorCell = false;
			forEachVoxelRange(Vec2i(0), testGrid.size(), [&hasInteriorCell, &testGrid](const Vec2i &cell)
			{
				if (hasInteriorCell) return;

				if (testGrid(cell) == CellLabels::INTERIOR)
					hasInteriorCell = true;
			});

			return hasInteriorCell;
		};

		assert(checkInteriorCell(myDomainLabels[0]));

		// Precompute the coarsening strategy. Cap level if there are no longer interior cells
		for (int level = 1; level < myMGLevels; ++level)
		{
			myDomainLabels[level] = buildCoarseCellLabels(myDomainLabels[level - 1]);

			if (!checkInteriorCell(myDomainLabels[level]))
			{
				myMGLevels = level - 1;
				myDomainLabels.resize(myMGLevels);
				break;
			}
		}

		myDx.resize(myMGLevels);
		myDx[0] = dx;
		
		for (int level = 1; level < myMGLevels; ++level)
			myDx[level] = 2. * myDx[level - 1];

		myBoundaryCells.resize(myMGLevels);
		for (int level = 0; level < myMGLevels; ++level)
			myBoundaryCells[level] = buildBoundaryCells(myDomainLabels[level], 3);

		// Initialize solution vectors
		mySolutionGrids.resize(myMGLevels);
		myRHSGrids.resize(myMGLevels);
		myResidualGrids.resize(myMGLevels);
		
		for (int level = 0; level < myMGLevels; ++level)
		{
			mySolutionGrids[level] = UniformGrid<Real>(myDomainLabels[level].size());
			myRHSGrids[level] = UniformGrid<Real>(myDomainLabels[level].size());
			myResidualGrids[level] = UniformGrid<Real>(myDomainLabels[level].size());
		}

		// Pre-build matrix at the coarsest level
		{
			int interiorCellCount = 0;
			Vec2i coarsestSize = myDomainLabels[myMGLevels - 1].size();

			myDirectSolverIndices = UniformGrid<int>(coarsestSize, -1);

			forEachVoxelRange(Vec2i(0), coarsestSize, [&](const Vec2i &cell)
			{
				if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR)
					myDirectSolverIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;
			Eigen::VectorXd rhsVector(interiorCellCount);

			myCoarseRHSVector = Eigen::VectorXd::Zero(interiorCellCount);

			Real gridScale = 1. / Util::sqr(myDx[myMGLevels - 1]);
			forEachVoxelRange(Vec2i(0), coarsestSize, [&](const Vec2i &cell)
			{
				if (myDomainLabels[myMGLevels - 1](cell) == CellLabels::INTERIOR)
				{
					int diagonal = 0;
					int index = myDirectSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							auto cellLabels = myDomainLabels[myMGLevels - 1](adjacentCell);
							if (cellLabels == CellLabels::INTERIOR)
							{
								int adjacentIndex = myDirectSolverIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								sparseElements.push_back(Eigen::Triplet<double>(index, adjacentIndex, -gridScale));
								++diagonal;
							}
							else if (cellLabels == CellLabels::DIRICHLET)
								++diagonal;
						}

					sparseElements.push_back(Eigen::Triplet<double>(index, index, gridScale * diagonal));
				}
			});

			// Solve system
			sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			myCoarseSolver.compute(sparseMatrix);

			assert(myCoarseSolver.info() == Eigen::Success);
		}
	}
	
	void applyMGVCycle(UniformGrid<Real> &solutionVector,
						const UniformGrid<Real> &rhsVector,
						const bool useInitialGuess = false);

	void setGradientWeights(const VectorGrid<Real> &gradientWeights)
	{
		myDoApplyGradientWeights = true;
		
		// Pad out gradient weights
		Transform tempXform(1, Vec2R(0));
		myFineGradientWeights = VectorGrid<Real>(tempXform, myDomainLabels[0].size(), 0, VectorGridSettings::SampleType::STAGGERED);

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i(0), gradientWeights.size(axis), [&](const Vec2i& face)
			{
				myFineGradientWeights(face + myExteriorOffset, axis) = gradientWeights(face, axis);
			});
		}
	}

private:

	std::vector<UniformGrid<CellLabels>> myDomainLabels;
	std::vector<UniformGrid<Real>> mySolutionGrids, myRHSGrids, myResidualGrids;

	std::vector<std::vector<Vec2i>> myBoundaryCells;

	UniformGrid<int> myDirectSolverIndices;

	int myMGLevels;
	std::vector<Real> myDx;
	Vec2i myExteriorOffset;

	bool myDoApplyGradientWeights;
	VectorGrid<Real> myFineGradientWeights;

	Eigen::VectorXd myCoarseRHSVector;
	Eigen::SparseMatrix<double> sparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
};

#endif