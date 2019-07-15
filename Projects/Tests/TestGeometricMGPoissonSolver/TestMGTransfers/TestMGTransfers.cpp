#include <memory>
#include <random>
#include <string>

#include "Eigen/Sparse"

#include "Common.h"
#include "GeometricMGOperations.h"
#include "InitialMGTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"

std::unique_ptr<Renderer> renderer;

static const Vec2i interiorResolution(128);

int main(int argc, char** argv)
{
	using namespace GeometricMGOperations;

	UniformGrid<CellLabels> interiorDomainCellLabels;

	// Simple domain set up
	interiorDomainCellLabels = buildSimpleDomain(interiorResolution, 1);

	// Initialize grid with sin function
	Vec2R interiorDx = Vec2R(1.) / Vec2R(interiorResolution);
	UniformGrid<Real> interiorInitialGuess(interiorResolution, 0);
	forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
	{
		if (interiorDomainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Vec2R point = interiorDx * Vec2R(cell);
			interiorInitialGuess(cell) = std::sin(2 * Util::PI * point[0]) * std::sin(2 * Util::PI * point[1]);
		}
	});

	// Print initial guess
	{
		Transform xform(interiorDx[0], Vec2R(0));
		ScalarGrid<Real> tempGrid(xform, interiorResolution);

		forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
		{
			tempGrid(cell) = interiorInitialGuess(cell);
		});

		tempGrid.printAsOBJ("initialGrid");
	}

	// Once we have the interior domain, we need to build the padded domain
	int exteriorLayerBand = 2;

	//
	// Build expanded grid with exterior bands
	//

	Vec2i expandedResolution = interiorResolution + 2 * Vec2i(exteriorLayerBand);
	Vec2i exteriorOffset(exteriorLayerBand);

	UniformGrid<CellLabels> expandedDomainCellLabels(expandedResolution, CellLabels::EXTERIOR);
	UniformGrid<Real> expandedInitialGuess(expandedResolution, 0);

	forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
	{
		Vec2i expandedCell = cell + exteriorOffset;
		expandedDomainCellLabels(expandedCell) = interiorDomainCellLabels(cell);
		expandedInitialGuess(expandedCell) = interiorInitialGuess(cell);
	});

	//
	// Coarsen cells
	//

	Vec2i coarseResolution = expandedResolution / 2;
	UniformGrid<CellLabels> coarseCellLabels = buildCoarseCellLabels(expandedDomainCellLabels);

	assert(unitTestCoarsening(coarseCellLabels, expandedDomainCellLabels));

	Vec2R expandedOrigin(-interiorDx * Vec2R(exteriorLayerBand));
	Real coarseDx = 2 * interiorDx[0];

	{
		//
		// Debug test for simple transfers
		//

		// Test a simple tansfer to the coarse grid and a transfer back
		UniformGrid<Real> coarseInitialGuess(coarseResolution, 0);

		downsample(coarseInitialGuess, expandedInitialGuess, coarseCellLabels, expandedDomainCellLabels);

		{
			Transform xform(2 * interiorDx[0], expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, coarseResolution);

			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = coarseInitialGuess(cell);
			});

			tempGrid.printAsOBJ("downsampledGrid");
		}

		// Transfer back
		UniformGrid<Real> expandedTransferGrid(expandedResolution, 0);
		upsample(expandedTransferGrid, coarseInitialGuess, expandedDomainCellLabels, coarseCellLabels);

		{
			Transform xform(interiorDx[0], expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, expandedResolution);

			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = expandedTransferGrid(cell);
			});

			tempGrid.printAsOBJ("upsampledGrid");
		}
	}

	//
	// Debug test by downsampling residual, solving for correction error and upsampling correction back
	//
	{
		//
		// Compute residual
		//

		UniformGrid<Real> expandedResidualGrid(expandedResolution, 0);
		UniformGrid<Real> expandedFineRHSGrid(expandedResolution, 0);

		computePoissonResidual(expandedResidualGrid, expandedInitialGuess, expandedFineRHSGrid, expandedDomainCellLabels, interiorDx[0]);

		{
			Transform xform(interiorDx[0], expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, expandedResolution);

			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = expandedResidualGrid(cell);
			});

			tempGrid.printAsOBJ("residualGrid");
		}

		//
		// Restrict residual to coarse RHS
		//

		UniformGrid<Real> coarseRHSGrid(coarseResolution, 0);
		downsample(coarseRHSGrid, expandedResidualGrid, coarseCellLabels, expandedDomainCellLabels);

		{
			Transform xform(coarseDx, expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, coarseResolution);

			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = coarseRHSGrid(cell);
			});

			tempGrid.printAsOBJ("downsampledResidual");
		}

		//
		// Apply direct solver
		//

		UniformGrid<Real> coarseSolution(coarseResolution, 0);

		{

			UniformGrid<Real> coarseResidualGrid(coarseResolution, 0);

			//
			// Solver with direct solver
			//

			// Build indices
			int interiorCellCount = 0;

			UniformGrid<int> interiorCellIndices(coarseResolution, -1);

			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR)
					interiorCellIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;
			Eigen::VectorXd rhsVector(interiorCellCount);

			Real gridScalar = 1. / Util::sqr(coarseDx);
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR)
				{
					int diagonal = 0;
					int index = interiorCellIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (coarseCellLabels(adjacentCell) == CellLabels::INTERIOR)
							{
								int adjacentIndex = interiorCellIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								sparseElements.push_back(Eigen::Triplet<double>(index, adjacentIndex, -gridScalar));
								++diagonal;
							}
							else if (coarseCellLabels(adjacentCell) == CellLabels::DIRICHLET)
								++diagonal;
						}

					sparseElements.push_back(Eigen::Triplet<double>(index, index, diagonal * gridScalar));

					rhsVector(index) = coarseRHSGrid(cell);
				}
			});

			// Solve system
			Eigen::SparseMatrix<double> sparseMatrix(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			//Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
			//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver;
			Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
			solver.compute(sparseMatrix);

			if (solver.info() != Eigen::Success) return false;

			Eigen::VectorXd solutionVector = solver.solve(rhsVector);
			if (solver.info() != Eigen::Success) return false;

			// Apply solution
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR)
				{
					int index = interiorCellIndices(cell);
					assert(index >= 0);

					coarseSolution(cell) = solutionVector(index);
				}
			});
		}

		{
			Transform xform(coarseDx, expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, coarseResolution);

			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = coarseSolution(cell);
			});

			tempGrid.printAsOBJ("coarseSolutionGrid");
		}

		//
		// Prolongate solution
		//

		UniformGrid<Real> expandedFineSolution(expandedResolution, 0);

		GeometricMGOperations::upsample(expandedFineSolution,
										coarseSolution,
										expandedDomainCellLabels,
										coarseCellLabels);

		{
			Transform xform(interiorDx[0], expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, expandedResolution);

			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = expandedFineSolution(cell);
			});

			tempGrid.printAsOBJ("fineSolutionGrid");
		}

		//
		// Apply correction
		//

		UniformGrid<Real> interiorFinalSolution(interiorResolution, 0);

		forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
		{
			Vec2i expandedCell = cell + exteriorOffset;
			if (expandedDomainCellLabels(expandedCell) == CellLabels::INTERIOR)
				interiorFinalSolution(cell) = interiorInitialGuess(cell) + expandedFineSolution(expandedCell);
		});

		{
			Transform xform(interiorDx[0], Vec2R(0));
			ScalarGrid<Real> tempGrid(xform, interiorResolution);

			forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = interiorFinalSolution(cell);
			});

			tempGrid.printAsOBJ("solutionGrid");
		}

		//
		// Print out grids
		//

		// Print domain labels to make sure they are set up correctly
		int pixelHeight = 1080;
		int pixelWidth = pixelHeight;
		renderer = std::make_unique<Renderer>("MG Error Correction and Transfer Test", Vec2i(pixelWidth, pixelHeight), expandedOrigin, interiorDx[0] * Real(expandedResolution[1]), &argc, argv);

		{
			Transform xform(coarseDx, expandedOrigin);
			ScalarGrid<Real> tempGrid(xform, coarseResolution);

			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				tempGrid(cell) = Real(coarseCellLabels(cell));
			});

			tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR), Real(CellLabels::DIRICHLET));
		}

		renderer->run();
	}
}