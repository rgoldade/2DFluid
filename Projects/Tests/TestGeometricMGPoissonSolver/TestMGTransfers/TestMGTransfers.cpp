#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <Eigen/Sparse>

#include "GeometricMultigridOperators.h"
#include "InitialMultigridTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;

std::unique_ptr<Renderer> gRenderer;

static constexpr int gGridSize = 256;
static constexpr bool gUseComplexDomain = true;
static constexpr bool gUseSolidSphere = true;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

	using Vector = Eigen::VectorXd;

	UniformGrid<CellLabels> domainCellLabels;
	VectorGrid<double> boundaryWeights;

	int mgLevels;
	{
		UniformGrid<CellLabels> baseDomainCellLabels;
		VectorGrid<double> baseBoundaryWeights;

		// Complex domain set up
		if (gUseComplexDomain)
			buildComplexDomain(baseDomainCellLabels,
								baseBoundaryWeights,
								gGridSize,
								gUseSolidSphere);

		// Simple domain set up
		else
			buildSimpleDomain(baseDomainCellLabels,
								baseBoundaryWeights,
								gGridSize,
								1 /*dirichlet band*/);

		// Build expanded domain
		std::pair<Vec2i, int> mgSettings = buildExpandedDomain(domainCellLabels, boundaryWeights, baseDomainCellLabels, baseBoundaryWeights);

		mgLevels = mgSettings.second;
	}

	double dx = boundaryWeights.dx();

	UniformGrid<double> solutionGrid(domainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				Vec2d point(dx * cell.cast<double>());
				solutionGrid(cell) = 4. * (std::sin(2 * PI * point[0]) * std::sin(2 * PI * point[1]) +
											std::sin(4 * PI * point[0]) * std::sin(4 * PI * point[1]));
			}
		}
	});

	// Print initial guess
	solutionGrid.printAsOBJ("initialGuess");

	UniformGrid<CellLabels> coarseCellLabels = buildCoarseCellLabels(domainCellLabels);

	assert(unitTestBoundaryCells(domainCellLabels, &boundaryWeights) && unitTestBoundaryCells(coarseCellLabels));
	assert(unitTestExteriorCells(domainCellLabels) && unitTestExteriorCells(coarseCellLabels));
	assert(unitTestCoarsening(coarseCellLabels, domainCellLabels));

	double coarseDx = 2 * dx;

	{
		//
		// Debug test for simple transfers
		//

		// Test a simple tansfer to the coarse grid and a transfer back
		UniformGrid<double> coarseInitialGuess(coarseCellLabels.size(), 0);

		downsample(coarseInitialGuess, solutionGrid, coarseCellLabels, domainCellLabels);
	
		coarseInitialGuess.printAsOBJ("downsampledGrid");

		// Transfer back
		UniformGrid<double> transferGrid(domainCellLabels.size(), 0);
		upsampleAndAdd(transferGrid, coarseInitialGuess, domainCellLabels, coarseCellLabels);

		transferGrid.printAsOBJ("upsampledGrid");
	}

	//
	// Debug test by downsampling residual, solving for correction error and upsampling correction back
	//
	{
		//
		// Compute residual
		//

		UniformGrid<double> residualGrid(domainCellLabels.size(), 0);
		UniformGrid<double> rhsGrid(domainCellLabels.size(), 0);

		computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

		residualGrid.printAsOBJ("residualGrid");

		//
		// Restrict residual to coarse RHS
		//

		UniformGrid<double> coarseRHSGrid(coarseCellLabels.size(), 0);
		downsample(coarseRHSGrid, residualGrid, coarseCellLabels, domainCellLabels);

		coarseRHSGrid.printAsOBJ("downsampledResidual");

		//
		// Apply direct solver
		//

		UniformGrid<double> coarseSolution(coarseCellLabels.size(), 0);

		{

			UniformGrid<double> coarseResidualGrid(coarseCellLabels.size(), 0);

			//
			// Solver with direct solver
			//

			// Build indices
			int interiorCellCount = 0;

			UniformGrid<int> interiorCellIndices(coarseCellLabels.size(), -1);

			forEachVoxelRange(Vec2i::Zero(), coarseCellLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					coarseCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					interiorCellIndices(cell) = interiorCellCount++;
			});
			
			Vector rhsVector = Vector::Zero(interiorCellCount);

			tbb::parallel_for(tbb::blocked_range<int>(0, coarseCellLabels.voxelCount()), [&](const tbb::blocked_range<int> &range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = interiorCellIndices.unflatten(cellIndex);

					if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = interiorCellIndices(cell);
						assert(index >= 0);

						rhsVector(index) = coarseRHSGrid(cell);
					}
					else
					{
						assert(interiorCellIndices(cell) == -1);
					}
				}
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;

			double gridScalar = 1. / std::pow(coarseDx, 2);
			forEachVoxelRange(Vec2i::Zero(), coarseCellLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL)
				{
					int index = interiorCellIndices(cell);
					assert(index >= 0);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);
							assert(coarseCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
									coarseCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);
							
							int adjacentIndex = interiorCellIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
						}

					sparseElements.emplace_back(index, index, 4. * gridScalar);
				}
				else if (coarseCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					double diagonal = 0;
					int index = interiorCellIndices(cell);
					assert(index >= 0);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (coarseCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
								coarseCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
							{
								int adjacentIndex = interiorCellIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
								++diagonal;
							}
							else
							{
								assert(interiorCellIndices(adjacentCell) == -1);
								if (coarseCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
									++diagonal;
							}
						}

					sparseElements.emplace_back(index, index, diagonal * gridScalar);
				}
			});

			// Solve system
			Eigen::SparseMatrix<double> sparseMatrix(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
			solver.compute(sparseMatrix);

			if (solver.info() != Eigen::Success)
			{
			    std::cout << "Solver failed to pre-compute system" << std::endl;
			    return 0;
			}

			Vector solutionVector = solver.solve(rhsVector);
			if (solver.info() != Eigen::Success)
			{
			    std::cout << "Solver failed" << std::endl;
			    return 0;
			}

			tbb::parallel_for(tbb::blocked_range<int>(0, coarseCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = interiorCellIndices.unflatten(cellIndex);

					if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = interiorCellIndices(cell);
						assert(index >= 0);

						coarseSolution(cell) = solutionVector(index);
					}
					else
					{
						assert(interiorCellIndices(cell) == -1);
					}
				}
			});

			coarseSolution.printAsOBJ("coarseSolutionGrid");
		}

		//
		// Prolongate solution
		//

		UniformGrid<double> correctionGrid(domainCellLabels.size(), 0);

		upsampleAndAdd(correctionGrid, coarseSolution, domainCellLabels, coarseCellLabels);

		correctionGrid.printAsOBJ("prolongatedCorrectio");

		//
		// Apply correction
		//

		upsampleAndAdd(solutionGrid, coarseSolution, domainCellLabels, coarseCellLabels);
		solutionGrid.printAsOBJ("solutionGrid");

		//
		// Print out grids
		//

		// Print domain labels to make sure they are set up correctly
		int pixelHeight = 1080;
		int pixelWidth = pixelHeight;
		gRenderer = std::make_unique<Renderer>("MG Error Correction and Transfer Test", Vec2i(pixelWidth, pixelHeight), Vec2d::Zero(), 1, &argc, argv);

		ScalarGrid<double> tempGrid(Transform(dx, Vec2d::Zero()), domainCellLabels.size());

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				tempGrid(cell) = double(domainCellLabels(cell));
			}
		});

		tempGrid.drawVolumetric(*gRenderer, Vec3d::Zero(), Vec3d::Ones(), double(CellLabels::INTERIOR_CELL), double(CellLabels::BOUNDARY_CELL));

		gRenderer->run();
	}
}