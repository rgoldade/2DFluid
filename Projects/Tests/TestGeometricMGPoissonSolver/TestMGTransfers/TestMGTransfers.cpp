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

using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SimTools;

std::unique_ptr<Renderer> renderer;

static constexpr int gridSize = 256;
static constexpr bool useComplexDomain = true;
static constexpr bool useSolidSphere = true;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

	using StoreReal = double;
	using SolveReal = double;

	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	UniformGrid<CellLabels> domainCellLabels;
	VectorGrid<StoreReal> boundaryWeights;

	int mgLevels;
	{
		UniformGrid<CellLabels> baseDomainCellLabels;
		VectorGrid<StoreReal> baseBoundaryWeights;

		// Complex domain set up
		if (useComplexDomain)
			buildComplexDomain(baseDomainCellLabels,
								baseBoundaryWeights,
								gridSize,
								useSolidSphere);

		// Simple domain set up
		else
			buildSimpleDomain(baseDomainCellLabels,
								baseBoundaryWeights,
								gridSize,
								1 /*dirichlet band*/);

		// Build expanded domain
		std::pair<Vec2i, int> mgSettings = buildExpandedDomain(domainCellLabels, boundaryWeights, baseDomainCellLabels, baseBoundaryWeights);

		mgLevels = mgSettings.second;
	}

	SolveReal dx = boundaryWeights.dx();

	UniformGrid<StoreReal> solutionGrid(domainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(),tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				Vec2f point(dx * Vec2f(cell));
				solutionGrid(cell) = 4. * (std::sin(2 * PI * point[0]) * std::sin(2 * PI * point[1]) +
											std::sin(4 * PI * point[0]) * std::sin(4 * PI * point[1]));
			}
		}
	});

	// Print initial guess
	solutionGrid.printAsOBJ("initialGuess");

	UniformGrid<CellLabels> coarseCellLabels = buildCoarseCellLabels(domainCellLabels);

	assert(unitTestBoundaryCells<StoreReal>(domainCellLabels, &boundaryWeights) && unitTestBoundaryCells<StoreReal>(coarseCellLabels));
	assert(unitTestExteriorCells(domainCellLabels) && unitTestExteriorCells(coarseCellLabels));
	assert(unitTestCoarsening(coarseCellLabels, domainCellLabels));

	SolveReal coarseDx = 2 * dx;

	{
		//
		// Debug test for simple transfers
		//

		// Test a simple tansfer to the coarse grid and a transfer back
		UniformGrid<StoreReal> coarseInitialGuess(coarseCellLabels.size(), 0);

		downsample<SolveReal>(coarseInitialGuess, solutionGrid, coarseCellLabels, domainCellLabels);
	
		coarseInitialGuess.printAsOBJ("downsampledGrid");

		// Transfer back
		UniformGrid<StoreReal> transferGrid(domainCellLabels.size(), 0);
		upsampleAndAdd<SolveReal>(transferGrid, coarseInitialGuess, domainCellLabels, coarseCellLabels);

		transferGrid.printAsOBJ("upsampledGrid");
	}

	//
	// Debug test by downsampling residual, solving for correction error and upsampling correction back
	//
	{
		//
		// Compute residual
		//

		UniformGrid<StoreReal> residualGrid(domainCellLabels.size(), 0);
		UniformGrid<StoreReal> rhsGrid(domainCellLabels.size(), 0);

		computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

		residualGrid.printAsOBJ("residualGrid");

		//
		// Restrict residual to coarse RHS
		//

		UniformGrid<StoreReal> coarseRHSGrid(coarseCellLabels.size(), 0);
		downsample<SolveReal>(coarseRHSGrid, residualGrid, coarseCellLabels, domainCellLabels);

		coarseRHSGrid.printAsOBJ("downsampledResidual");

		//
		// Apply direct solver
		//

		UniformGrid<StoreReal> coarseSolution(coarseCellLabels.size(), 0);

		{

			UniformGrid<StoreReal> coarseResidualGrid(coarseCellLabels.size(), 0);

			//
			// Solver with direct solver
			//

			// Build indices
			int interiorCellCount = 0;

			UniformGrid<int> interiorCellIndices(coarseCellLabels.size(), -1);

			forEachVoxelRange(Vec2i(0), coarseCellLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					coarseCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					interiorCellIndices(cell) = interiorCellCount++;
			});
			
			Vector rhsVector = Vector::Zero(interiorCellCount);

			tbb::parallel_for(tbb::blocked_range<int>(0, coarseCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
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
			std::vector<Eigen::Triplet<SolveReal>> sparseElements;

			SolveReal gridScalar = 1. / sqr(coarseDx);
			forEachVoxelRange(Vec2i(0), coarseCellLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseCellLabels(cell) == CellLabels::INTERIOR_CELL)
				{
					int diagonal = 0;
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
					int diagonal = 0;
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
			Eigen::SparseMatrix<SolveReal> sparseMatrix(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> solver;
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

		UniformGrid<StoreReal> correctionGrid(domainCellLabels.size(), 0);

		upsampleAndAdd<SolveReal>(correctionGrid, coarseSolution, domainCellLabels, coarseCellLabels);

		correctionGrid.printAsOBJ("prolongatedCorrectio");

		//
		// Apply correction
		//

		upsampleAndAdd<SolveReal>(solutionGrid, coarseSolution, domainCellLabels, coarseCellLabels);
		solutionGrid.printAsOBJ("solutionGrid");

		//
		// Print out grids
		//

		// Print domain labels to make sure they are set up correctly
		int pixelHeight = 1080;
		int pixelWidth = pixelHeight;
		renderer = std::make_unique<Renderer>("MG Error Correction and Transfer Test", Vec2i(pixelWidth, pixelHeight), Vec2f(0), 1, &argc, argv);

		ScalarGrid<float> tempGrid(Transform(dx, Vec2f(0)), domainCellLabels.size());

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				tempGrid(cell) = float(domainCellLabels(cell));
			}
		});

		tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), float(CellLabels::INTERIOR_CELL), float(CellLabels::BOUNDARY_CELL));

		renderer->run();
	}
}