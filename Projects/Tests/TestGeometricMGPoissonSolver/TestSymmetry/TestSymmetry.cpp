#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <Eigen/Sparse>

#include "GeometricMultigridOperators.h"
#include "GeometricMultigridPoissonSolver.h"
#include "InitialMultigridTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;

std::unique_ptr<Renderer> gRenderer;

static constexpr int gGridSize = 512;
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

	UniformGrid<double> rhsA(domainCellLabels.size(), 0);
	UniformGrid<double> rhsB(domainCellLabels.size(), 0);
	
	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0, 1);

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);
			
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				rhsA(cell) = distribution(generator);
				rhsB(cell) = distribution(generator);
			}
		}
	});

	Transform xform(dx, Vec2d::Zero());
	std::cout.precision(10);
	{
		UniformGrid<double> solutionA(domainCellLabels.size(), 0);
		UniformGrid<double> solutionB(domainCellLabels.size(), 0);

		VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

		// Test Jacobi symmetry
		boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
		boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

		interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
		interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

		boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
		boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

		double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
		double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

		std::cout << "Jacobi smoother symmetry test: " << dotA << ", " << dotB << std::endl;
		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
		// Test direct solve symmetry
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
		Eigen::SparseMatrix<double> sparseMatrix;

		// Pre-build matrix at the coarsest level
		int interiorCellCount = 0;
		UniformGrid<int> directSolverIndices(domainCellLabels.size(), -1);
		{
			forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i &cell)
			{
				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					directSolverIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;

			double gridScale = 1. / std::pow(dx, 2);
			forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i &cell)
			{
				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							assert(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
									domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

							Vec2i face = cellToFace(cell, axis, direction);
							assert(boundaryWeights(face, axis) == 1);

							int adjacentIndex = directSolverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							sparseElements.emplace_back(index, adjacentIndex, -gridScale);
						}
					sparseElements.emplace_back(index, index, 4. * gridScale);
				}
				else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					double diagonal = 0;

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								Vec2i face = cellToFace(cell, axis, direction);
								assert(boundaryWeights(face, axis) == 1);

								sparseElements.emplace_back(index, adjacentIndex, -gridScale);
								++diagonal;
							}
							else if (domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								Vec2i face = cellToFace(cell, axis, direction);
								double weight = boundaryWeights(face, axis);

								sparseElements.emplace_back(index, adjacentIndex, -gridScale * weight);
								diagonal += weight;
							}
							else if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
								assert(adjacentIndex == -1);

								Vec2i face = cellToFace(cell, axis, direction);
								double weight = boundaryWeights(face, axis);

								diagonal += weight;
							}
							else
							{
								assert(domainCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL);
								int adjacentIndex = directSolverIndices(adjacentCell);
								assert(adjacentIndex == -1);

								Vec2i face = cellToFace(cell, axis, direction);
								assert(boundaryWeights(face, axis) == 0);
							}
						}

					sparseElements.emplace_back(index, index, gridScale * diagonal);
				}
			});

			// Solve system
			sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			myCoarseSolver.compute(sparseMatrix);

			assert(myCoarseSolver.info() == Eigen::Success);
		}

		UniformGrid<double> solutionA(domainCellLabels.size(), 0);

		{
			Vector coarseRHSVector = Vector::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseRHSVector(index) = rhsA(cell);
					}
				}
			});

			Vector directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						solutionA(cell) = directSolution(index);
					}
				}
			});
		}

		UniformGrid<double> solutionB(domainCellLabels.size(), 0);

		{
			Vector coarseRHSVector = Vector::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseRHSVector(index) = rhsB(cell);
					}
				}
			});

			Vector directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						solutionB(cell) = directSolution(index);
					}
				}
			});
		}

		// Compute dot products
		double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
		double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

		std::cout << "Direct solver symmetry test: " << dotA << ", " << dotB << std::endl;
		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		// Test down and up sampling
		UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);

		assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels, &boundaryWeights));
		assert(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
		assert(unitTestCoarsening(coarseDomainLabels, domainCellLabels));

		UniformGrid<double> coarseRhs(coarseDomainLabels.size(), 0);

		UniformGrid<double> solutionA(domainCellLabels.size(), 0);

		{
			downsample(coarseRhs, rhsA, coarseDomainLabels, domainCellLabels);
			upsampleAndAdd(solutionA, coarseRhs, domainCellLabels, coarseDomainLabels);
		}
		
		UniformGrid<double> solutionB(domainCellLabels.size(), 0);
		
		{
			downsample(coarseRhs, rhsB, coarseDomainLabels, domainCellLabels);
			upsampleAndAdd(solutionB, coarseRhs, domainCellLabels, coarseDomainLabels);
		}

		// Compute dot products
		double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
		double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

		std::cout << "Coarse transfer symmetry test: " << dotA << ", " << dotB << std::endl;
		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
		// Test single level correction
		UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);

		assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels, &boundaryWeights));
		assert(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
		assert(unitTestCoarsening(coarseDomainLabels, domainCellLabels));
	
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
		Eigen::SparseMatrix<double> sparseMatrix;

		// Pre-build matrix at the coarsest level
		int interiorCellCount = 0;
		UniformGrid<int> directSolverIndices(coarseDomainLabels.size(), -1);
		{
			forEachVoxelRange(Vec2i::Zero(), coarseDomainLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
					coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
					directSolverIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;

			double gridScale = 1. / std::pow(2. * dx, 2);
			forEachVoxelRange(Vec2i::Zero(), coarseDomainLabels.size(), [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							auto adjacentLabels = coarseDomainLabels(adjacentCell);
							assert(adjacentLabels == CellLabels::INTERIOR_CELL ||
								adjacentLabels == CellLabels::BOUNDARY_CELL);

							int adjacentIndex = directSolverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							sparseElements.emplace_back(index, adjacentIndex, -gridScale);
						}

					sparseElements.emplace_back(index, index, 4. * gridScale);
				}
				else if (coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					double diagonal = 0;
					int index = directSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							auto cellLabels = coarseDomainLabels(adjacentCell);
							if (cellLabels == CellLabels::INTERIOR_CELL ||
								cellLabels == CellLabels::BOUNDARY_CELL)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
								assert(adjacentIndex >= 0);

								sparseElements.emplace_back(index, adjacentIndex, -gridScale);
								++diagonal;
							}
							else if (cellLabels == CellLabels::DIRICHLET_CELL)
								++diagonal;
						}

					sparseElements.emplace_back(index, index, diagonal * gridScale);
				}
			});

			sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			myCoarseSolver.compute(sparseMatrix);

			assert(myCoarseSolver.info() == Eigen::Success);
		}

		// Transfer rhs to coarse rhs as if it was a residual with a zero initial guess
		UniformGrid<double> solutionA(domainCellLabels.size(), 0);
		{
			// Pre-smooth to get an initial guess
			VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

			// Test Jacobi symmetry
			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);

			interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
		
			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
		
			// Compute new residual
			UniformGrid<double> residualA(domainCellLabels.size(), 0);

			computePoissonResidual(residualA, solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);

			UniformGrid<double> coarseRhs(coarseDomainLabels.size(), 0);
			downsample(coarseRhs, residualA, coarseDomainLabels, domainCellLabels);

			Vector coarseRHSVector = Vector::Zero(interiorCellCount);
			
			// Copy to Eigen and direct solve
			tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

					if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseRHSVector(index) = coarseRhs(cell);
					}
				}
			});

			UniformGrid<double> coarseSolution(coarseDomainLabels.size(), 0);

			Vector directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

					if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseSolution(cell) = directSolution(index);
					}
				}
			});

			upsampleAndAdd(solutionA, coarseSolution, domainCellLabels, coarseDomainLabels);

			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);

			interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);

			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
		}

		UniformGrid<double> solutionB(domainCellLabels.size(), 0);
		{
			// Pre-smooth to get an initial guess
			VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

			// Test Jacobi symmetry
			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

			interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

			// Compute new residual
			UniformGrid<double> residualB(domainCellLabels.size(), 0);

			computePoissonResidual(residualB, solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

			UniformGrid<double> coarseRhs(coarseDomainLabels.size(), 0);
			downsample(coarseRhs, residualB, coarseDomainLabels, domainCellLabels);

			Vector coarseRHSVector = Vector::Zero(interiorCellCount);

			// Copy to Eigen and direct solve
			tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

					if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseRHSVector(index) = coarseRhs(cell);
					}
				}
			});

			UniformGrid<double> coarseSolution(coarseDomainLabels.size(), 0);

			Vector directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

					if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
						coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = directSolverIndices(cell);
						assert(index >= 0);

						coarseSolution(cell) = directSolution(index);
					}
				}
			});

			upsampleAndAdd(solutionB, coarseSolution, domainCellLabels, coarseDomainLabels);

			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

			interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

			for (int iteration = 0; iteration < 3; ++iteration)
				boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);
		}

		double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
		double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

		std::cout << "One level correction symmetry: " << dotA << ", " << dotB << std::endl;
		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		// Pre-build multigrid preconditioner
		GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx);

		UniformGrid<double> solutionA(domainCellLabels.size(), 0);
		mgSolver.applyMGVCycle(solutionA, rhsA);
		mgSolver.applyMGVCycle(solutionA, rhsA, true);
		mgSolver.applyMGVCycle(solutionA, rhsA, true);
		mgSolver.applyMGVCycle(solutionA, rhsA, true);

		UniformGrid<double> solutionB(domainCellLabels.size(), 0);
		mgSolver.applyMGVCycle(solutionB, rhsB);
		mgSolver.applyMGVCycle(solutionB, rhsB, true);
		mgSolver.applyMGVCycle(solutionB, rhsB, true);
		mgSolver.applyMGVCycle(solutionB, rhsB, true);

		double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
		double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

		std::cout << "4 v-cycle symmetry: " << dotA << ", " << dotB << std::endl;
		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	gRenderer = std::make_unique<Renderer>("MG Symmetry Test", Vec2i(pixelWidth, pixelHeight), Vec2d::Zero(), 1, &argc, argv);

	ScalarGrid<double> tempGrid(Transform(dx, Vec2d::Zero()), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, tempGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = tempGrid.unflatten(cellIndex);

			tempGrid(cell) = double(domainCellLabels(cell));
		}
	});

	tempGrid.drawVolumetric(*gRenderer, Vec3d::Zero(), Vec3d::Ones(), double(CellLabels::INTERIOR_CELL), double(CellLabels::BOUNDARY_CELL));

	gRenderer->run();
}