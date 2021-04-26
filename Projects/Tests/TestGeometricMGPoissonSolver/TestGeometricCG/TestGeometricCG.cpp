#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "GeometricConjugateGradientSolver.h"
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

static constexpr bool gUseMGPreconditioner = true;
static constexpr bool gUseRandomGuess = true;

static constexpr bool gDoGeometricSolve = true;

static constexpr int gMaxIterations = 5000;
static constexpr double gSolverTolerance = 1E-10;

static constexpr double gDeltaAmplitude = 1000;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

	using Vector = Eigen::VectorXd;

	UniformGrid<CellLabels> domainCellLabels;
	VectorGrid<double> boundaryWeights;

	int mgLevels;
	Vec2i exteriorOffset;
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

		exteriorOffset = mgSettings.first;
		mgLevels = mgSettings.second;
	}

	double dx = boundaryWeights.dx();

	UniformGrid<double> rhsGrid(domainCellLabels.size(), 0);
	UniformGrid<double> solutionGrid(domainCellLabels.size(), 0);
	UniformGrid<double> residualGrid(domainCellLabels.size(), 0);

	if (gUseRandomGuess)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0, 1);

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int> &range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					solutionGrid(cell) = distribution(generator);
				}
			}
		});
	}

	// Set delta function
	double deltaPercent = .1;
	Vec2i deltaPoint = (deltaPercent * Vec2d(gGridSize, gGridSize)).cast<int>() + exteriorOffset;

	forEachVoxelRange(deltaPoint - Vec2i::Ones(), deltaPoint + Vec2i(2, 2), [&](const Vec2i &cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			rhsGrid(cell) = gDeltaAmplitude;
	});

	if (gDoGeometricSolve)
	{
		auto MatrixVectorMultiply = [&domainCellLabels, &boundaryWeights, dx](UniformGrid<double> &destinationGrid, const UniformGrid<double> &sourceGrid)
		{
			assert(destinationGrid.size() == sourceGrid.size() &&
					sourceGrid.size() == domainCellLabels.size());

			// Matrix-vector multiplication
			applyPoissonMatrix(destinationGrid, sourceGrid, domainCellLabels, dx, &boundaryWeights);
		};

		auto DotProduct = [&domainCellLabels](const UniformGrid<double> &grid0,
												const UniformGrid<double> &grid1)
		{
			assert(grid0.size() == grid1.size() &&
					grid1.size() == domainCellLabels.size());
			return dotProduct(grid0, grid1, domainCellLabels);
		};

		auto SquaredL2Norm = [&domainCellLabels](const UniformGrid<double> &grid)
		{
			assert(grid.size() == domainCellLabels.size());
			return squaredl2Norm(grid, domainCellLabels);
		};

		auto AddScaledVector = [&domainCellLabels](UniformGrid<double> &destination,
													const UniformGrid<double> &unscaledSource,
													const UniformGrid<double> &scaledSource,
													const double scale)
		{
			addVectors(destination, unscaledSource, scaledSource, domainCellLabels, scale);
		};

		auto AddToVector = [&domainCellLabels](UniformGrid<double> &destination,
												const UniformGrid<double> &scaledSource,
												const double scale)
		{
			addToVector(destination, scaledSource, domainCellLabels, scale);
		};

		if (gUseMGPreconditioner)
		{
			// Pre-build multigrid preconditioner
			GeometricMultigridPoissonSolver mgPreconditioner(domainCellLabels, boundaryWeights, mgLevels, dx);

			auto MultiGridPreconditioner = [&mgPreconditioner, &domainCellLabels](UniformGrid<double> &destinationGrid,
																					const UniformGrid<double> &sourceGrid)
			{
				assert(destinationGrid.size() == sourceGrid.size() &&
						sourceGrid.size() == domainCellLabels.size());
				mgPreconditioner.applyMGVCycle(destinationGrid, sourceGrid);
			};

			solveGeometricConjugateGradient(solutionGrid,
											rhsGrid,
											MatrixVectorMultiply,
											MultiGridPreconditioner,
											DotProduct,
											SquaredL2Norm,
											AddToVector,
											AddScaledVector,
											gSolverTolerance,
											gMaxIterations);
		}
		else
		{
			UniformGrid<double> diagonalPrecondGrid(domainCellLabels.size(), 0);

			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
			{
				const double gridScalar = 1. / std::pow(dx, 2);
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
					{
						for (int axis : {0, 1})
							for (int direction : {0, 1})
							{
								Vec2i adjacentCell = cellToCell(cell, axis, direction);

								assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.size()[axis]);
								assert(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
										domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

								Vec2i face = cellToFace(cell, axis, direction);
								assert(boundaryWeights(face, axis) == 1);
							}

						diagonalPrecondGrid(cell) = 1. / (4. * gridScalar);
					}
					else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						double diagonal = 0;
						for (int axis : {0, 1})
							for (int direction : {0, 1})
							{
								Vec2i adjacentCell = cellToCell(cell, axis, direction);

								assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.size()[axis]);

								Vec2i face = cellToFace(cell, axis, direction);

								if (domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
								{
									assert(boundaryWeights(face, axis) == 1);
									++diagonal;
								}
								else if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL ||
											domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
								{
									diagonal += boundaryWeights(face, axis);
								}
								else
								{
									assert(domainCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL);
									assert(boundaryWeights(face, axis) == 0);
								}
							}

						diagonalPrecondGrid(cell) = 1. / (diagonal * gridScalar);
					}
				}
			});

			auto DiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid](UniformGrid<double> &destinationGrid,
																									const UniformGrid<double> &sourceGrid)
			{
				assert(destinationGrid.size() == sourceGrid.size() &&
						sourceGrid.size() == domainCellLabels.size());

				tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
					{
						Vec2i cell = domainCellLabels.unflatten(cellIndex);

						if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
							domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
						{
							destinationGrid(cell) = sourceGrid(cell) * diagonalPrecondGrid(cell);
						}
					}
				});
			};

			solveGeometricConjugateGradient(solutionGrid,
											rhsGrid,
											MatrixVectorMultiply,
											DiagonalPreconditioner,
											DotProduct,
											SquaredL2Norm,
											AddToVector,
											AddScaledVector,
											gSolverTolerance,
											gMaxIterations);
		}
	}
	else
	{
		// Solve using Eigen
		int interiorCellCount = 0;
		UniformGrid<int> solverIndices(domainCellLabels.size(), -1);

		forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				solverIndices(cell) = interiorCellCount++;
		});

		// Build rows
		std::vector<Eigen::Triplet<double>> sparseElements;

		double gridScalar = 1. / std::pow(dx, 2);
		forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i &cell)
		{
			int index = solverIndices(cell);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				assert(index >= 0);

				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						auto adjacentLabels = domainCellLabels(adjacentCell);

						assert(adjacentLabels == CellLabels::INTERIOR_CELL ||
								adjacentLabels == CellLabels::BOUNDARY_CELL);

						int adjacentIndex = solverIndices(adjacentCell);
						assert(adjacentIndex >= 0);

						Vec2i face = cellToFace(cell, axis, direction);
						assert(boundaryWeights(face, axis) == 1);

						sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
					}

				sparseElements.emplace_back(index, index, 4. * gridScalar);
			}
			else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(index >= 0);

				double diagonal = 0;

				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						auto cellLabel = domainCellLabels(adjacentCell);

						if (cellLabel == CellLabels::INTERIOR_CELL)
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							Vec2i face = cellToFace(cell, axis, direction);
							assert(boundaryWeights(face, axis) == 1);

							sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
							++diagonal;
						}
						else if (cellLabel == CellLabels::BOUNDARY_CELL)
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							Vec2i face = cellToFace(cell, axis, direction);
							double weight = boundaryWeights(face, axis);

							sparseElements.emplace_back(index, adjacentIndex, -weight * gridScalar);
							diagonal += weight;
						}
						else if (cellLabel == CellLabels::DIRICHLET_CELL)
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex == -1);

							Vec2i face = cellToFace(cell, axis, direction);
							double weight = boundaryWeights(face, axis);
							
							diagonal += weight;
						}
						else
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex == -1);
							assert(cellLabel == CellLabels::EXTERIOR_CELL);
						}
					}

				sparseElements.emplace_back(index, index, diagonal * gridScalar);
			}
			else assert(index == -1);
		});

		Eigen::SparseMatrix<double> sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
		sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
		sparseMatrix.makeCompressed();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver(sparseMatrix);
		assert(solver.info() == Eigen::Success);

		// Build RHS
		Vector rhsVector = Vector::Zero(interiorCellCount);

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					int index = solverIndices(cell);
					assert(index >= 0);
					rhsVector(index) = rhsGrid(cell);
				}
			}
		});

		Vector initialGuessVector = Vector::Zero(interiorCellCount);
		if (gUseRandomGuess)
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(cellIndex);

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
						domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
					{
						int index = solverIndices(cell);
						assert(index >= 0);
						initialGuessVector(index) = solutionGrid(cell);
					}
				}
			});
		}

		solver.setTolerance(gSolverTolerance);
		solver.setMaxIterations(gMaxIterations);
		Vector solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

		std::cout << "GC iterations:     " << solver.iterations() << std::endl;
		std::cout << "GC residual: " << solver.error() << std::endl;

		// Recompute residual
		Vector residual = rhsVector - sparseMatrix * solutionVector;

		std::cout << "Drifted residual: " << std::sqrt(residual.squaredNorm() / rhsVector.squaredNorm()) << std::endl;
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	gRenderer = std::make_unique<Renderer>("Geometric CG and MG Preconditioner Test", Vec2i(pixelWidth, pixelHeight), Vec2d::Zero(), 1, &argc, argv);
	
	ScalarGrid<double> tempGrid(Transform(dx, Vec2d::Zero()), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
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