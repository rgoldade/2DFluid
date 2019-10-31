#include <memory>
#include <random>
#include <string>

#include "Common.h"
#include "GeometricConjugateGradientSolver.h"
#include "GeometricMultigridOperators.h"
#include "GeometricMultigridPoissonSolver.h"
#include "InitialMultigridTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"

std::unique_ptr<Renderer> renderer;

static constexpr int gridSize = 512;
static constexpr bool useComplexDomain = true;
static constexpr bool useSolidSphere = true;

static constexpr bool useMGPreconditioner = true;
static constexpr bool useRandomGuess = true;

static constexpr bool doGeometricSolve = true;

static constexpr int maxIterations = 5000;
static constexpr Real solverTolerance = 1E-10;

static constexpr double deltaAmplitude = 1000;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

	using StoreReal = double;
	using SolveReal = double;

	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	UniformGrid<CellLabels> domainCellLabels;
	VectorGrid<StoreReal> boundaryWeights;

	int mgLevels;
	Vec2i exteriorOffset;
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

		exteriorOffset = mgSettings.first;
		mgLevels = mgSettings.second;
	}

	SolveReal dx = boundaryWeights.dx();

	UniformGrid<StoreReal> rhsGrid(domainCellLabels.size(), 0);
	UniformGrid<StoreReal> solutionGrid(domainCellLabels.size(), 0);
	UniformGrid<StoreReal> residualGrid(domainCellLabels.size(), 0);

	int totalVoxels = domainCellLabels.size()[0] * domainCellLabels.size()[1];
	if (useRandomGuess)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<StoreReal> distribution(0, 1);

		tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(flatIndex);

				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					solutionGrid(cell) = distribution(generator);
				}
			}
		});
	}

	// Set delta function
	StoreReal deltaPercent = .1;
	Vec2i deltaPoint = Vec2i(deltaPercent * Vec2R(gridSize)) + exteriorOffset;

	forEachVoxelRange(deltaPoint - Vec2i(1), deltaPoint + Vec2i(2), [&](const Vec2i &cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			rhsGrid(cell) = deltaAmplitude;
	});

	if (doGeometricSolve)
	{
		auto MatrixVectorMultiply = [&domainCellLabels, &boundaryWeights, dx](UniformGrid<StoreReal> &destinationGrid, const UniformGrid<StoreReal> &sourceGrid)
		{
			assert(destinationGrid.size() == sourceGrid.size() &&
					sourceGrid.size() == domainCellLabels.size());

			// Matrix-vector multiplication
			applyPoissonMatrix<SolveReal>(destinationGrid, sourceGrid, domainCellLabels, dx, &boundaryWeights);
		};

		auto DotProduct = [&domainCellLabels](const UniformGrid<StoreReal> &grid0,
												const UniformGrid<StoreReal> &grid1)
		{
			assert(grid0.size() == grid1.size() &&
					grid1.size() == domainCellLabels.size());
			return dotProduct<SolveReal>(grid0, grid1, domainCellLabels);
		};

		auto SquaredL2Norm = [&domainCellLabels](const UniformGrid<StoreReal> &grid)
		{
			assert(grid.size() == domainCellLabels.size());
			return squaredl2Norm<SolveReal>(grid, domainCellLabels);
		};

		auto AddScaledVector = [&domainCellLabels](UniformGrid<StoreReal> &destination,
													const UniformGrid<StoreReal> &unscaledSource,
													const UniformGrid<StoreReal> &scaledSource,
													const SolveReal scale)
		{
			addVectors<SolveReal>(destination, unscaledSource, scaledSource, domainCellLabels, scale);
		};

		auto AddToVector = [&domainCellLabels](UniformGrid<StoreReal> &destination,
												const UniformGrid<StoreReal> &scaledSource,
												const SolveReal scale)
		{
			addToVector<SolveReal>(destination, scaledSource, domainCellLabels, scale);
		};

		if (useMGPreconditioner)
		{
			// Pre-build multigrid preconditioner
			GeometricMultigridPoissonSolver mgPreconditioner(domainCellLabels, boundaryWeights, mgLevels, dx);

			auto MultiGridPreconditioner = [&mgPreconditioner, &domainCellLabels](UniformGrid<StoreReal> &destinationGrid,
																					const UniformGrid<StoreReal> &sourceGrid)
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
											SolveReal(solverTolerance),
											maxIterations);
		}
		else
		{
			UniformGrid<StoreReal> diagonalPrecondGrid(domainCellLabels.size(), 0);

			tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
			{
				const SolveReal gridScalar = 1. / Util::sqr(dx);
				for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(flatIndex);

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
						SolveReal diagonal = 0;
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

			auto DiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid, totalVoxels](UniformGrid<StoreReal> &destinationGrid,
																									const UniformGrid<StoreReal> &sourceGrid)
			{
				assert(destinationGrid.size() == sourceGrid.size() &&
						sourceGrid.size() == domainCellLabels.size());

				tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
				{
					for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
					{
						Vec2i cell = domainCellLabels.unflatten(flatIndex);

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
											SolveReal(solverTolerance),
											maxIterations);
		}
	}
	else
	{
		// Solve using Eigen
		int interiorCellCount = 0;
		UniformGrid<int> solverIndices(domainCellLabels.size(), -1);

		forEachVoxelRange(Vec2i(0), domainCellLabels.size(), [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				solverIndices(cell) = interiorCellCount++;
		});

		// Build rows
		std::vector<Eigen::Triplet<SolveReal>> sparseElements;

		SolveReal gridScalar = 1. / Util::sqr(dx);
		forEachVoxelRange(Vec2i(0), domainCellLabels.size(), [&](const Vec2i &cell)
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

						sparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -gridScalar));
					}

				sparseElements.push_back(Eigen::Triplet<SolveReal>(index, index, 4. * gridScalar));
			}
			else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(index >= 0);

				SolveReal diagonal = 0;

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

							sparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -gridScalar));
							++diagonal;
						}
						else if (cellLabel == CellLabels::BOUNDARY_CELL)
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							Vec2i face = cellToFace(cell, axis, direction);
							SolveReal weight = boundaryWeights(face, axis);

							sparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -weight * gridScalar));
							diagonal += weight;
						}
						else if (cellLabel == CellLabels::DIRICHLET_CELL)
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex == -1);

							Vec2i face = cellToFace(cell, axis, direction);
							SolveReal weight = boundaryWeights(face, axis);
							
							diagonal += weight;
						}
						else
						{
							int adjacentIndex = solverIndices(adjacentCell);
							assert(adjacentIndex == -1);
							assert(cellLabel == CellLabels::EXTERIOR_CELL);
						}
					}

				sparseElements.push_back(Eigen::Triplet<SolveReal>(index, index, diagonal * gridScalar));
			}
			else assert(index == -1);
		});

		Eigen::SparseMatrix<SolveReal> sparseMatrix = Eigen::SparseMatrix<SolveReal>(interiorCellCount, interiorCellCount);
		sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
		sparseMatrix.makeCompressed();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal>, Eigen::Upper | Eigen::Lower> solver(sparseMatrix);
		assert(solver.info() == Eigen::Success);

		// Build RHS
		Vector rhsVector = Vector::Zero(interiorCellCount);

		tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(flatIndex);

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
		if (useRandomGuess)
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
			{
				for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
				{
					Vec2i cell = domainCellLabels.unflatten(flatIndex);

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

		solver.setTolerance(solverTolerance);
		solver.setMaxIterations(maxIterations);
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
	renderer = std::make_unique<Renderer>("Geometric CG and MG Preconditioner Test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);
	
	ScalarGrid<Real> tempGrid(Transform(dx, Vec2R(0)), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(flatIndex);

			tempGrid(cell) = Real(domainCellLabels(cell));
		}
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR_CELL), Real(CellLabels::BOUNDARY_CELL));

	renderer->run();
}