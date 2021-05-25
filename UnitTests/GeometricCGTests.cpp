#include "gtest/gtest.h"

#include <Eigen/Sparse>

#include "GeometricConjugateGradientSolver.h"
#include "GeometricMultigridOperators.h"
#include "GeometricMultigridPoissonSolver.h"
#include "InitialMultigridTestDomains.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;
using namespace GeometricMultigridOperators;

static VectorXd eigenSolve(const UniformGrid<CellLabels>& domainCellLabels, const VectorGrid<double>& boundaryWeights, const UniformGrid<double>& rhsGrid, double dx)
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
			ASSERT_TRUE(index >= 0);

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					auto adjacentLabels = domainCellLabels(adjacentCell);

					ASSERT_TRUE(adjacentLabels == CellLabels::INTERIOR_CELL ||
						adjacentLabels == CellLabels::BOUNDARY_CELL);

					int adjacentIndex = solverIndices(adjacentCell);
					ASSERT_TRUE(adjacentIndex >= 0);

					Vec2i face = cellToFace(cell, axis, direction);
					ASSERT_TRUE(boundaryWeights(face, axis) == 1);

					sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
				}

			sparseElements.emplace_back(index, index, 4. * gridScalar);
		}
		else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		{
			ASSERT_TRUE(index >= 0);

			double diagonal = 0;

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					auto cellLabel = domainCellLabels(adjacentCell);

					if (cellLabel == CellLabels::INTERIOR_CELL)
					{
						int adjacentIndex = solverIndices(adjacentCell);
						ASSERT_TRUE(adjacentIndex >= 0);

						Vec2i face = cellToFace(cell, axis, direction);
						ASSERT_TRUE(boundaryWeights(face, axis) == 1);

						sparseElements.emplace_back(index, adjacentIndex, -gridScalar);
						++diagonal;
					}
					else if (cellLabel == CellLabels::BOUNDARY_CELL)
					{
						int adjacentIndex = solverIndices(adjacentCell);
						ASSERT_TRUE(adjacentIndex >= 0);

						Vec2i face = cellToFace(cell, axis, direction);
						double weight = boundaryWeights(face, axis);

						sparseElements.emplace_back(index, adjacentIndex, -weight * gridScalar);
						diagonal += weight;
					}
					else if (cellLabel == CellLabels::DIRICHLET_CELL)
					{
						int adjacentIndex = solverIndices(adjacentCell);
						ASSERT_TRUE(adjacentIndex == -1);

						Vec2i face = cellToFace(cell, axis, direction);
						double weight = boundaryWeights(face, axis);

						diagonal += weight;
					}
					else
					{
						int adjacentIndex = solverIndices(adjacentCell);
						ASSERT_TRUE(adjacentIndex == -1);
						ASSERT_TRUE(cellLabel == CellLabels::EXTERIOR_CELL);
					}
				}

			sparseElements.emplace_back(index, index, diagonal * gridScalar);
		}
		else ASSERT_TRUE(index == -1);
	});

	Eigen::SparseMatrix<double> sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
	sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver(sparseMatrix);
	EXPECT_TRUE(solver.info() == Eigen::Success);

	// Build RHS
	VectorXd rhsVector = VectorXd::Zero(interiorCellCount);

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(cellIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				int index = solverIndices(cell);
				ASSERT_TRUE(index >= 0);
				rhsVector(index) = rhsGrid(cell);
			}
		}
	});

	VectorXd solutionVector = solver.solve(rhsVector);

	return solutionVector;
}

static void solverTests(bool useComplexDomain, bool useSolidSphere, bool useMGPreconditioner, int gridSize)
{
	UniformGrid<CellLabels> domainCellLabels;
	VectorGrid<double> boundaryWeights;
	int mgLevels;
	Vec2i exteriorOffset;
	{
		UniformGrid<CellLabels> baseDomainCellLabels;
		VectorGrid<double> baseBoundaryWeights;

		// Complex domain set up
		if (useComplexDomain)
			buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, useSolidSphere);
		// Simple domain set up
		else
			buildSimpleDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, 1 /*dirichlet band*/);

		// Build expanded domain
		std::pair<Vec2i, int> mgSettings = buildExpandedDomain(domainCellLabels, boundaryWeights, baseDomainCellLabels, baseBoundaryWeights);

		exteriorOffset = mgSettings.first;
		mgLevels = mgSettings.second;
	}

	double dx = boundaryWeights.dx();

	UniformGrid<double> rhsGrid(domainCellLabels.size(), 0);
	UniformGrid<double> solutionGrid(domainCellLabels.size(), 0);
	UniformGrid<double> residualGrid(domainCellLabels.size(), 0);

	// Set delta function
	double deltaPercent = .1;
	double deltaAmplitude = 1000;
	Vec2i deltaPoint = (deltaPercent * Vec2d(gridSize, gridSize)).cast<int>() + exteriorOffset;

	forEachVoxelRange(deltaPoint - Vec2i::Ones(), deltaPoint + Vec2i(2, 2), [&](const Vec2i& cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			rhsGrid(cell) = deltaAmplitude;
	});

	VectorXd eigenSolution = eigenSolve(domainCellLabels, boundaryWeights, rhsGrid, dx);

	// Build components for geometric solver
	auto MatrixVectorMultiply = [&domainCellLabels, &boundaryWeights, dx](UniformGrid<double> &destinationGrid, const UniformGrid<double> &sourceGrid)
	{
		ASSERT_TRUE(destinationGrid.size() == sourceGrid.size() && sourceGrid.size() == domainCellLabels.size());

		// Matrix-vector multiplication
		applyPoissonMatrix(destinationGrid, sourceGrid, domainCellLabels, dx, &boundaryWeights);
	};

	auto DotProduct = [&domainCellLabels](const UniformGrid<double> &grid0, const UniformGrid<double> &grid1) -> double
	{
        EXPECT_TRUE(grid0.size()[0] == grid1.size()[0]);
        EXPECT_TRUE(grid0.size()[1] == grid1.size()[1]);
        EXPECT_TRUE(grid1.size()[0] == domainCellLabels.size()[0]);
        EXPECT_TRUE(grid1.size()[1] == domainCellLabels.size()[1]);

		return dotProduct(grid0, grid1, domainCellLabels);
	};

	auto SquaredL2Norm = [&domainCellLabels](const UniformGrid<double> &grid) -> double
	{
        EXPECT_TRUE(grid.size()[0] == domainCellLabels.size()[0]);
        EXPECT_TRUE(grid.size()[1] == domainCellLabels.size()[1]);
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

	double tolerance = 1e-12;
	int iterations = 3000;

	if (useMGPreconditioner)
	{
		// Pre-build multigrid preconditioner
		GeometricMultigridPoissonSolver mgPreconditioner(domainCellLabels, boundaryWeights, mgLevels, dx);

		auto MultiGridPreconditioner = [&mgPreconditioner, &domainCellLabels](UniformGrid<double> &destinationGrid,
			const UniformGrid<double> &sourceGrid)
		{
			ASSERT_TRUE(destinationGrid.size()[0] == sourceGrid.size()[0]);
			ASSERT_TRUE(destinationGrid.size()[1] == sourceGrid.size()[1]);
			ASSERT_TRUE(sourceGrid.size()[0] == domainCellLabels.size()[0]);
			ASSERT_TRUE(sourceGrid.size()[1] == domainCellLabels.size()[1]);

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
			tolerance,
			iterations);
	}
	else
	{
		UniformGrid<double> diagonalPrecondGrid(domainCellLabels.size(), 0);

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
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
			tolerance,
			iterations);
	}

	// Solve using Eigen
    int interiorCellCount = 0;
    UniformGrid<int> solverIndices(domainCellLabels.size(), -1);

    forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i& cell) {
        if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
            solverIndices(cell) = interiorCellCount++;
    });

	VectorXd gridSolution(interiorCellCount);

	// Compare result to Eigen-based solver
	forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i& cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
        {
            int row = solverIndices(cell);
            gridSolution(row) = solutionGrid(cell);
        }
	});

	for (int row = 0; row < gridSolution.rows(); ++row)
    {
        if (!isNearlyEqual(gridSolution(row), eigenSolution(row), 1e-6, true))
        {
            std::cout << std::setprecision(15);
            std::cout << "Check failed" << std::endl;
			std::cout << gridSolution(row) << std::endl;
            std::cout << eigenSolution(row) << std::endl;
		}
        EXPECT_TRUE(isNearlyEqual(gridSolution(row), eigenSolution(row), 1e-6, true)) << "Eigen: " << eigenSolution(row) << ". Grid: " << gridSolution(row);
	}
}

TEST(GEOMETRIC_CG_TESTS, MULTIGRID_PRECONDITIONER_TEST)
{
	std::vector<bool> complexDomainSettings = { false, true };
	std::vector<bool> solidSphereSettings = { false, true };

	const int startGrid = 16;
	const int endGrid = startGrid * int(pow(2, 3));

	for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
		for (bool useComplexDomain : complexDomainSettings)
		{
			if (useComplexDomain)
			{
				for (bool useSolidSphere : solidSphereSettings)
				{
					solverTests(useComplexDomain, useSolidSphere, true /*use MG precon*/, gridSize);
				}
			}
			else
			{
				solverTests(useComplexDomain, false, true /*use MG precon*/, gridSize);
			}
		}
}

TEST(GEOMETRIC_CG_TESTS, DIAGONAL_PRECONDITIONER_TEST)
{
	std::vector<bool> complexDomainSettings = { false, true };
	std::vector<bool> solidSphereSettings = { false, true };

	const int startGrid = 16;
	const int endGrid = startGrid * int(pow(2, 3));

	for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
		for (bool useComplexDomain : complexDomainSettings)
		{
			if (useComplexDomain)
			{
				for (bool useSolidSphere : solidSphereSettings)
				{
					solverTests(useComplexDomain, useSolidSphere, false /*use MG precon*/, gridSize);
				}
			}
			else
			{
				solverTests(useComplexDomain, false, false /*use MG precon*/, gridSize);
			}
		}
}