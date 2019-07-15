#include <memory>
#include <random>
#include <string>

#include "Eigen/Sparse"

#include "Common.h"
#include "GeometricMGOperations.h"
#include "GeometricMGPoissonSolver.h"
#include "InitialMGTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"

std::unique_ptr<Renderer> renderer;

static const Vec2i interiorResolution(128);

static const bool useComplexDomain = true;
static const bool useSolidSphere = true;
static const Real sphereRadius = .125;

std::default_random_engine generator;
std::uniform_real_distribution<Real> distribution(0, 1);

int main(int argc, char** argv)
{
	using namespace GeometricMGOperations;

	UniformGrid<CellLabels> interiorDomainCellLabels;

	// Complex domain set up
	if (useComplexDomain)
		interiorDomainCellLabels = buildComplexDomain(interiorResolution,
												useSolidSphere,
												sphereRadius);
	// Simple domain set up
	else
		interiorDomainCellLabels = buildSimpleDomain(interiorResolution,
												1 /*dirichlet band*/);

	// Initialize grid with sin function
	Real dx = 1. / interiorResolution[0];

	Vec2i expandedOffset(2);
	Vec2i expandedResolution = interiorResolution + 2 * expandedOffset;

	// Build expanded domain cell labels
	UniformGrid<CellLabels> expandedDomainCellLabels(expandedResolution, CellLabels::EXTERIOR);

	forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
	{
		expandedDomainCellLabels(cell + expandedOffset) = interiorDomainCellLabels(cell);
	});


	UniformGrid<Real> rhsA(expandedResolution, 0);
	forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
	{
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Vec2R point = dx * Vec2R(cell);
			rhsA(cell) = distribution(generator);
		}
	});

	UniformGrid<Real> rhsB(expandedResolution, 0);
	forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
	{
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Vec2R point = dx * Vec2R(cell);
			rhsB(cell) = distribution(generator);
		}
	});

	Transform xform(dx, Vec2R(0));
	VectorGrid<Real> dummyWeights(xform, expandedResolution, 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), dummyWeights.size(axis), [&](const Vec2i& face)
		{
			bool isInterior = false;
			bool isExterior = false;
			for (int direction : {0, 1})
			{
				Vec2i cell = faceToCell(face, axis, direction);

				if (cell[axis] < 0 || cell[axis] >= expandedDomainCellLabels.size()[axis])
					continue;

				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
					isInterior = true;
				else if (expandedDomainCellLabels(cell) == CellLabels::EXTERIOR)
					isExterior = true;
			}

			if (isInterior && !isExterior)
				dummyWeights(face, axis) = 1;
		});
	}

	{
		UniformGrid<Real> solutionA(expandedResolution, 0);
		UniformGrid<Real> solutionB(expandedResolution, 0);

		// Test Jacobi symmetry
		dampedJacobiWeightedPoissonSmoother(solutionA,
			rhsA,
			expandedDomainCellLabels,
			dummyWeights,
			dx);

		dampedJacobiWeightedPoissonSmoother(solutionB,
			rhsB,
			expandedDomainCellLabels,
			dummyWeights,
			dx);


		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		UniformGrid<Real> solutionA(expandedResolution, 0);
		UniformGrid<Real> solutionB(expandedResolution, 0);

		// Test Jacobi symmetry
		dampedJacobiPoissonSmoother(solutionA,
			rhsA,
			expandedDomainCellLabels,
			dx);

		dampedJacobiPoissonSmoother(solutionB,
			rhsB,
			expandedDomainCellLabels,
			dx);


		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		UniformGrid<Real> solutionA(expandedResolution, 0);
		UniformGrid<Real> solutionB(expandedResolution, 0);

		std::vector<Vec2i> tempInteriorCells;
		forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
		{
			if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
			{
				tempInteriorCells.push_back(cell);
			}
		});

		dampedJacobiPoissonSmoother(solutionA,
			rhsA,
			expandedDomainCellLabels,
			tempInteriorCells,
			dx);

		dampedJacobiPoissonSmoother(solutionB,
			rhsB,
			expandedDomainCellLabels,
			tempInteriorCells,
			dx);

		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		UniformGrid<Real> solutionA(expandedResolution, 0);
		UniformGrid<Real> solutionB(expandedResolution, 0);

		std::vector<Vec2i> tempInteriorCells;
		forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
		{
			if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
			{
				tempInteriorCells.push_back(cell);
			}
		});

		dampedJacobiWeightedPoissonSmoother(solutionA,
			rhsA,
			expandedDomainCellLabels,
			tempInteriorCells,
			dummyWeights,
			dx);

		dampedJacobiWeightedPoissonSmoother(solutionB,
			rhsB,
			expandedDomainCellLabels,
			tempInteriorCells,
			dummyWeights,
			dx);

		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		// Test direct solve symmetry
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
		Eigen::SparseMatrix<Real> sparseMatrix;

		// Pre-build matrix at the coarsest level
		int interiorCellCount = 0;
		UniformGrid<int> directSolverIndices(expandedResolution, -1);
		{
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
					directSolverIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;

			Real gridScale = 1. / Util::sqr(dx);
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
				{
					int diagonal = 0;
					int index = directSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							auto cellLabels = expandedDomainCellLabels(adjacentCell);
							if (cellLabels == CellLabels::INTERIOR)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
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
			sparseMatrix = Eigen::SparseMatrix<Real>(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			myCoarseSolver.compute(sparseMatrix);

			assert(myCoarseSolver.info() == Eigen::Success);
		}

		UniformGrid<Real> solutionA(expandedResolution, 0);

		{
			Eigen::VectorXd coarseRHSVector = Eigen::VectorXd::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseRHSVector(index) = rhsA(cell);
				}
			});

			Eigen::VectorXd directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					solutionA(cell) = directSolution(index);
				}
			});
		}

		UniformGrid<Real> solutionB(expandedResolution, 0);

		{
			Eigen::VectorXd coarseRHSVector = Eigen::VectorXd::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseRHSVector(index) = rhsB(cell);
				}
			});

			Eigen::VectorXd directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			forEachVoxelRange(Vec2i(0), expandedResolution, [&](const Vec2i &cell)
			{
				if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					solutionB(cell) = directSolution(index);
				}
			});
		}

		// Compute dot products
		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		// Test down and up sampling
		UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);
		UniformGrid<Real> solutionA(expandedResolution, 0);
		Vec2i coarseResolution = coarseDomainLabels.size();
		
		{
			UniformGrid<Real> coarseRhs(coarseResolution, 0);
			downsample(coarseRhs, rhsA, coarseDomainLabels, expandedDomainCellLabels);
			upsampleAndAdd(solutionA, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
		}
		
		UniformGrid<Real> solutionB(expandedResolution, 0);
		
		{
			UniformGrid<Real> coarseRhs(coarseResolution, 0);
			downsample(coarseRhs, rhsB, coarseDomainLabels, expandedDomainCellLabels);
			upsampleAndAdd(solutionB, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
		}

		// Compute dot products
		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
		// Test single level correction
		UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);
		Vec2i coarseResolution = coarseDomainLabels.size();

		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
		Eigen::SparseMatrix<Real> sparseMatrix;

		// Pre-build matrix at the coarsest level
		int interiorCellCount = 0;
		UniformGrid<int> directSolverIndices(coarseResolution, -1);
		{
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
					directSolverIndices(cell) = interiorCellCount++;
			});

			// Build rows
			std::vector<Eigen::Triplet<double>> sparseElements;

			Real gridScale = 1. / Util::sqr(2 * dx);
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
				{
					int diagonal = 0;
					int index = directSolverIndices(cell);
					assert(index >= 0);
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							auto cellLabels = coarseDomainLabels(adjacentCell);
							if (cellLabels == CellLabels::INTERIOR)
							{
								int adjacentIndex = directSolverIndices(adjacentCell);
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

			sparseMatrix = Eigen::SparseMatrix<Real>(interiorCellCount, interiorCellCount);
			sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
			sparseMatrix.makeCompressed();

			myCoarseSolver.compute(sparseMatrix);

			assert(myCoarseSolver.info() == Eigen::Success);
		}

		// Transfer rhs to coarse rhs as if it was a residual with a zero initial guess
		UniformGrid<Real> solutionA(expandedResolution, 0);
		{
			// Pre-smooth to get an initial guess
			dampedJacobiWeightedPoissonSmoother(solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);

			// Compute new residual
			UniformGrid<Real> residualA(expandedResolution, 0);
			computeWeightedPoissonResidual(residualA, solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);

			UniformGrid<Real> coarseRhs(coarseResolution, 0);
			downsample(coarseRhs, residualA, coarseDomainLabels, expandedDomainCellLabels);

			Eigen::VectorXd coarseRHSVector = Eigen::VectorXd::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseRHSVector(index) = coarseRhs(cell);
				}
			});

			UniformGrid<Real> coarseSolution(coarseResolution, 0);

			Eigen::VectorXd directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseSolution(cell) = directSolution(index);
				}
			});

			upsampleAndAdd(solutionA, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

			dampedJacobiWeightedPoissonSmoother(solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);
		}

		UniformGrid<Real> solutionB(expandedResolution, 0);
		{
			// Pre-smooth to get an initial guess
			dampedJacobiWeightedPoissonSmoother(solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);

			// Compute new residual
			UniformGrid<Real> residualB(expandedResolution, 0);
			computeWeightedPoissonResidual(residualB, solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);

			UniformGrid<Real> coarseRhs(coarseResolution, 0);
			downsample(coarseRhs, residualB, coarseDomainLabels, expandedDomainCellLabels);

			Eigen::VectorXd coarseRHSVector = Eigen::VectorXd::Zero(interiorCellCount);
			// Copy to Eigen and direct solve
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseRHSVector(index) = coarseRhs(cell);
				}
			});

			UniformGrid<Real> coarseSolution(coarseResolution, 0);

			Eigen::VectorXd directSolution = myCoarseSolver.solve(coarseRHSVector);

			// Copy solution back
			forEachVoxelRange(Vec2i(0), coarseResolution, [&](const Vec2i &cell)
			{
				if (coarseDomainLabels(cell) == CellLabels::INTERIOR)
				{
					int index = directSolverIndices(cell);
					assert(index >= 0);

					coarseSolution(cell) = directSolution(index);
				}
			});

			upsampleAndAdd(solutionB, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

			dampedJacobiWeightedPoissonSmoother(solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);
		}

		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
		// Pre-build multigrid preconditioner
		GeometricMGPoissonSolver MGPreconditionerA(expandedDomainCellLabels, 4, dx);
		MGPreconditionerA.setGradientWeights(dummyWeights);

		UniformGrid<Real> solutionA(expandedResolution, 0);
		MGPreconditionerA.applyMGVCycle(solutionA, rhsA);
		MGPreconditionerA.applyMGVCycle(solutionA, rhsA, true);
		MGPreconditionerA.applyMGVCycle(solutionA, rhsA, true);
		MGPreconditionerA.applyMGVCycle(solutionA, rhsA, true);

		GeometricMGPoissonSolver MGPreconditionerB(expandedDomainCellLabels, 4, dx);
		MGPreconditionerB.setGradientWeights(dummyWeights);

		UniformGrid<Real> solutionB(expandedResolution, 0);
		MGPreconditionerB.applyMGVCycle(solutionB, rhsB);
		MGPreconditionerB.applyMGVCycle(solutionB, rhsB, true);
		MGPreconditionerB.applyMGVCycle(solutionB, rhsB, true);
		MGPreconditionerB.applyMGVCycle(solutionB, rhsB, true);

		Real dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
		Real dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

		assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	renderer = std::make_unique<Renderer>("MG Symmetry Test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);
	
	ScalarGrid<Real> tempGrid(xform, interiorResolution);

	forEachVoxelRange(Vec2i(0), interiorResolution, [&](const Vec2i &cell)
	{
		tempGrid(cell) = Real(interiorDomainCellLabels(cell));
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR), Real(CellLabels::DIRICHLET));

	renderer->run();
}