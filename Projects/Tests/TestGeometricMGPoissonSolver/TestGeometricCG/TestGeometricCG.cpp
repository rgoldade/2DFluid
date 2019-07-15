#include <memory>
#include <random>
#include <string>

#include "Common.h"
#include "GeometricCGPoissonSolver.h"
#include "GeometricMGPoissonSolver.h"
#include "GeometricMGOperations.h"
#include "InitialMGTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"

std::unique_ptr<Renderer> renderer;

static const Vec2i resolution(256);
static const bool useComplexDomain = false;
static const bool useSolidSphere = true;
static const Real sphereRadius = .125;

static const bool useMGPreconditioner = true;
static const int mgLevels = 4;

static const bool useRandomGuess = true;

std::default_random_engine generator;
std::uniform_real_distribution<Real> distribution(0, 1);

static const int maxSmootherIterations = 1000;
static const int deltaAmplitude = 1000;

int main(int argc, char** argv)
{
	using namespace GeometricMGOperations;

	UniformGrid<CellLabels> domainCellLabels;
	UniformGrid<Real> rhsGrid(resolution, 0);
	UniformGrid<Real> solutionGrid(resolution, 0);
	
	UniformGrid<Real> residualGrid(resolution, 0);
	// Complex domain set up
	if (useComplexDomain)
		domainCellLabels = buildComplexDomain(resolution, useSolidSphere, sphereRadius);
	// Simple domain set up
	else
		domainCellLabels = buildSimpleDomain(resolution, 1);

	if (useRandomGuess)
	{
		forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR)
				solutionGrid(cell) = distribution(generator);
		});
	}

	// Set delta function
	Real deltaPercent = .1;
	Vec2i deltaPoint = Vec2i(deltaPercent * Vec2R(resolution));

	forEachVoxelRange(deltaPoint - Vec2i(1), deltaPoint + Vec2i(2), [&](const Vec2i &cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR)
			rhsGrid(cell) = deltaAmplitude;
	});

	Real dx = 1. / resolution[0];

	Transform xform(dx, Vec2R(0));
	VectorGrid<Real> dummyWeights(xform, resolution, 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), dummyWeights.size(axis), [&](const Vec2i& face)
		{
			bool isInterior = false;
			bool isExterior = false;
			for (int direction : {0, 1})
			{
				Vec2i cell = faceToCell(face, axis, direction);

				if (cell[axis] < 0 || cell[axis] >= domainCellLabels.size()[axis])
					continue;

				if (domainCellLabels(cell) == CellLabels::INTERIOR)
					isInterior = true;
				else if (domainCellLabels(cell) == CellLabels::EXTERIOR)
					isExterior = true;
			}

			if (isInterior && !isExterior)
				dummyWeights(face, axis) = 1;
		});
	}

	auto MatrixVectorMultiply = [&domainCellLabels, &dummyWeights, dx](UniformGrid<Real> &residualGrid, const UniformGrid<Real> &solutionGrid)
	{
		// Matrix-vector multiplication
		applyWeightedPoissonMatrix(residualGrid, solutionGrid, domainCellLabels, dummyWeights, dx);
	};

	UniformGrid<Real> diagonalPrecondGrid(resolution, 0);

	forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
	{
		Real gridScalar = 1. / Util::sqr(dx);
		if (domainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Real diagonal = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);
					
					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= resolution[axis])
						continue;

					Vec2i face = cellToFace(cell, axis, direction);

					if (domainCellLabels(adjacentCell) == CellLabels::INTERIOR ||
						domainCellLabels(adjacentCell) == CellLabels::DIRICHLET)
					{
						diagonal += dummyWeights(face, axis);
					}
					else assert(dummyWeights(face, axis) == 0);
				}
			diagonal *= gridScalar;
			diagonalPrecondGrid(cell) = 1. / diagonal;
		}
	});

	auto DiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid](UniformGrid<Real> &solutionGrid,
																			const UniformGrid<Real> &rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		forEachVoxelRange(Vec2i(0), solutionGrid.size(), [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR)
			{
				solutionGrid(cell) = rhsGrid(cell) * diagonalPrecondGrid(cell);
			}
		});
	};

	// Pre-build multigrid preconditioner
	GeometricMGPoissonSolver MGPreconditioner(domainCellLabels, mgLevels, dx);
	MGPreconditioner.setGradientWeights(dummyWeights);

	auto MultiGridPreconditioner = [&MGPreconditioner](UniformGrid<Real> &solutionGrid,
														const UniformGrid<Real> &rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		MGPreconditioner.applyMGVCycle(solutionGrid, rhsGrid);
	};

	auto DotProduct = [&domainCellLabels](const UniformGrid<Real> &grid0,
											const UniformGrid<Real> &grid1) -> Real
	{
		assert(grid0.size() == grid1.size());
		return dotProduct(grid0, grid1, domainCellLabels);
	};

	auto L2Norm = [&domainCellLabels](const UniformGrid<Real> &grid) -> Real
	{
		return l2Norm(grid, domainCellLabels);
	};

	auto AddScaledVector = [&domainCellLabels](UniformGrid<Real> &destination,
												const UniformGrid<Real> &unscaledSource,
												const UniformGrid<Real> &scaledSource,
												const Real scale)
	{
		addToVector(destination, unscaledSource, scaledSource, domainCellLabels, scale);
	};

	if (useMGPreconditioner)
	{
		solveGeometricCGPoisson(solutionGrid,
								rhsGrid,
								MatrixVectorMultiply,
								MultiGridPreconditioner,
								DotProduct,
								L2Norm,
								AddScaledVector,
								1E-5 /*solver tolerance*/,
								true, false, true);
	}
	else
	{
		solveGeometricCGPoisson(solutionGrid,
								rhsGrid,
								MatrixVectorMultiply,
								DiagonalPreconditioner,
								DotProduct,
								L2Norm,
								AddScaledVector,
								1E-5 /*solver tolerance*/,
								true, false, true);
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	renderer = std::make_unique<Renderer>("Geometric CG and MG Preconditioner Test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);
	
	ScalarGrid<Real> tempGrid(xform, resolution);

	forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
	{
		tempGrid(cell) = Real(domainCellLabels(cell));
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR), Real(CellLabels::DIRICHLET));

	renderer->run();
}