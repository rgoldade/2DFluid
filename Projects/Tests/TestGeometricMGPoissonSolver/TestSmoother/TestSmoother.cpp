#include <iostream>
#include <memory>
#include <random>
#include <string>

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

static constexpr bool gUseRandomGuess = true;

static constexpr int gMaxIterations = 1000;
static constexpr double gDeltaAmplitude = 1000;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

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
		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(0, 1);

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

	forEachVoxelRange(deltaPoint - Vec2i::Ones(), deltaPoint + Vec2i(2, 2), [&](const Vec2i& cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			rhsGrid(cell) = gDeltaAmplitude;
	});
	
	double oldLInfinityError = std::numeric_limits<double>::max();
	double oldL2Error = oldLInfinityError;

	VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

	std::cout.precision(10);

	for (int iteration = 0; iteration < gMaxIterations; ++iteration)
	{
		boundaryJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx, &boundaryWeights);

		interiorJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

		boundaryJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx, &boundaryWeights);

		computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

		double lInfinityError = lInfinityNorm(residualGrid, domainCellLabels);
		double l2Error = squaredl2Norm(residualGrid, domainCellLabels);

		if (oldLInfinityError < lInfinityError)
			std::cout << "L-Infinity error didn't decrease" << std::endl;
		if (oldL2Error < l2Error)
			std::cout << "L-2 error didn't decrease" << std::endl;

		oldLInfinityError = lInfinityError;
		oldL2Error = l2Error;
		std::cout << "Iteration: " << iteration << ". L-infinity error: " << lInfinityError << ". L-2 error: " << std::sqrt(l2Error) << std::endl;
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	gRenderer = std::make_unique<Renderer>("Smoother Test", Vec2i(pixelWidth, pixelHeight), Vec2d::Zero(), 1, &argc, argv);

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