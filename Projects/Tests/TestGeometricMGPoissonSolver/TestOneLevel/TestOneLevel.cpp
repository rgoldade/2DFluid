#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <Eigen/Sparse>

#include "tbb/blocked_range.h"
#include "tbb/parallel_deterministic_reduce.h"
#include "tbb/parallel_for.h"

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

	UniformGrid<double> rhsGrid(domainCellLabels.size(), 0);

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

	auto l1Norm = [&](const UniformGrid<double> &grid)
	{
		assert(grid.size() == domainCellLabels.size());

		double accumulatedValue = tbb::parallel_deterministic_reduce(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double accumulatedValue) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					accumulatedValue += fabs(grid(cell));
				}
			}
			
			return accumulatedValue;
		},
		[](double a, double b) -> double
		{
			return a + b;
		});

		return accumulatedValue;
	};

	// Print initial guess
	solutionGrid.printAsOBJ("initialGuess");

	std::cout << "L-1 initial: " << l1Norm(solutionGrid) << std::endl;

	// Pre-build multigrid preconditioner
	GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx);
	
	for (int iteration = 0; iteration < 50; ++iteration)
	{
		mgSolver.applyMGVCycle(solutionGrid, rhsGrid, true /* use initial guess */);

		std::cout << "L-1 v-cycle " << iteration << ": " << l1Norm(solutionGrid) << std::endl;

		// Print corrected solution after one v-cycle
		//solutionGrid.printAsOBJ("solutionGrid" + std::to_string(iteration));
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	gRenderer = std::make_unique<Renderer>("One level correction test", Vec2i(pixelWidth, pixelHeight), Vec2d::Zero(), 1, &argc, argv);

	ScalarGrid<double> tempGrid(Transform(dx, Vec2d::Zero()), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, tempGrid.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = tempGrid.unflatten(flatIndex);

			tempGrid(cell) = double(domainCellLabels(cell));
		}
	});

	tempGrid.drawVolumetric(*gRenderer, Vec3d::Zero(), Vec3d::Ones(), double(CellLabels::INTERIOR_CELL), double(CellLabels::BOUNDARY_CELL));

	gRenderer->run();
}