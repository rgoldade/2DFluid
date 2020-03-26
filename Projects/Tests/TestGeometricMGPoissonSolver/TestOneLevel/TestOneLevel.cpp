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

using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SimTools;

std::unique_ptr<Renderer> renderer;

static constexpr int gridSize = 512;
static constexpr bool useComplexDomain = true;
static constexpr bool useSolidSphere = true;

int main(int argc, char** argv)
{
	using namespace GeometricMultigridOperators;

	using StoreReal = double;
	using SolveReal = double;

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

	UniformGrid<StoreReal> rhsGrid(domainCellLabels.size(), 0);

	UniformGrid<StoreReal> solutionGrid(domainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
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

	auto l1Norm = [&](const UniformGrid<StoreReal> &grid)
	{
		assert(grid.size() == domainCellLabels.size());

		tbb::enumerable_thread_specific<SolveReal> parallelAccumulatedValue(0);
		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			auto& localAccumulatedValue = parallelAccumulatedValue.local();

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(cellIndex);

				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					localAccumulatedValue += fabs(grid(cell));
				}
			}
		});

		SolveReal accumulatedValue = 0;
		parallelAccumulatedValue.combine_each([&](const SolveReal localAccumulatedValue)
		{
			accumulatedValue += localAccumulatedValue;
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
	renderer = std::make_unique<Renderer>("One level correction test", Vec2i(pixelWidth, pixelHeight), Vec2f(0), 1, &argc, argv);

	ScalarGrid<float> tempGrid(Transform(dx, Vec2f(0)), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, tempGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = tempGrid.unflatten(flatIndex);

			tempGrid(cell) = float(domainCellLabels(cell));
		}
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), float(CellLabels::INTERIOR_CELL), float(CellLabels::BOUNDARY_CELL));

	renderer->run();
}