#include <memory>
#include <random>
#include <string>

#include "Eigen/Sparse"

#include "Common.h"
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

	int totalVoxels = domainCellLabels.size()[0] * domainCellLabels.size()[1];
	tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(flatIndex);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				Vec2R point = dx * Vec2R(cell);
				solutionGrid(cell) = 4. * (std::sin(2 * Util::PI * point[0]) * std::sin(2 * Util::PI * point[1]) +
										std::sin(4 * Util::PI * point[0]) * std::sin(4 * Util::PI * point[1]));
			}
		}
	});

	auto l1Norm = [&](const UniformGrid<StoreReal> &grid)
	{
		assert(grid.size() == domainCellLabels.size());

		using ParallelReal = tbb::enumerable_thread_specific<SolveReal>;
		ParallelReal parallelAccumulatedValue(0);
		tbb::parallel_for(tbb::blocked_range<int>(0, totalVoxels, tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			auto &localAccumulatedValue = parallelAccumulatedValue.local();

			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = domainCellLabels.unflatten(flatIndex);

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
	renderer = std::make_unique<Renderer>("One level correction test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);

	ScalarGrid<Real> tempGrid(Transform(dx, Vec2R(0)), domainCellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, tempGrid.size()[0] * tempGrid.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = tempGrid.unflatten(flatIndex);

			tempGrid(cell) = Real(domainCellLabels(cell));
		}
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR_CELL), Real(CellLabels::BOUNDARY_CELL));

	renderer->run();
}