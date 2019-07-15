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

static const Vec2i resolution(128);

static const bool useComplexDomain = true;
static const bool useSolidSphere = true;
static const Real sphereRadius = .125;

std::default_random_engine generator;
std::uniform_real_distribution<Real> distribution(0, 1);

int main(int argc, char** argv)
{
	using namespace GeometricMGOperations;

	UniformGrid<CellLabels> domainCellLabels;
	UniformGrid<Real> rhsGrid(resolution, 0);

	// Complex domain set up
	if (useComplexDomain)
		domainCellLabels = buildComplexDomain(resolution,
												useSolidSphere,
												sphereRadius);
	// Simple domain set up
	else
		domainCellLabels = buildSimpleDomain(resolution,
												1 /*dirichlet band*/);

	// Initialize grid with sin function
	Real dx = 1. / resolution[0];

	UniformGrid<Real> initialGuess(resolution, 0);
	forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
	{
		if (domainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Vec2R point = dx * Vec2R(cell);
			initialGuess(cell) = std::sin(2 * Util::PI * point[0]) * std::sin(2 * Util::PI * point[1]) +
									std::sin(4 * Util::PI * point[0]) * std::sin(4 * Util::PI * point[1]);
		}
	});

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

	// Print initial guess
	initialGuess.printAsOBJ("initialGuess");

	// Pre-build multigrid preconditioner
	GeometricMGPoissonSolver MGPreconditioner(domainCellLabels, 2, dx);
	MGPreconditioner.setGradientWeights(dummyWeights);

	MGPreconditioner.applyMGVCycle(initialGuess, rhsGrid, true);

	// Print corrected solution after one v-cycle
	initialGuess.printAsOBJ("solutionGrid");

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	renderer = std::make_unique<Renderer>("One Level MG V-cycle Test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);
	
	ScalarGrid<Real> tempGrid(xform, resolution);

	forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
	{
		tempGrid(cell) = Real(domainCellLabels(cell));
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR), Real(CellLabels::DIRICHLET));

	renderer->run();
}