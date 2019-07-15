#include <memory>
#include <random>
#include <string>

#include "Common.h"
#include "GeometricMGOperations.h"
#include "InitialMGTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"

std::unique_ptr<Renderer> renderer;

static const Vec2i resolution(64);
static const bool useComplexDomain = true;
static const bool useSolidSphere = true;
static const Real sphereRadius = .125;

static const bool useGaussSeidelSmoother = true;

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
		domainCellLabels = buildComplexDomain(resolution,
												useSolidSphere,
												sphereRadius);
	// Simple domain set up
	else
		domainCellLabels = buildSimpleDomain(resolution,
												1 /*dirichlet band*/);
	if (useRandomGuess)
	{
		forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR)
			{
				solutionGrid(cell) = distribution(generator);
			}
		});
	}

	// Set delta function
	Real deltaPercent = .1;
	Vec2i deltaPoint = Vec2i(deltaPercent * Vec2R(resolution));

	forEachVoxelRange(deltaPoint - Vec2i(1), deltaPoint + Vec2i(2), [&](const Vec2i &cell)
	{
		rhsGrid(cell) = deltaAmplitude;
	});
	
	Real oldLInfinityError = std::numeric_limits<Real>::max();
	Real oldL2Error = oldLInfinityError;

	Real dx = 1. / resolution[0];

	for (int iteration = 0; iteration < maxSmootherIterations; ++iteration)
	{
		dampedJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx);

		computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

		// Print out residual as a height field
		residualGrid.printAsOBJ("smootherResidual" + std::to_string(iteration + 1));
		
		// Print solution grid
		solutionGrid.printAsOBJ("solutionGrid" + std::to_string(iteration + 1));

		// Print out L-infinity norm;
		Real LInfinityError = 0;
		Real L2Error = 0;
		forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
		{
			if (domainCellLabels(cell) == CellLabels::INTERIOR)
			{
				LInfinityError = std::max(LInfinityError, fabs(residualGrid(cell)));
				L2Error += Util::sqr(residualGrid(cell));
			}
		});

		if (oldLInfinityError < LInfinityError)
			std::cout << "L-Infinity error didn't decrease" << std::endl;
		if (oldL2Error < L2Error)
			std::cout << "L-2 error didn't decrease" << std::endl;

		oldLInfinityError = LInfinityError;
		oldL2Error = L2Error;
		std::cout << "Iteration: " << iteration << ". L-infinity error: " << LInfinityError << ". L-2 error: " << L2Error << std::endl;
	}

	// Print domain labels to make sure they are set up correctly
	int pixelHeight = 1080;
	int pixelWidth = pixelHeight;
	renderer = std::make_unique<Renderer>("MG Smoother Test", Vec2i(pixelWidth, pixelHeight), Vec2R(0), 1, &argc, argv);
	
	Transform xform(1. / resolution[0], Vec2R(0));
	ScalarGrid<Real> tempGrid(xform, resolution);

	forEachVoxelRange(Vec2i(0), resolution, [&](const Vec2i &cell)
	{
		tempGrid(cell) = Real(domainCellLabels(cell));
	});

	tempGrid.drawVolumetric(*renderer, Vec3f(0), Vec3f(1), Real(CellLabels::INTERIOR), Real(CellLabels::DIRICHLET));

	renderer->run();
}