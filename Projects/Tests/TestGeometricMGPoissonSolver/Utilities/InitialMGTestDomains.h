#ifndef TESTS_GEOMETRIC_MG_UTILITIES_H
#define TESTS_GEOMETRIC_MG_UTILITIES_H

#include "Common.h"
#include "GeometricMGOperations.h"
#include "UniformGrid.h"

using namespace GeometricMGOperations;

UniformGrid<CellLabels> buildComplexDomain(const Vec2i &resolution,
											const bool useSolidSphere,
											const Real sphereRadius)
{
	UniformGrid<CellLabels> domainCellLabels(resolution, CellLabels::EXTERIOR);

	Vec2R sphereCenter(.5);

	Vec2R dx(1. / resolution[0], 1. / resolution[1]);

	// Maintain a single band of exterior cells around the domain
	forEachVoxelRange(Vec2i(1), resolution - Vec2i(1), [&](const Vec2i &cell)
	{
		Vec2R point = dx * Vec2R(cell);

		// Solid sphere at center of domain
		if (useSolidSphere)
		{
			if (mag2(point - sphereCenter) - Util::sqr(sphereRadius) < 0)
			{
				domainCellLabels(cell) = CellLabels::EXTERIOR;
				return;
			}
		}

		// Sine wave for air-liquid boundary.
		Real sdf = point[0] - .5 + .25 * std::sin(2. * Util::PI * point[1]);
		if (sdf > 0)
			domainCellLabels(cell) = CellLabels::DIRICHLET;
		else
			domainCellLabels(cell) = CellLabels::INTERIOR;
	});

	return domainCellLabels;
}

UniformGrid<CellLabels> buildSimpleDomain(const Vec2i &resolution,
											int dirichletBand)
{
	assert(resolution[0] % 2 == 0 && resolution[1] % 2 == 0);

	dirichletBand = std::max(0, dirichletBand);

	UniformGrid<CellLabels> domainCellLabels(resolution, CellLabels::EXTERIOR);

	// Set outer layers to DIRICHLET
	Vec2i start(0);
	Vec2i end(resolution[0], dirichletBand);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET;
	});

	start = Vec2i(0, resolution[1] - dirichletBand);
	end = Vec2i(resolution[0], resolution[1]);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET;
	});

	start = Vec2i(0);
	end = Vec2i(dirichletBand, resolution[1]);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET;
	});

	start = Vec2i(resolution[0] - dirichletBand, 0);
	end = Vec2i(resolution[0], resolution[1]);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET;
	});

	start = Vec2i(dirichletBand);
	end = Vec2i(resolution[0] - dirichletBand, resolution[1] - dirichletBand);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::INTERIOR;
	});

	return domainCellLabels;
}

#endif