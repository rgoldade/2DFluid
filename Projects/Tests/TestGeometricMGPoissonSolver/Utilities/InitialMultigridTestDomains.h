#ifndef TESTS_GEOMETRIC_MULTIGRID_UTILITIES_H
#define TESTS_GEOMETRIC_MULTIGRID_UTILITIES_H

#include "tbb/tbb.h"

#include "ComputeWeights.h"
#include "GeometricMultigridOperators.h"
#include "LevelSet.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::Utilities;

using GeometricMultigridOperators::CellLabels;

template<typename StoreReal>
std::pair<Vec2i, int> buildExpandedDomain(UniformGrid<CellLabels> &expandedDomainCellLabels,
											VectorGrid<StoreReal> &expandedBoundaryWeights,
											const UniformGrid<CellLabels> &baseDomainCellLabels,
											const VectorGrid<StoreReal> &baseBoundaryWeights)
{
	std::pair<Vec2i, int> mgSettings = buildExpandedDomainLabels(expandedDomainCellLabels,
																		baseDomainCellLabels);

	Vec2i exteriorOffset = mgSettings.first;
	int mgLevels = mgSettings.second;

	Transform xform(baseBoundaryWeights.dx(), baseBoundaryWeights.xform().offset() - baseBoundaryWeights.dx() * Vec2f(exteriorOffset));
	expandedBoundaryWeights = VectorGrid<StoreReal>(xform, expandedDomainCellLabels.size(), 0, VectorGridSettings::SampleType::STAGGERED);
	// Build expanded boundary weights
	for (int axis : {0,1})
		buildExpandedBoundaryWeights(expandedBoundaryWeights, baseBoundaryWeights, expandedDomainCellLabels, exteriorOffset, axis);

	// Build boundary cells
	setBoundaryDomainLabels(expandedDomainCellLabels, expandedBoundaryWeights);
	
	assert(unitTestBoundaryCells(expandedDomainCellLabels, &expandedBoundaryWeights));
	assert(unitTestExteriorCells(expandedDomainCellLabels));

	return mgSettings;
}

template<typename StoreReal>
void buildComplexDomain(UniformGrid<CellLabels> &domainCellLabels,
						VectorGrid<StoreReal> &boundaryWeights,
						const int gridSize,
						const bool useSolidSphere)
{
	assert(gridSize > 0);
	domainCellLabels.resize(Vec2i(gridSize), CellLabels::EXTERIOR_CELL);

	StoreReal dx = 1. / StoreReal(gridSize);

	Transform xform(dx, Vec2f(0));

	auto dirichletIsoSurface = [](const Vec2f &point)
	{
		return point[0] - .5 + .25 * std::sin(2. * PI * point[1]);
	};

	LevelSet dirichletSurface(xform, Vec2i(gridSize));

	tbb::parallel_for(tbb::blocked_range<int>(0, dirichletSurface.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = dirichletSurface.unflatten(flatIndex);

			Vec2f point = dirichletSurface.indexToWorld(Vec2f(cell));
			dirichletSurface(cell) = dirichletIsoSurface(point);
		}
	});

	auto sphereIsoSurface = [](const Vec2f &point)
	{
		const Vec2f sphereCenter(.5);
		constexpr StoreReal sphereRadius = .125;

		return mag2(point - sphereCenter) - sqr(sphereRadius);
	};

	boundaryWeights = VectorGrid<StoreReal>(xform, Vec2i(gridSize), 1, VectorGridSettings::SampleType::STAGGERED);

	// Compute cut-cell weights
	if (useSolidSphere)
	{
		LevelSet solidSphereSurface(xform, Vec2i(gridSize), 5);

		tbb::parallel_for(tbb::blocked_range<int>(0, gridSize * gridSize, tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = solidSphereSurface.unflatten(flatIndex);

				Vec2f point = solidSphereSurface.indexToWorld(Vec2f(cell));
				solidSphereSurface(cell) = sphereIsoSurface(point);
			}
		});

		VectorGrid<float> cutCellWeights = computeCutCellWeights(solidSphereSurface, true);

		for (int axis : {0, 1})
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.size(axis)[0] * boundaryWeights.size(axis)[1], tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
			{
				for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
				{
					Vec2i face = boundaryWeights.grid(axis).unflatten(flatIndex);
					boundaryWeights(face, axis) = cutCellWeights(face, axis);						
				}
			});
		}
	}

	// Set weights along volume limits to zero
	for (int axis : {0, 1})
		for (int direction : {0, 1})
		{
			Vec2i startFace(0);
			Vec2i endFace = boundaryWeights.size(axis);

			if (direction == 0)
				endFace[axis] = 1;
			else
				startFace[axis] = endFace[axis] - 1;

			forEachVoxelRange(startFace, endFace, [&](const Vec2i &face)
			{
				boundaryWeights(face, axis) = 0;
			});
		}

	tbb::parallel_for(tbb::blocked_range<int>(0, gridSize * gridSize, tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = domainCellLabels.unflatten(flatIndex);

			// Make sure there is an open cut-cell face
			bool hasOpenFace = false;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					if (boundaryWeights(face, axis) > 0)
						hasOpenFace = true;
				}

			if (hasOpenFace)
			{
				// Sine wave for air-liquid boundary.
				float sdf = dirichletSurface(cell);
				if (sdf > 0)
					domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
				else
					domainCellLabels(cell) = CellLabels::INTERIOR_CELL;
			}
			else assert(domainCellLabels(cell) == CellLabels::EXTERIOR_CELL);
		}
	});

	VectorGrid<float> ghostFluidWeights = computeGhostFluidWeights(dirichletSurface);
	// Build ghost fluid weights
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.size(axis)[0] * boundaryWeights.size(axis)[1], tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = boundaryWeights.grid(axis).unflatten(flatIndex);

				if (boundaryWeights(face, axis) > 0)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					assert(backwardCell[axis] >= 0 && forwardCell[axis] < domainCellLabels.size()[axis]);

					auto backwardLabel = domainCellLabels(backwardCell);
					auto forwardLabel = domainCellLabels(forwardCell);

					assert(backwardLabel != CellLabels::EXTERIOR_CELL && forwardLabel != CellLabels::EXTERIOR_CELL);

					if (backwardLabel == CellLabels::DIRICHLET_CELL && forwardLabel == CellLabels::DIRICHLET_CELL)
						boundaryWeights(face, axis) = 0;
					else
					{
						if (backwardLabel == CellLabels::DIRICHLET_CELL || forwardLabel == CellLabels::DIRICHLET_CELL)
						{
							float backwardSDF = dirichletSurface(backwardCell);
							float forwardSDF = dirichletSurface(forwardCell);

							// Sine wave for air-liquid boundary.
							assert((backwardSDF > 0 && forwardSDF <= 0) || (backwardSDF <= 0 && forwardSDF > 0));

							StoreReal theta = ghostFluidWeights(face, axis);
							theta = clamp(theta, StoreReal(.01), StoreReal(1));

							boundaryWeights(face, axis) /= theta;
						}
					}
				}
			}
		});
	}
}

template<typename StoreReal>
void buildSimpleDomain(UniformGrid<CellLabels> &domainCellLabels,
						VectorGrid<StoreReal> &boundaryWeights,
						const int gridSize,
						const int dirichletBand)
{
	assert(gridSize > 0);
	assert(dirichletBand >= 0);

	domainCellLabels.resize(Vec2i(gridSize), CellLabels::EXTERIOR_CELL);

	StoreReal dx = 1. / StoreReal(gridSize);

	// Set outer layers to DIRICHLET

	// Set bottom face
	Vec2i start(0);
	Vec2i end(gridSize, dirichletBand);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
	});

	// Set top face
	start = Vec2i(0, gridSize - dirichletBand);
	end = Vec2i(gridSize);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
	});

	// Set left face
	start = Vec2i(0);
	end = Vec2i(dirichletBand, gridSize);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
	});

	// Set right face

	start = Vec2i(gridSize - dirichletBand, 0);
	end = Vec2i(gridSize);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
	});

	start = Vec2i(dirichletBand);
	end = Vec2i(gridSize - dirichletBand);

	forEachVoxelRange(start, end, [&](const Vec2i &cell)
	{
		domainCellLabels(cell) = CellLabels::INTERIOR_CELL;
	});

	// TODO: use proper ghost fluid and cut-cell weights
	boundaryWeights = VectorGrid<StoreReal>(Transform(dx, Vec2f(0)), Vec2i(gridSize), 0, VectorGridSettings::SampleType::STAGGERED);

	// Build boundary weights
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.size(axis)[0] * boundaryWeights.size(axis)[1], tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = boundaryWeights.grid(axis).unflatten(flatIndex);
				bool isInterior = false;
				bool isExterior = false;
				for (int direction : {0, 1})
				{
					Vec2i cell = faceToCell(face, axis, direction);

					if (cell[axis] < 0 || cell[axis] >= domainCellLabels.size()[axis])
					{
						isExterior = true;
						continue;
					}

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
						isInterior = true;
					else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
						isExterior = true;
				}

				if (isInterior && !isExterior)
					boundaryWeights(face, axis) = 1;
			}
		});
	}
}

#endif