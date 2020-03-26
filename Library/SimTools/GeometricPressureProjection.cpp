#include "GeometricPressureProjection.h"

#include <iostream>

#include "tbb/tbb.h"

#include "GeometricConjugateGradientSolver.h"
#include "GeometricMultigridPoissonSolver.h"

namespace FluidSim2D::SimTools
{

GeometricPressureProjection::GeometricPressureProjection(const LevelSet& surface,
	const VectorGrid<float>& cutCellWeights,
	const VectorGrid<float>& ghostFluidWeights,
	const VectorGrid<float>& solidVelocity)
	: mySurface(surface)
	, myCutCellWeights(cutCellWeights)
	, myGhostFluidWeights(ghostFluidWeights)
	, mySolidVelocity(solidVelocity)
	, myUseInitialGuessPressure(false)
	, myInitialGuessPressure(nullptr)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled surface and collision

	assert(solidVelocity.sampleType() == VectorGridSettings::SampleType::STAGGERED);

#if !defined(NDEBUG)
	for (int axis : {0, 1})
	{
		Vec2i faceCount = solidVelocity.size(axis);

		Vec2i cellSize = faceCount;
		--cellSize[axis];

		assert(cellSize == surface.size());
	}
#endif

	assert(solidVelocity.isGridMatched(cutCellWeights) &&
		solidVelocity.isGridMatched(ghostFluidWeights));

	myPressure = ScalarGrid<float>(surface.xform(), surface.size(), 0);
	myValidFaces = VectorGrid<VisitedCellLabels>(surface.xform(), surface.size(), VisitedCellLabels::UNVISITED_CELL, VectorGridSettings::SampleType::STAGGERED);
}

void GeometricPressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .25, 1, 2);
}

void GeometricPressureProjection::project(VectorGrid<float>& velocity,
	bool useMGPreconditioner)
{
	using GeometricMultigridOperators::CellLabels;

	assert(velocity.isGridMatched(mySolidVelocity));

	using MGCellLabels = GeometricMultigridOperators::CellLabels;

	UniformGrid<CellLabels> baseDomainCellLabels(mySurface.size(), MGCellLabels::EXTERIOR_CELL);

	bool hasDirichletCell = false;
	// Build domain labels
	tbb::parallel_for(tbb::blocked_range<int>(0, mySurface.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mySurface.unflatten(cellIndex);

				bool isFluidCell = false;

				for (int axis = 0; axis < 2 && !isFluidCell; ++axis)
					for (int direction : {0, 1})
					{
						Vec2i face = cellToFace(cell, axis, direction);

						if (myCutCellWeights(face, axis) > 0)
						{
							isFluidCell = true;
							break;
						}
					}

				if (isFluidCell)
				{
					if (mySurface(cell) <= 0)
						baseDomainCellLabels(cell) = MGCellLabels::INTERIOR_CELL;
					else
					{
						baseDomainCellLabels(cell) = MGCellLabels::DIRICHLET_CELL;
						hasDirichletCell = true;
					}
				}
			}
		});

	// Build RHS
	UniformGrid<StoreReal> rhsGrid(mySurface.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, mySurface.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mySurface.unflatten(cellIndex);

				if (baseDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
				{
					// Build RHS divergence
					SolveReal divergence = 0;

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal weight = myCutCellWeights(face, axis);

							SolveReal sign = (direction == 0) ? 1 : -1;

							if (weight > 0)
								divergence += sign * velocity(face, axis) * weight;
							if (weight < 1)
								divergence += sign * (1. - weight) * mySolidVelocity(face, axis);
						}

					rhsGrid(cell) = divergence;
				}
			}
		});

	// Set a single interior cell to dirichlet and remove the average divergence
	if (!hasDirichletCell)
	{
		tbb::enumerable_thread_specific<SolveReal> parallelAccumulatedDivergence(SolveReal(0)), parallelCellCount(SolveReal(0));

		tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto& localAccumulatedDivergence = parallelAccumulatedDivergence.local();
				auto& localCellCount = parallelCellCount.local();

				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = baseDomainCellLabels.unflatten(cellIndex);

					if (baseDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
					{
						localAccumulatedDivergence += rhsGrid(cell);
						++localCellCount;
					}
				}
			});

		SolveReal averageDivergence = 0;

		parallelAccumulatedDivergence.combine_each([&](const SolveReal localAccumulatedDivergence)
			{
				averageDivergence += localAccumulatedDivergence;
			});

		SolveReal cellCount = 0;

		parallelCellCount.combine_each([&](const SolveReal localCellCount)
			{
				cellCount += localCellCount;
			});

		averageDivergence /= cellCount;

		tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = baseDomainCellLabels.unflatten(cellIndex);

					if (baseDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
						rhsGrid(cell) -= averageDivergence;
				}
			});

		// Remove single interior cell
		bool hasRemovedInteriorCell = false;
		forEachVoxelRange(Vec2i(0), baseDomainCellLabels.size(), [&](const Vec2i& cell)
		{
			if (hasRemovedInteriorCell) return;
			
			if (baseDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
			{
				bool hasNonInterior = false;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						if (baseDomainCellLabels(adjacentCell) != MGCellLabels::INTERIOR_CELL)
							hasNonInterior = true;
					}

				if (!hasNonInterior)
				{
					hasRemovedInteriorCell = true;
					baseDomainCellLabels(cell) = MGCellLabels::DIRICHLET_CELL;
				}
			}
		});

		assert(hasRemovedInteriorCell);
	}

	// Build poisson face weights
	VectorGrid<StoreReal> poissonFaceWeights(mySurface.xform(), mySurface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, poissonFaceWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = poissonFaceWeights.grid(axis).unflatten(faceIndex);

					SolveReal weight = myCutCellWeights(face, axis);

					if (weight > 0)
					{
						Vec2i backwardCell = faceToCell(face, axis, 0);
						Vec2i forwardCell = faceToCell(face, axis, 1);

						auto backwardLabel = baseDomainCellLabels(backwardCell);
						auto forwardLabel = baseDomainCellLabels(forwardCell);

						if ((backwardLabel == MGCellLabels::INTERIOR_CELL && forwardLabel == MGCellLabels::DIRICHLET_CELL) ||
							(backwardLabel == MGCellLabels::DIRICHLET_CELL && forwardLabel == MGCellLabels::INTERIOR_CELL))
						{
							SolveReal theta = myGhostFluidWeights(face, axis);
							theta = clamp(theta, SolveReal(.01), SolveReal(1));

							weight /= theta;
						}

						poissonFaceWeights(face, axis) = weight;
					}
				}
			});
	}

	// Build extended MG domain
	UniformGrid<CellLabels> mgDomainCellLabels;
	std::pair<Vec2i, int> mgSettings = GeometricMultigridOperators::buildExpandedDomainLabels(mgDomainCellLabels, baseDomainCellLabels);

	Vec2i expandedOffset = mgSettings.first;
	int mgLevels = mgSettings.second;

	VectorGrid<StoreReal> mgBoundaryWeights(poissonFaceWeights.xform(), mgDomainCellLabels.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
		GeometricMultigridOperators::buildExpandedBoundaryWeights(mgBoundaryWeights, poissonFaceWeights, mgDomainCellLabels, expandedOffset, axis);

	GeometricMultigridOperators::setBoundaryDomainLabels(mgDomainCellLabels, mgBoundaryWeights);

	// Build expanded rhs grid and solution grid
	UniformGrid<StoreReal> mgRHSGrid(mgDomainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mgDomainCellLabels.unflatten(cellIndex);

				if (mgDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == MGCellLabels::BOUNDARY_CELL)
				{
					assert(baseDomainCellLabels(cell - expandedOffset) == MGCellLabels::INTERIOR_CELL);
					mgRHSGrid(cell) = rhsGrid(cell - expandedOffset);
				}
			}
		});

	UniformGrid<StoreReal> mgSolutionGrid(mgDomainCellLabels.size(), 0);

	// Add initial guess
	if (myUseInitialGuessPressure)
	{
		assert(myInitialGuessPressure != nullptr);
		tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = mgDomainCellLabels.unflatten(cellIndex);

					if (mgDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == MGCellLabels::BOUNDARY_CELL)
					{
						assert(baseDomainCellLabels(cell - expandedOffset) == MGCellLabels::INTERIOR_CELL);
						mgSolutionGrid(cell) = (*myInitialGuessPressure)(cell - expandedOffset);
					}
				}
			});
	}

	UniformGrid<StoreReal> diagonalPrecondGrid(mgDomainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mgDomainCellLabels.unflatten(cellIndex);

				if (mgDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
				{
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);
							assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < mgDomainCellLabels.size()[axis]);

							assert(mgDomainCellLabels(adjacentCell) == MGCellLabels::INTERIOR_CELL ||
								mgDomainCellLabels(adjacentCell) == MGCellLabels::BOUNDARY_CELL);

							Vec2i face = cellToFace(cell, axis, direction);
							assert(mgBoundaryWeights(face, axis) == 1);
						}

					diagonalPrecondGrid(cell) = 1. / 4.;
				}
				else if (mgDomainCellLabels(cell) == MGCellLabels::BOUNDARY_CELL)
				{
					SolveReal diagonal = 0;
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);
							Vec2i face = cellToFace(cell, axis, direction);

							if (mgDomainCellLabels(adjacentCell) == MGCellLabels::INTERIOR_CELL)
							{
								assert(mgBoundaryWeights(face, axis) == 1);
								++diagonal;
							}
							else if (mgDomainCellLabels(adjacentCell) == MGCellLabels::BOUNDARY_CELL ||
								mgDomainCellLabels(adjacentCell) == MGCellLabels::DIRICHLET_CELL)
							{
								diagonal += mgBoundaryWeights(face, axis);
							}
							else
							{
								assert(mgDomainCellLabels(adjacentCell) == MGCellLabels::EXTERIOR_CELL);
								assert(mgBoundaryWeights(face, axis) == 0);
							}
						}
					diagonalPrecondGrid(cell) = 1. / diagonal;
				}
			}
		});

	auto diagonalPreconditioner = [&](UniformGrid<StoreReal>& solutionGrid, const UniformGrid<StoreReal>& rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = mgDomainCellLabels.unflatten(cellIndex);

					if (mgDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL ||
						mgDomainCellLabels(cell) == MGCellLabels::BOUNDARY_CELL)
					{
						solutionGrid(cell) = rhsGrid(cell) * diagonalPrecondGrid(cell);
					}
				}
			});
	};

	auto matrixVectorMultiply = [&](UniformGrid<StoreReal>& destinationGrid, const UniformGrid<StoreReal>& sourceGrid)
	{
		assert(destinationGrid.size() == sourceGrid.size() && sourceGrid.size() == mgDomainCellLabels.size());

		// Matrix-vector multiplication
		GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destinationGrid, sourceGrid, mgDomainCellLabels, 1., &mgBoundaryWeights);
	};

	// Pre-build multigrid preconditioner
	GeometricMultigridPoissonSolver mgPreconditioner(mgDomainCellLabels, mgBoundaryWeights, mgLevels, 1.);

	auto multiGridPreconditioner = [&](UniformGrid<StoreReal>& destinationGrid, const UniformGrid<StoreReal>& sourceGrid)
	{
		assert(destinationGrid.size() == sourceGrid.size() && sourceGrid.size() == mgDomainCellLabels.size());

		mgPreconditioner.applyMGVCycle(destinationGrid, sourceGrid);
	};

	auto dotProduct = [&](const UniformGrid<StoreReal>& grid0, const UniformGrid<StoreReal>& grid1)
	{
		assert(grid0.size() == grid1.size() && grid1.size() == mgDomainCellLabels.size());

		return GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, mgDomainCellLabels);
	};

	auto squaredL2Norm = [&](const UniformGrid<StoreReal>& grid)
	{
		assert(grid.size() == mgDomainCellLabels.size());

		return GeometricMultigridOperators::squaredl2Norm<SolveReal>(grid, mgDomainCellLabels);
	};

	auto addToVector = [&](UniformGrid<StoreReal>& destination, const UniformGrid<StoreReal>& scaledSource, const SolveReal scale)
	{
		assert(destination.size() == scaledSource.size() && scaledSource.size() == mgDomainCellLabels.size());

		GeometricMultigridOperators::addToVector<SolveReal>(destination, scaledSource, mgDomainCellLabels, scale);
	};

	auto addScaledVector = [&](UniformGrid<StoreReal>& destination,
		const UniformGrid<StoreReal>& unscaledSource,
		const UniformGrid<StoreReal>& scaledSource,
		const SolveReal scale)
	{
		assert(destination.size() == unscaledSource.size() &&
			unscaledSource.size() == scaledSource.size() &&
			scaledSource.size() == mgDomainCellLabels.size());

		GeometricMultigridOperators::addVectors<SolveReal>(destination, unscaledSource, scaledSource, mgDomainCellLabels, scale);
	};

	if (useMGPreconditioner)
	{
		solveGeometricConjugateGradient(mgSolutionGrid,
			mgRHSGrid,
			matrixVectorMultiply,
			multiGridPreconditioner,
			dotProduct,
			squaredL2Norm,
			addToVector,
			addScaledVector,
			1E-3 /*solver tolerance*/, 1000 /* max iterations */);
	}
	else
	{
		solveGeometricConjugateGradient(mgSolutionGrid,
			mgRHSGrid,
			matrixVectorMultiply,
			diagonalPreconditioner,
			dotProduct,
			squaredL2Norm,
			addToVector,
			addScaledVector,
			1E-3 /*solver tolerance*/, 1000 /* max iterations */);
	}

	{
		UniformGrid<StoreReal> residualGrid(mgDomainCellLabels.size(), 0);

		GeometricMultigridOperators::computePoissonResidual<SolveReal>(residualGrid, mgSolutionGrid, mgRHSGrid, mgDomainCellLabels, 1.);

		std::cout << "L-infinity error of solution: " << GeometricMultigridOperators::lInfinityNorm<SolveReal>(residualGrid, mgDomainCellLabels) << std::endl;
	}

	// Apply solution back to pressure grid
	myPressure.resize(myPressure.size(), 0);
	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = mgDomainCellLabels.unflatten(cellIndex);

				if (mgDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == MGCellLabels::BOUNDARY_CELL)
				{
					assert(baseDomainCellLabels(cell - expandedOffset) == MGCellLabels::INTERIOR_CELL);
					myPressure(cell - expandedOffset) = mgSolutionGrid(cell);
				}
			}
		});

	// Set valid faces
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = myValidFaces.grid(axis).unflatten(faceIndex);

					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					if (backwardCell[axis] >= 0 && forwardCell[axis] < mySurface.size()[axis])
					{
						if ((baseDomainCellLabels(backwardCell) == MGCellLabels::INTERIOR_CELL || baseDomainCellLabels(forwardCell) == MGCellLabels::INTERIOR_CELL) && myCutCellWeights(face, axis) > 0)
							myValidFaces(face, axis) = VisitedCellLabels::FINISHED_CELL;
						else assert(myValidFaces(face, axis) == VisitedCellLabels::UNVISITED_CELL);
					}
					else assert(myValidFaces(face, axis) == VisitedCellLabels::UNVISITED_CELL);
				}
			});
	}

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, velocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = myValidFaces.grid(axis).unflatten(faceIndex);

					SolveReal localVelocity = 0;
					if (myValidFaces(face, axis) == VisitedCellLabels::FINISHED_CELL)
					{
						Vec2i backwardCell = faceToCell(face, axis, 0);
						Vec2i forwardCell = faceToCell(face, axis, 1);

						assert(backwardCell[axis] >= 0 && forwardCell[axis] < mySurface.size()[axis]);
						assert(baseDomainCellLabels(backwardCell) == MGCellLabels::INTERIOR_CELL ||
							baseDomainCellLabels(forwardCell) == MGCellLabels::INTERIOR_CELL);

						SolveReal gradient = myPressure(forwardCell);
						gradient -= myPressure(backwardCell);

						if (baseDomainCellLabels(backwardCell) == MGCellLabels::DIRICHLET_CELL ||
							baseDomainCellLabels(forwardCell) == MGCellLabels::DIRICHLET_CELL)
						{
							SolveReal theta = myGhostFluidWeights(face, axis);
							theta = clamp(theta, SolveReal(.01), SolveReal(1));

							gradient /= theta;
						}

						localVelocity = velocity(face, axis) - gradient;
					}

					velocity(face, axis) = localVelocity;
				}
			});
	}

	// Verify divergence free
	UniformGrid<StoreReal> divergenceGrid(mySurface.size(), 0);
	tbb::parallel_for(tbb::blocked_range<int>(0, divergenceGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = baseDomainCellLabels.unflatten(cellIndex);

				if (baseDomainCellLabels(cell) == MGCellLabels::INTERIOR_CELL)
				{
					// Build RHS divergence
					SolveReal divergence = 0;
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal weight = myCutCellWeights(face, axis);

							SolveReal sign = (direction == 0) ? 1 : -1;

							if (weight > 0)
								divergence += sign * velocity(face, axis) * weight;
							if (weight < 1.)
								divergence += sign * mySolidVelocity(face, axis) * (1. - weight);
						}

					divergenceGrid(cell) = divergence;
				}
			}
		});

	UniformGrid<StoreReal> dummyGrid(mySurface.size(), 1);
	std::cout << "Accumulated divergence: " << GeometricMultigridOperators::dotProduct<SolveReal>(divergenceGrid, dummyGrid, baseDomainCellLabels) << std::endl;
	std::cout << "Max norm divergence: " << GeometricMultigridOperators::lInfinityNorm<SolveReal>(divergenceGrid, baseDomainCellLabels) << std::endl;
}

}