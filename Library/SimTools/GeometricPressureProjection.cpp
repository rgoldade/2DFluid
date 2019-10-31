#include "GeometricPressureProjection.h"

#include "tbb/tbb.h"

#include "GeometricConjugateGradientSolver.h"
#include "GeometricMultigridPoissonSolver.h"

void GeometricPressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSuperSampledValues(renderer, .25, 1, 2);
}

void GeometricPressureProjection::project(VectorGrid<Real>& velocity,
											const bool useMGPreconditioner)
{
	using GeometricMultigridOperators::CellLabels;

	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled liquid surface.
	assert(velocity.sampleType() == VectorGridSettings::SampleType::STAGGERED &&
			velocity.size(0)[0] - 1 == mySurface.size()[0] &&
			velocity.size(0)[1] == mySurface.size()[1] &&
			velocity.size(1)[0] == mySurface.size()[0] &&
			velocity.size(1)[1] - 1 == mySurface.size()[1]);

	assert(velocity.isGridMatched(myCutCellWeights));

	bool hasDirichlet = false;

	UniformGrid<CellLabels> baseDomainCellLabels(mySurface.size(), CellLabels::EXTERIOR_CELL);
	// Build domain labels
	tbb::parallel_for(tbb::blocked_range<int>(0, mySurface.size()[0] * mySurface.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = mySurface.unflatten(flatIndex);

			bool isInsideLiquid = mySurface(cell) <= 0;
			bool isInsideSolid = true;
			for (int axis : { 0, 1 })
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					if (myCutCellWeights(face, axis) > 0)
						isInsideSolid = false;
				}

			if (isInsideSolid)
				baseDomainCellLabels(cell) = CellLabels::EXTERIOR_CELL;
			else if (isInsideLiquid)
				baseDomainCellLabels(cell) = CellLabels::INTERIOR_CELL;
			else
			{
				baseDomainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
				hasDirichlet = true;
			}
		}
	});

	// Build RHS
	UniformGrid<StoreReal> rhsGrid(mySurface.size(), 0);
	tbb::parallel_for(tbb::blocked_range<int>(0, mySurface.size()[0] * mySurface.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = baseDomainCellLabels.unflatten(flatIndex);

			if (baseDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
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
							divergence += sign * (1. - weight) * mySolidVelocity(face, axis);
					}

				rhsGrid(cell) = divergence;
			}
		}
	});

	// Set a single interior cell to dirichlet and remove the average divergence
	if (!hasDirichlet)
	{
		using ParallelReal = tbb::enumerable_thread_specific<SolveReal>;

		ParallelReal parallelAccumulatedDivergence(0);
		ParallelReal parallelCellCount(0);

		tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.size()[0] * baseDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			auto &localAccumulatedDivergence = parallelAccumulatedDivergence.local();
			auto &localCellCount = parallelCellCount.local();

			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = baseDomainCellLabels.unflatten(flatIndex);

				if (baseDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
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

		tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.size()[0] * baseDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = baseDomainCellLabels.unflatten(flatIndex);

				if (baseDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
					rhsGrid(cell) -= averageDivergence;
			}
		});

		// Remove single interior cell
		bool hasRemovedInteriorCell = false;
		forEachVoxelRange(Vec2i(0), baseDomainCellLabels.size(), [&](const Vec2i& cell)
		{
			if (hasRemovedInteriorCell) return;
			
			if (baseDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				bool hasNonInterior = false;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						if (baseDomainCellLabels(adjacentCell) != CellLabels::INTERIOR_CELL)
							hasNonInterior = true;
					}

				if (!hasNonInterior)
				{
					hasRemovedInteriorCell = true;
					baseDomainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
				}
			}
		});

		assert(hasRemovedInteriorCell);
	}

	// Build poisson face weights
	VectorGrid<StoreReal> poissonFaceWeights(mySurface.xform(), mySurface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, poissonFaceWeights.size(axis)[0] * poissonFaceWeights.size(axis)[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = poissonFaceWeights.grid(axis).unflatten(flatIndex);

				Real weight = myCutCellWeights(face, axis);

				if (weight > 0)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					auto backwardLabel = baseDomainCellLabels(backwardCell);
					auto forwardLabel = baseDomainCellLabels(forwardCell);

					if ((backwardLabel == CellLabels::INTERIOR_CELL && forwardLabel == CellLabels::DIRICHLET_CELL) ||
						(backwardLabel == CellLabels::DIRICHLET_CELL && forwardLabel == CellLabels::INTERIOR_CELL))
					{
						Real theta = myGhostFluidWeights(face, axis);
						theta = Util::clamp(theta, MINTHETA, Real(1.));

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

	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.size()[0] * mgDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = mgDomainCellLabels.unflatten(flatIndex);

			if (mgDomainCellLabels(cell) == CellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(baseDomainCellLabels(cell - expandedOffset) == CellLabels::INTERIOR_CELL);
				mgRHSGrid(cell) = rhsGrid(cell - expandedOffset);
			}
		}
	});

	UniformGrid<StoreReal> mgSolutionGrid(mgDomainCellLabels.size(), 0);
	
	// Add initial guess
	if (myUseInitialGuessPressure)
	{
		assert(myInitialGuessPressure != nullptr);
		tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.size()[0] * mgDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = mgDomainCellLabels.unflatten(flatIndex);

				if (mgDomainCellLabels(cell) == CellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					assert(baseDomainCellLabels(cell - expandedOffset) == CellLabels::INTERIOR_CELL);
					mgSolutionGrid(cell) = (*myInitialGuessPressure)(cell - expandedOffset);
				}
			}
		});
	}

	UniformGrid<StoreReal> diagonalPrecondGrid(mgDomainCellLabels.size(), 0);

	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.size()[0] * mgDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = mgDomainCellLabels.unflatten(flatIndex);

			if (mgDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < mgDomainCellLabels.size()[axis]);

						assert(mgDomainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
								mgDomainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

						Vec2i face = cellToFace(cell, axis, direction);
						assert(mgBoundaryWeights(face, axis) == 1);
					}

				diagonalPrecondGrid(cell) = 1. / 4.;
			}
			else if (mgDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				Real diagonal = 0;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);
						Vec2i face = cellToFace(cell, axis, direction);

						if (mgDomainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
						{
							assert(mgBoundaryWeights(face, axis) == 1);
							++diagonal;							
						}
						else if (mgDomainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL || 
									mgDomainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
						{
							diagonal += mgBoundaryWeights(face, axis);
						}
						else
						{
							assert(mgDomainCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL);
							assert(mgBoundaryWeights(face, axis) == 0);
						}
					}
				diagonalPrecondGrid(cell) = 1. / diagonal;
			}
		}
	});

	auto diagonalPreconditioner = [&](UniformGrid<StoreReal> &solutionGrid, const UniformGrid<StoreReal> &rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.size()[0] * mgDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i cell = mgDomainCellLabels.unflatten(flatIndex);
				
				if (mgDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
					mgDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				{
					solutionGrid(cell) = rhsGrid(cell) * diagonalPrecondGrid(cell);
				}
			}
		});
	};

	auto matrixVectorMultiply = [&](UniformGrid<StoreReal> &destinationGrid, const UniformGrid<StoreReal> &sourceGrid)
	{
		assert(destinationGrid.size() == sourceGrid.size() &&
				sourceGrid.size() == mgDomainCellLabels.size());
		// Matrix-vector multiplication
		GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destinationGrid, sourceGrid, mgDomainCellLabels, 1., &mgBoundaryWeights);
	};

	// Pre-build multigrid preconditioner
	GeometricMultigridPoissonSolver mgPreconditioner(mgDomainCellLabels, mgBoundaryWeights, mgLevels, 1.);

	auto multiGridPreconditioner = [&](UniformGrid<StoreReal> &destinationGrid, const UniformGrid<StoreReal> &sourceGrid)
	{
		assert(destinationGrid.size() == sourceGrid.size() &&
				sourceGrid.size() == mgDomainCellLabels.size());
		mgPreconditioner.applyMGVCycle(destinationGrid, sourceGrid);
	};

	auto dotProduct = [&](const UniformGrid<StoreReal> &grid0, const UniformGrid<StoreReal> &grid1)
	{
		assert(grid0.size() == grid1.size() &&
				grid1.size() == mgDomainCellLabels.size());
		return GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, mgDomainCellLabels);
	};

	auto squaredL2Norm = [&](const UniformGrid<StoreReal> &grid)
	{
		assert(grid.size() == mgDomainCellLabels.size());
		return GeometricMultigridOperators::squaredl2Norm<SolveReal>(grid, mgDomainCellLabels);
	};

	auto addToVector = [&](UniformGrid<StoreReal> &destination,
							const UniformGrid<StoreReal> &scaledSource,
							const SolveReal scale)
	{
		assert(destination.size() == scaledSource.size() &&
				scaledSource.size() == mgDomainCellLabels.size());
		GeometricMultigridOperators::addToVector<SolveReal>(destination, scaledSource, mgDomainCellLabels, scale);
	};

	auto addScaledVector = [&](UniformGrid<StoreReal> &destination,
								const UniformGrid<StoreReal> &unscaledSource,
								const UniformGrid<StoreReal> &scaledSource,
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
										1E-5 /*solver tolerance*/, 4000 /* max iterations */);
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
										1E-5 /*solver tolerance*/, 4000 /* max iterations */);
	}

	{
		UniformGrid<StoreReal> residualGrid(mgDomainCellLabels.size(), 0);

		GeometricMultigridOperators::computePoissonResidual<SolveReal>(residualGrid, mgSolutionGrid, mgRHSGrid, mgDomainCellLabels, 1.);

		std::cout << "L-infinity error of solution: " << GeometricMultigridOperators::lInfinityNorm<SolveReal>(residualGrid, mgDomainCellLabels) << std::endl;

	}

	// Apply solution back to pressure grid
	myPressure.reset(0);
	tbb::parallel_for(tbb::blocked_range<int>(0, mgDomainCellLabels.size()[0] * mgDomainCellLabels.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = mgDomainCellLabels.unflatten(flatIndex);

			if (mgDomainCellLabels(cell) == CellLabels::INTERIOR_CELL || mgDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
				assert(baseDomainCellLabels(cell - expandedOffset) == CellLabels::INTERIOR_CELL);
				myPressure(cell - expandedOffset) = mgSolutionGrid(cell);
			}
		}
	});

	// Set valid faces
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.size(axis)[0] * myValidFaces.size(axis)[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = myValidFaces.grid(axis).unflatten(flatIndex);
				
				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (backwardCell[axis] >= 0 && forwardCell[axis] < mySurface.size()[axis])
				{
					if ((baseDomainCellLabels(backwardCell) == CellLabels::INTERIOR_CELL || baseDomainCellLabels(forwardCell) == CellLabels::INTERIOR_CELL) && myCutCellWeights(face, axis) > 0)
						myValidFaces(face, axis) = MarkedCells::FINISHED;
					else assert(myValidFaces(face, axis) == MarkedCells::UNVISITED);
				}
				else assert(myValidFaces(face, axis) == MarkedCells::UNVISITED);
			}
		});
	}

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, velocity.size(axis)[0] * velocity.size(axis)[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec2i face = myValidFaces.grid(axis).unflatten(flatIndex);

				SolveReal localVelocity = 0;
				if (myValidFaces(face, axis) == MarkedCells::FINISHED)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					assert(backwardCell[axis] >= 0 && forwardCell[axis] < mySurface.size()[axis]);
					assert(baseDomainCellLabels(backwardCell) == CellLabels::INTERIOR_CELL ||
							baseDomainCellLabels(forwardCell) == CellLabels::INTERIOR_CELL);

					SolveReal gradient = myPressure(forwardCell);
					gradient -= myPressure(backwardCell);

					if (baseDomainCellLabels(backwardCell) == CellLabels::DIRICHLET_CELL ||
						baseDomainCellLabels(forwardCell) == CellLabels::DIRICHLET_CELL)
					{
						SolveReal theta = myGhostFluidWeights(face, axis);
						theta = Util::clamp(theta, SolveReal(MINTHETA), SolveReal(1.));

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
	tbb::parallel_for(tbb::blocked_range<int>(0, mySurface.size()[0] * mySurface.size()[1], tbbGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = baseDomainCellLabels.unflatten(flatIndex);

			if (baseDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
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