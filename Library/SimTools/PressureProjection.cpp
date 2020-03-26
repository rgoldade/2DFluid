#include "PressureProjection.h"

#include <iostream>

#include <Eigen/Sparse>

#include "tbb/tbb.h"

namespace FluidSim2D::SimTools
{

PressureProjection::PressureProjection(const LevelSet& surface,
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

void PressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .25, 1, 2);
}

void PressureProjection::project(VectorGrid<float>& velocity)
{
	using SolveReal = double;
	using Vector = Eigen::VectorXd;

	assert(velocity.isGridMatched(mySolidVelocity));

	enum class MaterialLabels { SOLID_CELL, AIR_CELL, LIQUID_CELL };

	UniformGrid<MaterialLabels> materialCellLabels(mySurface.size(), MaterialLabels::SOLID_CELL);

	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = materialCellLabels.unflatten(cellIndex);

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
						materialCellLabels(cell) = MaterialLabels::LIQUID_CELL;
					else
						materialCellLabels(cell) = MaterialLabels::AIR_CELL;
				}
			}
		});

	constexpr int UNSOLVED_CELL = -1;

	UniformGrid<int> liquidCellIndices(mySurface.size(), UNSOLVED_CELL);

	int liquidCellCount = 0;

	forEachVoxelRange(Vec2i(0), liquidCellIndices.size(), [&](const Vec2i& cell)
		{
			if (materialCellLabels(cell) == MaterialLabels::LIQUID_CELL)
				liquidCellIndices(cell) = liquidCellCount++;
		});

	Vector rhsVector = Vector::Zero(liquidCellCount);
	Vector initialGuessVector = Vector::Zero(liquidCellCount);

	Eigen::SparseMatrix<SolveReal> sparseMatrix(liquidCellCount, liquidCellCount);

	{
		tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<SolveReal>>> parallelSparseMatrixElements;

		tbb::parallel_for(tbb::blocked_range<int>(0, liquidCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			auto& localSparseMatrixElements = parallelSparseMatrixElements.local();

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = liquidCellIndices.unflatten(cellIndex);

				int liquidIndex = liquidCellIndices(cell);

				if (liquidIndex >= 0)
				{
					assert(materialCellLabels(cell) == MaterialLabels::LIQUID_CELL);

					// Compute divergence to add to RHS
					SolveReal divergence = 0;

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal weight = myCutCellWeights(face, axis);

							SolveReal sign = (direction == 0) ? 1 : -1;

							// Add divergence from faces
							if (weight > 0)
								divergence += sign * weight * velocity(face, axis);
							if (weight < 1)
								divergence += sign * (1. - weight) * mySolidVelocity(face, axis);
						}

					rhsVector(liquidIndex) = divergence;

					SolveReal diagonal = 0;

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
							if (adjacentCell[axis] < 0 || adjacentCell[axis] >= mySurface.size()[axis])
								continue;

							Vec2i face = cellToFace(cell, axis, direction);

							SolveReal weight = myCutCellWeights(face, axis);

							if (weight > 0)
							{
								int adjacentLiquidIndex = liquidCellIndices(adjacentCell);
								if (adjacentLiquidIndex >= 0)
								{
									assert(materialCellLabels(adjacentCell) == MaterialLabels::LIQUID_CELL);

									localSparseMatrixElements.emplace_back(liquidIndex, adjacentLiquidIndex, -weight);
									diagonal += weight;
								}
								else
								{
									assert(materialCellLabels(adjacentCell) == MaterialLabels::AIR_CELL);

									SolveReal theta = myGhostFluidWeights(face, axis);

									theta = clamp(theta, SolveReal(.01), SolveReal(1));
									diagonal += weight / theta;
								}
							}
							else assert(materialCellLabels(adjacentCell) == MaterialLabels::SOLID_CELL);
						}

					assert(diagonal > 0);
					localSparseMatrixElements.emplace_back(liquidIndex, liquidIndex, diagonal);

					if (myUseInitialGuessPressure)
					{
						assert(myInitialGuessPressure != nullptr);
						initialGuessVector(liquidIndex) = (*myInitialGuessPressure)(cell);
					}
				}
				else assert(materialCellLabels(cell) != MaterialLabels::LIQUID_CELL);
			}
		});

		std::vector<Eigen::Triplet<SolveReal>> sparseMatrixElements;
		mergeLocalThreadVectors(sparseMatrixElements, parallelSparseMatrixElements);
		sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal>, Eigen::Upper | Eigen::Lower> solver;
	solver.compute(sparseMatrix);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return;
	}

	solver.setTolerance(1E-5);

	Vector solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to converge" << std::endl;
		return;
	}
	else
	{
		std::cout << "    Solver iterations:     " << solver.iterations() << std::endl;
		std::cout << "    Solver error: " << solver.error() << std::endl;
	}

	// Copy resulting vector to pressure grid
	tbb::parallel_for(tbb::blocked_range<int>(0, liquidCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = liquidCellIndices.unflatten(cellIndex);

			int liquidIndex = liquidCellIndices(cell);

			if (liquidIndex >= 0)
			{
				assert(materialCellLabels(cell) == MaterialLabels::LIQUID_CELL);
				myPressure(cell) = solutionVector(liquidIndex);
			}
			else
			{
				myPressure(cell) = 0;
				assert(materialCellLabels(cell) != MaterialLabels::LIQUID_CELL);
			}
		}
	});

	// Build valid faces
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = myValidFaces.grid(axis).unflatten(faceIndex);

				bool isValidFace = false;
				if (myCutCellWeights(face, axis) > 0)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					if (backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis])
						continue;

					if (liquidCellIndices(backwardCell) >= 0 || liquidCellIndices(forwardCell) >= 0)
					{
						assert(materialCellLabels(backwardCell) == MaterialLabels::LIQUID_CELL ||
							materialCellLabels(forwardCell) == MaterialLabels::LIQUID_CELL);

						isValidFace = true;
					}
				}

				if (isValidFace)
					myValidFaces(face, axis) = VisitedCellLabels::FINISHED_CELL;
				else
					assert(myValidFaces(face, axis) == VisitedCellLabels::UNVISITED_CELL);
			}
		});
	}

	// Apply pressure update
	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = myValidFaces.grid(axis).unflatten(faceIndex);

				SolveReal tempVelocity = 0;

				if (myValidFaces(face, axis) == VisitedCellLabels::FINISHED_CELL)
				{
					Vec2i backwardCell = faceToCell(face, axis, 0);
					Vec2i forwardCell = faceToCell(face, axis, 1);

					assert(myCutCellWeights(face, axis) > 0);
					assert(backwardCell[axis] >= 0 && forwardCell[axis] <= mySurface.size()[axis]);
					assert(materialCellLabels(backwardCell) == MaterialLabels::LIQUID_CELL || materialCellLabels(forwardCell) == MaterialLabels::LIQUID_CELL);

					SolveReal gradient = myPressure(forwardCell) - myPressure(backwardCell);

					if (materialCellLabels(backwardCell) == MaterialLabels::AIR_CELL ||
						materialCellLabels(forwardCell) == MaterialLabels::AIR_CELL)
					{
						SolveReal theta = myGhostFluidWeights(face, axis);
						theta = clamp(theta, SolveReal(.01), SolveReal(1));

						gradient /= theta;
					}

					tempVelocity = velocity(face, axis) - gradient;
				}

				velocity(face, axis) = tempVelocity;
			}
		});
	}

	//
	// Debug test to verify divergence free constraints are met
	//

	{
		tbb::enumerable_thread_specific<SolveReal> parallelMaxDivergence(SolveReal(0));
		tbb::enumerable_thread_specific<SolveReal> parallelAccumulatedDivergence(SolveReal(0));
		tbb::enumerable_thread_specific<SolveReal> parallelCellCount(SolveReal(0));

		tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto& localMaxDivergence = parallelMaxDivergence.local();
				auto& localAccumulatedDivergence = parallelAccumulatedDivergence.local();
				auto& localCellCount = parallelCellCount.local();

				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = materialCellLabels.unflatten(cellIndex);

					if (materialCellLabels(cell) == MaterialLabels::LIQUID_CELL)
					{
						assert(liquidCellIndices(cell) >= 0);

						SolveReal divergence = 0;
						for (int axis : {0, 1})
							for (int direction : {0, 1})
							{
								Vec2i face = cellToFace(cell, axis, direction);
								SolveReal sign = (direction == 0) ? -1 : 1;

								SolveReal weight = myCutCellWeights(face, axis);
								
								if (weight > 0)
									divergence += sign * weight * velocity(face, axis);
							}

						localAccumulatedDivergence += std::fabs(divergence);
						localMaxDivergence = std::max(localMaxDivergence, std::fabs(divergence));
						++localCellCount;
					}
					else assert(liquidCellIndices(cell) == UNSOLVED_CELL);
				}
			});

		SolveReal accumulatedDivergence = 0;
		SolveReal maxDivergence;
		SolveReal cellCount = 0;

		parallelMaxDivergence.combine_each([&](SolveReal localMaxDivergence)
			{
				maxDivergence = std::max(localMaxDivergence, maxDivergence);
			});

		parallelAccumulatedDivergence.combine_each([&](SolveReal localAccumulatedDivergence)
			{
				accumulatedDivergence += localAccumulatedDivergence;
			});

		parallelCellCount.combine_each([&](SolveReal localCellCount)
			{
				cellCount += localCellCount;
			});

		assert(cellCount == liquidCellCount);

		std::cout << "Accumulated divergence: " << accumulatedDivergence << std::endl;
		std::cout << "Average divergence: " << accumulatedDivergence / SolveReal(cellCount) << std::endl;
		std::cout << "Max divergence: " << maxDivergence << std::endl;
	}

}

}