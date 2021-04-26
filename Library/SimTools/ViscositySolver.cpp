#include "ViscositySolver.h"

#include <iostream>

#include <Eigen/Sparse>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "ComputeWeights.h"
#include "LevelSet.h"

namespace FluidSim2D
{

using SolveReal = double;
using Vector = Eigen::VectorXd;

void ViscositySolver(double dt,
						const LevelSet& surface,
						VectorGrid<double>& velocity,
						const LevelSet& solidSurface,
						const VectorGrid<double>& solidVelocity,
						const ScalarGrid<double>& viscosity)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled surface and collision
	assert(surface.isGridMatched(solidSurface));
	assert(surface.isGridMatched(viscosity));
	assert(velocity.isGridMatched(solidVelocity));

	for (int axis : {0, 1})
	{
		Vec2i faceSize = velocity.size(axis);
		--faceSize[axis];

		assert(faceSize == surface.size());
	}

	int samples = 3;

	ScalarGrid<double> centerAreas = computeSupersampledAreas(surface, ScalarGridSettings::SampleType::CENTER, samples);
	ScalarGrid<double> nodeAreas = computeSupersampledAreas(surface, ScalarGridSettings::SampleType::NODE, samples);
	VectorGrid<double> faceAreas = computeSupersampledFaceAreas(surface, samples);

	SolveReal discreteScalar = dt / std::pow(surface.dx(), 2);

	// Pre-scale all the control volumes with coefficients to reduce
	// redundant operations when building the linear system.

	tbb::parallel_for(tbb::blocked_range<int>(0, centerAreas.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = centerAreas.unflatten(cellIndex);

			if (centerAreas(cell) > 0)
				centerAreas(cell) *= 2. * discreteScalar * viscosity(cell);
		}
	});


	tbb::parallel_for(tbb::blocked_range<int>(0, nodeAreas.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int nodeIndex = range.begin(); nodeIndex != range.end(); ++nodeIndex)
		{
			Vec2i node = nodeAreas.unflatten(nodeIndex);

			if (nodeAreas(node) > 0)
				nodeAreas(node) *= discreteScalar * viscosity.biLerp(nodeAreas.indexToWorld(node.cast<double>()));
		}
	});

	enum class MaterialLabels { SOLID_FACE, LIQUID_FACE, AIR_FACE };

	VectorGrid<MaterialLabels> materialFaceLabels(surface.xform(), surface.size(), MaterialLabels::AIR_FACE, VectorGridSettings::SampleType::STAGGERED);

	// Set material labels for each grid face. We assume faces along the simulation boundary
	// are solid.

	for (int faceAxis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);

				if (face[faceAxis] == 0 || face[faceAxis] == materialFaceLabels.size(faceAxis)[faceAxis] - 1)
					continue;

				bool isFaceInSolve = false;

				for (int direction : {0, 1})
				{
					Vec2i cell = faceToCell(face, faceAxis, direction);
					if (centerAreas(cell) > 0) isFaceInSolve = true;
				}

				if (!isFaceInSolve)
				{
					for (int direction : {0, 1})
					{
						Vec2i node = faceToNode(face, faceAxis, direction);
						if (nodeAreas(node) > 0) isFaceInSolve = true;
					}
				}

				if (isFaceInSolve)
				{
					if (solidSurface.biLerp(materialFaceLabels.indexToWorld(face.cast<double>(), faceAxis)) <= 0)
						materialFaceLabels(face, faceAxis) = MaterialLabels::SOLID_FACE;
					else
						materialFaceLabels(face, faceAxis) = MaterialLabels::LIQUID_FACE;
				}
			}
		});
	}

	int liquidDOFCount = 0;

	constexpr int UNLABELLED_CELL = -1;

	VectorGrid<int> liquidFaceIndices(surface.xform(), surface.size(), UNLABELLED_CELL, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), materialFaceLabels.size(axis), [&](const Vec2i& face)
		{
			if (materialFaceLabels(face, axis) == MaterialLabels::LIQUID_FACE)
				liquidFaceIndices(face, axis) = liquidDOFCount++;
		});
	}

	std::vector<Eigen::Triplet<SolveReal>> sparseElements;
	Vector initialGuessVector = Vector::Zero(liquidDOFCount);
	Vector rhsVector = Vector::Zero(liquidDOFCount);

	{
		tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<SolveReal>>> parallelSparseElements;

		for (int faceAxis : {0, 1})
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount(), materialFaceLabels.grid(faceAxis).voxelCount() + 1/*tbbLightGrainSize*/), [&](const tbb::blocked_range<int>& range)
			{
				auto& localSparseElements = parallelSparseElements.local();

				for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
				{
					Vec2i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);
					int liquidFaceIndex = liquidFaceIndices(face, faceAxis);

					if (liquidFaceIndex >= 0)
					{
						assert(materialFaceLabels(face, faceAxis) == MaterialLabels::LIQUID_FACE);

						// Use old velocity as an initial guess since we're solving for a new
						// velocity field with viscous forces applied to the old velocity field.
						initialGuessVector(liquidFaceIndex) = velocity(face, faceAxis);

						// Build RHS with volume weights
						SolveReal localFaceArea = faceAreas(face, faceAxis);

						rhsVector(liquidFaceIndex) = localFaceArea * velocity(face, faceAxis);

						// Add volume weight to diagonal
						SolveReal diagonal = localFaceArea;

						// Build cell centered stress terms
						for (int divergenceDirection : {0, 1})
						{
							Vec2i cell = faceToCell(face, faceAxis, divergenceDirection);

							assert(cell[faceAxis] >= 0 && cell[faceAxis] < centerAreas.size()[faceAxis]);

							if (centerAreas(cell) > 0)
							{
								SolveReal divergenceSign = (divergenceDirection == 0) ? -1 : 1;

								for (int gradientDirection : {0, 1})
								{
									Vec2i adjacentFace = cellToFace(cell, faceAxis, gradientDirection);

									SolveReal gradientSign = (gradientDirection == 0) ? -1. : 1.;

									SolveReal coefficient = divergenceSign * gradientSign * centerAreas(cell);

									int adjacentFaceIndex = liquidFaceIndices(adjacentFace, faceAxis);
									if (adjacentFaceIndex >= 0)
									{
										if (adjacentFaceIndex == liquidFaceIndex)
											diagonal -= coefficient;
										else
											localSparseElements.emplace_back(liquidFaceIndex, adjacentFaceIndex, -coefficient);
									}
									else if (materialFaceLabels(adjacentFace, faceAxis) == MaterialLabels::SOLID_FACE)
										rhsVector(liquidFaceIndex) += coefficient * solidVelocity(adjacentFace, faceAxis);
									else
										assert(materialFaceLabels(adjacentFace, faceAxis) == MaterialLabels::AIR_FACE);
								}
							}
						}

						// Build node stresses.
						for (int divergenceDirection : {0, 1})
						{
							Vec2i node = faceToNode(face, faceAxis, divergenceDirection);

							if (nodeAreas(node) > 0)
							{
								SolveReal divergenceSign = (divergenceDirection == 0) ? -1. : 1.;

								for (int gradientAxis : {0, 1})
								{
									int gradientFaceAxis = (gradientAxis + 1) % 2;
									for (int gradientDirection : {0, 1})
									{
										SolveReal gradientSign = gradientDirection == 0 ? -1. : 1.;

										Vec2i localGradientFace = nodeToFace(node, gradientAxis, gradientDirection);

										int gradientFaceIndex = liquidFaceIndices(localGradientFace, gradientFaceAxis);

										SolveReal coefficient = divergenceSign * gradientSign * nodeAreas(node);

										if (gradientFaceIndex >= 0)
										{
											if (gradientFaceIndex == liquidFaceIndex)
												diagonal -= coefficient;
											else
												localSparseElements.emplace_back(liquidFaceIndex, gradientFaceIndex, -coefficient);
										}
										else if (materialFaceLabels(localGradientFace, gradientFaceAxis) == MaterialLabels::SOLID_FACE)
											rhsVector(liquidFaceIndex) += coefficient * solidVelocity(localGradientFace, gradientFaceAxis);
										else assert(materialFaceLabels(localGradientFace, gradientFaceAxis) == MaterialLabels::AIR_FACE);
									}
								}
							}
						}

						assert(diagonal > 0);
						localSparseElements.emplace_back(liquidFaceIndex, liquidFaceIndex, diagonal);
					}
					else assert(materialFaceLabels(face, faceAxis) != MaterialLabels::LIQUID_FACE);
				}
			});
		}

		mergeLocalThreadVectors(sparseElements, parallelSparseElements);
	}

	Eigen::SparseMatrix<SolveReal> sparseMatrix(liquidDOFCount, liquidDOFCount);
	sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());

	Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal>, Eigen::Upper | Eigen::Lower> solver;
	solver.compute(sparseMatrix);
	solver.setTolerance(1E-3);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return;
	}

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

	for (int faceAxis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);
				int liquidFaceIndex = liquidFaceIndices(face, faceAxis);
				if (liquidFaceIndex >= 0)
				{
					assert(materialFaceLabels(face, faceAxis) == MaterialLabels::LIQUID_FACE);
					velocity(face, faceAxis) = solutionVector(liquidFaceIndex);
				}
			}
		});
	}
}

}