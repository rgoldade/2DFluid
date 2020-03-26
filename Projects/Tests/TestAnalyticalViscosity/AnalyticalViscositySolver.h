#ifndef TESTS_ANALYTICAL_VISCOSITY_H
#define TESTS_ANALYTICAL_VISCOSITY_H

#include <Eigen/Sparse>

#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// AnalyticalViscositySolver.h/cpp
// Ryan Goldade 2017
// 
// Solves an analytical viscosity problem
// used to test convergence of the PDE from
// Batty and Bridson 2008. There are only
// grid-aligned solid boundary conditions.
// This is a useful starting point for 
// building tests on non-regular meshes.
//
////////////////////////////////////

using namespace FluidSim2D::Utilities;
using namespace FluidSim2D::RenderTools;


class AnalyticalViscositySolver
{
	static constexpr int UNASSIGNED = -1;

	using SolveReal = double;
	using Vector = Eigen::VectorXd;

public:
	AnalyticalViscositySolver(const Transform& xform, const Vec2i& size)
		: myXform(xform)
	{
		myVelocityIndex = VectorGrid<int>(xform, size, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
	}

	// Returns the infinity-norm error of the numerical solution
	template<typename Initial, typename Solution, typename Viscosity>
	float solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity, const float dt);

	Vec2f cellIndexToWorld(const Vec2f& index) const { return myXform.indexToWorld(index + Vec2f(.5)); }
	Vec2f nodeIndexToWorld(const Vec2f& index) const { return myXform.indexToWorld(index); }
	
	void drawGrid(Renderer& renderer) const;
	void drawActiveVelocity(Renderer& renderer) const;

private:

	Transform myXform;
	unsigned setVelocityIndex();
	VectorGrid<int> myVelocityIndex;
};

template<typename Initial, typename Solution, typename Viscosity>
float AnalyticalViscositySolver::solve(const Initial& initialFunction,
										const Solution& solutionFunction,
										const Viscosity& viscosityFunction,
										const float dt)
{
	int velocityDOFCount = setVelocityIndex();

	// Build reduced system.
	// (Note we don't need control volumes since the cells are the same size and there is no free surface).
	// (I - dt * mu * D^T K D) u(n+1) = u(n)

	std::vector<Eigen::Triplet<SolveReal>> sparseMatrixElements;

	Vector rhsVector = Vector::Zero(velocityDOFCount);

	float dx = myVelocityIndex.dx();
	float baseCoeff = dt / sqr(dx);

	for (int axis : { 0,1 })
	{
		Vec2i size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			int velocityIndex = myVelocityIndex(face, axis);

			if (velocityIndex >= 0)
			{
				Vec2f facePosition = myVelocityIndex.indexToWorld(Vec2f(face), axis);

				rhsVector(velocityIndex) += initialFunction(facePosition, axis);

				sparseMatrixElements.emplace_back(velocityIndex, velocityIndex, 1.);

				// Build cell-centered stresses.
				for (int cellDirection : { 0,1 })
				{
					Vec2i cell = faceToCell(Vec2i(face), axis, cellDirection);

					Vec2f cellPosition = cellIndexToWorld(Vec2f(cell));
					float cellCoeff = 2. * viscosityFunction(cellPosition) * baseCoeff;

					float cellSign = (cellDirection == 0) ? -1. : 1.;

					for (int faceDirection : { 0,1 })
					{
						Vec2i adjacentFace = cellToFace(cell, axis, faceDirection);

						float faceSign = (faceDirection == 0) ? -1. : 1.;

						int faceRow = myVelocityIndex(adjacentFace, axis);
						if (faceRow >= 0)
							sparseMatrixElements.emplace_back(velocityIndex, faceRow, -cellSign * faceSign * cellCoeff);
						// No solid boundary to deal with since faces on the boundary
						// are not included.
					}
				}

				// Build node stresses.
				for (int nodeDirection : { 0, 1})
				{
					Vec2i node = faceToNode(face, axis, nodeDirection);

					float nodeSign = (nodeDirection == 0) ? -1. : 1.;

					Vec2f nodePosition = nodeIndexToWorld(Vec2f(node));
					float nodeCoeff = viscosityFunction(nodePosition) * baseCoeff;

					for (int gradientAxis : {0, 1})
						for (int faceDirection : {0, 1})
						{
							Vec2i adjacentFace = nodeToFace(node, gradientAxis, faceDirection);

							int faceAxis = (gradientAxis + 1) % 2;

							float faceSign = (faceDirection == 0) ? -1. : 1.;

							// Check for out of bounds
							if ((faceDirection == 0 && adjacentFace[gradientAxis] < 0) ||
								(faceDirection == 1 && adjacentFace[gradientAxis] >= size[gradientAxis]))
							{
								Vec2f facePosition = myVelocityIndex.indexToWorld(Vec2f(adjacentFace), faceAxis);
								rhsVector(velocityIndex) += nodeSign * faceSign * nodeCoeff * solutionFunction(facePosition, faceAxis);
							}
							// Check for on the bounds
							else if ((nodeDirection == 0 && adjacentFace[faceAxis] == 0) ||
										(nodeDirection == 1 &&
										adjacentFace[faceAxis] == myVelocityIndex.size(faceAxis)[faceAxis] - 1))
							{
								Vec2f facePosition = myVelocityIndex.indexToWorld(Vec2f(adjacentFace), faceAxis);
								rhsVector(velocityIndex) += nodeSign * faceSign * nodeCoeff * solutionFunction(facePosition, faceAxis);
							}
							else
							{
								int adjacentRow = myVelocityIndex(adjacentFace, faceAxis);
								assert(adjacentRow >= 0);

								sparseMatrixElements.emplace_back(velocityIndex, adjacentRow, -nodeSign * faceSign * nodeCoeff);
							}
						}
				}
			}
		});
	}

	Eigen::SparseMatrix<SolveReal> sparseMatrix(velocityDOFCount, velocityDOFCount);
	sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<SolveReal>> solver;
	solver.compute(sparseMatrix);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return -1;
	}

	Vector solutionVector = solver.solve(rhsVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed" << std::endl;
		return -1;
	}

	float error = 0;

	for (int axis : {0, 1})
	{
		Vec2i size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			int velocityIndex = myVelocityIndex(face, axis);

			if (velocityIndex >= 0)
			{
				Vec2f facePosition = myVelocityIndex.indexToWorld(Vec2f(face), axis);
				float localError = fabs(solutionVector(velocityIndex) - solutionFunction(facePosition, axis));

				if (error < localError) error = localError;
			}
		});
	}

	return error;
}

#endif