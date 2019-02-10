#pragma once

#include "Eigen/Sparse"

#include "Common.h"
#include "Renderer.h"
#include "Transform.h"
#include "ScalarGrid.h"
#include "Solver.h"
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

class AnalyticalViscositySolver
{
	static constexpr int UNASSIGNED = -1;

public:
	AnalyticalViscositySolver(const Transform& xform, const Vec2ui& size)
		: myXform(xform)
	{
		myVelocityIndex = VectorGrid<int>(xform, size, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
	}

	// Returns the infinity-norm error of the numerical solution
	template<typename Initial, typename Solution, typename Viscosity>
	Real solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity,
				const Real dt);

	Vec2R cellIndexToWorld(const Vec2R& index) const { return myXform.indexToWorld(index + Vec2R(.5)); }
	Vec2R nodeIndexToWorld(const Vec2R& index) const { return myXform.indexToWorld(index); }
	
	void drawGrid(Renderer& renderer) const;
	void drawActiveVelocity(Renderer& renderer) const;

private:

	Transform myXform;
	unsigned setVelocityIndex();
	VectorGrid<int> myVelocityIndex;
};

template<typename Initial, typename Solution, typename Viscosity>
Real AnalyticalViscositySolver::solve(const Initial& initialFunction,
										const Solution& solutionFunction,
										const Viscosity& viscosityFunction,
										const Real dt)
{
	unsigned velocityDOFCount = setVelocityIndex();

	// Build reduced system.
	// (Note we don't need control volumes since the cells are the same size and there is no free surface).
	// (I - dt * mu * D^T K D) u(n+1) = u(n)

	Solver<false> solver(velocityDOFCount, velocityDOFCount * 9);

	Real dx = myVelocityIndex.dx();
	Real baseCoeff = dt / Util::sqr(dx);

	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int row = myVelocityIndex(face, axis);

			if (row >= 0)
			{
				Vec2R facePosition = myVelocityIndex.indexToWorld(Vec2R(face), axis);

				solver.addRhs(row, initialFunction(facePosition, axis));
				solver.addElement(row, row, 1.);

				// Build cell-centered stresses.
				for (auto cellDirection : { 0,1 })
				{
					Vec2i cell = faceToCell(Vec2i(face), axis, cellDirection);

					Vec2R cellPosition = cellIndexToWorld(Vec2R(cell));
					Real cellCoeff = 2. * viscosityFunction(cellPosition) * baseCoeff;

					Real cellSign = (cellDirection == 0) ? -1. : 1.;

					for (auto faceDirection : { 0,1 })
					{
						Vec2ui adjacentFace = cellToFace(Vec2ui(cell), axis, faceDirection);

						Real faceSign = (faceDirection == 0) ? -1. : 1.;

						int faceRow = myVelocityIndex(adjacentFace, axis);
						if (faceRow >= 0)
							solver.addElement(row, faceRow, -cellSign * faceSign * cellCoeff);
						// No solid boundary to deal with since faces on the boundary
						// are not included.
					}
				}

				// Build node stresses.
				for (auto nodeDirection : { 0,1 })
				{
					Vec2ui node = faceToNode(face, axis, nodeDirection);

					Real nodeSign = (nodeDirection == 0) ? -1. : 1.;

					Vec2R nodePosition = nodeIndexToWorld(Vec2R(node));
					Real nodeCoeff = viscosityFunction(nodePosition) * baseCoeff;

					for (auto gradientAxis : { 0,1 })
						for (auto faceDirection : { 0,1 })
						{
							Vec2i adjacentFace = nodeToFace(Vec2i(node), gradientAxis, faceDirection);

							unsigned faceAxis = (gradientAxis + 1) % 2;

							Real faceSign = (faceDirection == 0) ? -1. : 1.;

							// Check for out of bounds
							if (faceDirection == 0 && adjacentFace[gradientAxis] < 0 ||
								faceDirection == 1 && adjacentFace[gradientAxis] >= size[gradientAxis])
							{
								Vec2R facePosition = myVelocityIndex.indexToWorld(Vec2R(adjacentFace), faceAxis);
								solver.addRhs(row, nodeSign * faceSign * nodeCoeff * solutionFunction(facePosition, faceAxis));
							}
							// Check for on the bounds
							else if (nodeDirection == 0 && adjacentFace[faceAxis] == 0 ||
								nodeDirection == 1 &&
									adjacentFace[faceAxis] == myVelocityIndex.size(faceAxis)[faceAxis] - 1)
							{
								Vec2R facePosition = myVelocityIndex.indexToWorld(Vec2R(adjacentFace), faceAxis);
								solver.addRhs(row, nodeSign * faceSign * nodeCoeff * solutionFunction(facePosition, faceAxis));
							}
							else
							{
								int adjacentRow = myVelocityIndex(Vec2ui(adjacentFace), faceAxis);
								assert(adjacentRow >= 0);

								solver.addElement(row, adjacentRow, -nodeSign * faceSign * nodeCoeff);
							}
						}
				}
			}
		});
	}

	bool solved = solver.solve();

	Real error = 0;

	for (auto axis : { 0,1 })
	{
		Vec2ui size = myVelocityIndex.size(axis);

		forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& face)
		{
			int row = myVelocityIndex(face, axis);

			if (row >= 0)
			{
				Vec2R facePosition = myVelocityIndex.indexToWorld(Vec2R(face), axis);
				Real localError = fabs(solver.solution(row) - solutionFunction(facePosition, axis));

				if (error < localError) error = localError;
			}
		});
	}

	return error;
}
