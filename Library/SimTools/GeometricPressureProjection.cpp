#include "GeometricPressureProjection.h"

#include "GeometricCGPoissonSolver.h"
#include "GeometricMGPoissonSolver.h"

void GeometricPressureProjection::drawPressure(Renderer& renderer) const
{
	myPressure.drawSupersampledValues(renderer, .25, 1, 2);
}

void GeometricPressureProjection::project(VectorGrid<Real>& velocity,
											const bool useMGPreconditioner)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
		// that matches the center sampled liquid and solid surfaces.
	assert(velocity.size(0)[0] - 1 == mySurface.size()[0] &&
			velocity.size(0)[1] == mySurface.size()[1] &&
			velocity.size(1)[0] == mySurface.size()[0] &&
			velocity.size(1)[1] - 1 == mySurface.size()[1]);

	bool hasDirichlet = false;

	// Build domain labels
	forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i& cell)
	{
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
			myDomainCellLabels(cell) = GeometricMGOperations::CellLabels::EXTERIOR;
		else if (isInsideLiquid)
			myDomainCellLabels(cell) = GeometricMGOperations::CellLabels::INTERIOR;
		else
		{
			myDomainCellLabels(cell) = GeometricMGOperations::CellLabels::DIRICHLET;
			hasDirichlet = true;
		}
	});

	// Build RHS
	UniformGrid<Real> rhsGrid(mySurface.size(), 0);
	
	forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i& cell)
	{
		if (myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::INTERIOR)
		{
			// Build RHS divergence
			double divergence = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					Real weight = myCutCellWeights(face, axis);

					double sign = (direction == 0) ? 1 : -1;

					if (weight > 0)
						divergence += sign * velocity(face, axis) * weight;
					if (weight < 1.)
						divergence += sign * mySolidVelocity(face, axis) * (1. - weight);
				}

			rhsGrid(cell) = divergence / mySurface.dx();
		}
	});

	// Set a single interior cell to dirichlet and remove the average divergence
	if (!hasDirichlet)
	{
		Real averageDivergence = 0;
		Real cellCount = 0;

		forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i& cell)
		{
			if (myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::INTERIOR)
			{
				averageDivergence += rhsGrid(cell);
				++cellCount;
			}
		});

		averageDivergence /= cellCount;

		forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i& cell)
		{
			if (myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::INTERIOR)
				rhsGrid(cell) -= averageDivergence;
		});

		// Remove divergence
		bool hasRemovedInteriorCell = false;
		forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i& cell)
		{
			if (!hasRemovedInteriorCell &&
				myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::INTERIOR)
			{
				bool hasNonInterior = false;
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						if (myDomainCellLabels(cell) != GeometricMGOperations::CellLabels::INTERIOR)
							hasNonInterior = true;
					}

				if (!hasNonInterior)
				{
					hasRemovedInteriorCell = true;
					myDomainCellLabels(cell) = GeometricMGOperations::CellLabels::DIRICHLET;
				}
			}
		});
	}

	// Build poisson face weights
	VectorGrid<Real> poissonFaceWeights(mySurface.xform(), mySurface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		Vec2i start(0);
		++start[axis];
		Vec2i end = poissonFaceWeights.size(axis);
		--end[axis];

		forEachVoxelRange(start, end, [&](const Vec2i& face)
		{
			Real weight = myCutCellWeights(face, axis);
			if (weight > 0)
			{
				bool localHasInterior = false;
				bool localHasDirichlet = false;
				for (int direction : {0, 1})
				{
					Vec2i cell = faceToCell(face, axis, direction);
					if (myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::DIRICHLET)
						localHasDirichlet = true;
					if (myDomainCellLabels(cell) == GeometricMGOperations::CellLabels::INTERIOR)
						localHasInterior = true;
				}

				if (localHasInterior)
				{
					if (localHasDirichlet)
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

	Real dx = mySurface.dx();
	auto MatrixVectorMultiply = [&myDomainCellLabels = this->myDomainCellLabels, &poissonFaceWeights, dx](UniformGrid<Real> &residualGrid, const UniformGrid<Real> &solutionGrid)
	{
		// Matrix-vector multiplication
		applyWeightedPoissonMatrix(residualGrid, solutionGrid, myDomainCellLabels, poissonFaceWeights, dx);
	};

	UniformGrid<Real> diagonalPrecondGrid(mySurface.size(), 0);

	forEachVoxelRange(Vec2i(0), mySurface.size(), [&](const Vec2i &cell)
	{
		Real gridScalar = 1. / Util::sqr(dx);
		if (myDomainCellLabels(cell) == CellLabels::INTERIOR)
		{
			Real diagonal = 0;
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);
					Vec2i face = cellToFace(cell, axis, direction);

					if (myDomainCellLabels(adjacentCell) == CellLabels::INTERIOR ||
						myDomainCellLabels(adjacentCell) == CellLabels::DIRICHLET)
					{
						diagonal += poissonFaceWeights(face, axis);
					}
					else assert(poissonFaceWeights(face, axis) == 0);
				}
			diagonal *= gridScalar;
			diagonalPrecondGrid(cell) = 1. / diagonal;
		}
	});

	auto DiagonalPreconditioner = [&myDomainCellLabels = this->myDomainCellLabels, &diagonalPrecondGrid](UniformGrid<Real> &solutionGrid,
		const UniformGrid<Real> &rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		forEachVoxelRange(Vec2i(0), solutionGrid.size(), [&](const Vec2i &cell)
		{
			if (myDomainCellLabels(cell) == CellLabels::INTERIOR)
			{
				solutionGrid(cell) = rhsGrid(cell) * diagonalPrecondGrid(cell);
			}
		});
	};

	// Pre-build multigrid preconditioner
	GeometricMGPoissonSolver MGPreconditioner(myDomainCellLabels, 4, dx);
	MGPreconditioner.setGradientWeights(poissonFaceWeights);

	auto MultiGridPreconditioner = [&MGPreconditioner](UniformGrid<Real> &solutionGrid,
		const UniformGrid<Real> &rhsGrid)
	{
		assert(solutionGrid.size() == rhsGrid.size());
		MGPreconditioner.applyMGVCycle(solutionGrid, rhsGrid);
	};

	auto DotProduct = [&myDomainCellLabels = this->myDomainCellLabels](const UniformGrid<Real> &grid0,
		const UniformGrid<Real> &grid1) -> Real
	{
		assert(grid0.size() == grid1.size());
		return dotProduct(grid0, grid1, myDomainCellLabels);
	};

	auto L2Norm = [&myDomainCellLabels = this->myDomainCellLabels](const UniformGrid<Real> &grid) -> Real
	{
		return l2Norm(grid, myDomainCellLabels);
	};

	auto AddScaledVector = [&myDomainCellLabels = this->myDomainCellLabels](UniformGrid<Real> &destination,
		const UniformGrid<Real> &unscaledSource,
		const UniformGrid<Real> &scaledSource,
		const Real scale)
	{
		addToVector(destination, unscaledSource, scaledSource, myDomainCellLabels, scale);
	};

	if (useMGPreconditioner)
	{
		solveGeometricCGPoisson(dynamic_cast<UniformGrid<Real>&>(myPressure),
			rhsGrid,
			MatrixVectorMultiply,
			MultiGridPreconditioner,
			DotProduct,
			L2Norm,
			AddScaledVector,
			1E-5 /*solver tolerance*/, false, true, false);
	}
	else
	{
		solveGeometricCGPoisson(dynamic_cast<UniformGrid<Real>&>(myPressure),
			rhsGrid,
			MatrixVectorMultiply,
			DiagonalPreconditioner,
			DotProduct,
			L2Norm,
			AddScaledVector,
			1E-5 /*solver tolerance*/, true, false, true);
	}

	// Set valid faces
	for (int axis : {0, 1})
	{
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
			{
				if ((myDomainCellLabels(backwardCell) == GeometricMGOperations::CellLabels::INTERIOR ||
					myDomainCellLabels(forwardCell) == GeometricMGOperations::CellLabels::INTERIOR) &&
					myCutCellWeights(face, axis) > 0)
					myValidFaces(face, axis) = MarkedCells::FINISHED;
				else myValidFaces(face, axis) = MarkedCells::UNVISITED;
			}
			else myValidFaces(face, axis) = MarkedCells::UNVISITED;
		});
	}

	for (int axis : {0, 1})
	{
		Vec2i size = velocity.size(axis);

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& face)
		{
			Real localVelocity = 0;
			if (myValidFaces(face, axis) == MarkedCells::FINISHED)
			{
				Real theta = myGhostFluidWeights(face, axis);
				theta = Util::clamp(theta, MINTHETA, Real(1.));

				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
				{
					Real gradient = 0;

					if (myDomainCellLabels(Vec2i(backwardCell)) == GeometricMGOperations::CellLabels::INTERIOR)
						gradient -= myPressure(Vec2i(backwardCell));

					if (myDomainCellLabels(Vec2i(forwardCell)) == GeometricMGOperations::CellLabels::INTERIOR)
						gradient += myPressure(Vec2i(forwardCell));

					localVelocity = velocity(face, axis) - gradient / (theta * myPressure.dx());
				}
			}

			velocity(face, axis) = localVelocity;
		});
	}
}