#ifndef LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H
#define LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H

#include "Common.h"
#include "ComputeWeights.h"
#include "GeometricCGPoissonSolver.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// GeometricPressureProjection.h/cpp
// Ryan Goldade 2019
//
////////////////////////////////////

class GeometricPressureProjection
{
public:
	GeometricPressureProjection(const LevelSet& surface,
								const LevelSet& solidSurface,
								const VectorGrid<Real>& solidVelocity)
		: mySurface(surface)
		, mySolidSurface(solidSurface)
		, mySolidVelocity(solidVelocity)
		, myGhostFluidWeights(computeGhostFluidWeights(surface))
		, myCutCellWeights(computeCutCellWeights(solidSurface, true))
	{
		assert(surface.isGridMatched(solidSurface));

		assert(solidVelocity.size(0)[0] - 1 == surface.size()[0] &&
			solidVelocity.size(0)[1] == surface.size()[1] &&
			solidVelocity.size(1)[0] == surface.size()[0] &&
			solidVelocity.size(1)[1] - 1 == surface.size()[1]);

		myPressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
		myValidFaces = VectorGrid<MarkedCells>(surface.xform(), surface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
		myDomainCellLabels = UniformGrid<GeometricMGOperations::CellLabels>(surface.size(), GeometricMGOperations::CellLabels::EXTERIOR);
	}

	void project(VectorGrid<Real>& velocity,
					const bool useMGPreconditioner);

	void setInitialGuess(const ScalarGrid<Real>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myPressure = initialGuessPressure;
	}

	ScalarGrid<Real> getPressureGrid()
	{
		return myPressure;
	}

	const VectorGrid<MarkedCells>& getValidFaces()
	{
		return myValidFaces;
	}

	void drawPressure(Renderer& renderer) const;

private:

	const VectorGrid<Real> &mySolidVelocity;

	VectorGrid<MarkedCells> myValidFaces; // Store solved faces

	const LevelSet &mySurface, &mySolidSurface;

	ScalarGrid<Real> myPressure;

	UniformGrid<GeometricMGOperations::CellLabels> myDomainCellLabels;
	const VectorGrid<Real> myGhostFluidWeights, myCutCellWeights;
};

#endif