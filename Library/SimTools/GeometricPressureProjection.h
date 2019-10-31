#ifndef LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H
#define LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H

#include "Common.h"
#include "ComputeWeights.h"
#include "GeometricConjugateGradientSolver.h"
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
	using CellLabels = GeometricMultigridOperators::CellLabels;

	using StoreReal = double;
	using SolveReal = double;

public:
	GeometricPressureProjection(const LevelSet& surface,
								const VectorGrid<Real>& cutCellWeights,
								const VectorGrid<Real>& ghostFluidWeights,
								const VectorGrid<Real>& solidVelocity)
		: mySurface(surface)
		, myCutCellWeights(cutCellWeights)
		, myGhostFluidWeights(ghostFluidWeights)
		, mySolidVelocity(solidVelocity)
		, myUseInitialGuessPressure(false)
		, myInitialGuessPressure(nullptr)
	{
		assert(solidVelocity.size(0)[0] - 1 == surface.size()[0] &&
				solidVelocity.size(0)[1] == surface.size()[1] &&
				solidVelocity.size(1)[0] == surface.size()[0] &&
				solidVelocity.size(1)[1] - 1 == surface.size()[1]);

		assert(solidVelocity.sampleType() == VectorGridSettings::SampleType::STAGGERED);

		assert(solidVelocity.isGridMatched(cutCellWeights) &&
				solidVelocity.isGridMatched(ghostFluidWeights));

		myPressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
		myValidFaces = VectorGrid<MarkedCells>(surface.xform(), surface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
	}

	void project(VectorGrid<Real>& velocity,
					const bool useMGPreconditioner);

	void setInitialGuess(const ScalarGrid<Real>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myUseInitialGuessPressure = true;
		myInitialGuessPressure = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuessPressure = false;
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

	const LevelSet &mySurface;

	ScalarGrid<Real> myPressure;

	const ScalarGrid<Real> *myInitialGuessPressure;
	bool myUseInitialGuessPressure;
	const VectorGrid<Real>& myGhostFluidWeights, &myCutCellWeights;
};

#endif