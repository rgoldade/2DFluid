#ifndef LIBRARY_PRESSURE_PROJECTION_H
#define LIBRARY_PRESSURE_PROJECTION_H

#include "Common.h"
#include "ComputeWeights.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// PressureProjection.h/cpp
// Ryan Goldade 2017
//
// Variational pressure solve. Allows
// for moving solids.
//
////////////////////////////////////

namespace PressureCellLabels
{
	constexpr int UNSOLVED_CELL = -1;
	constexpr int FLUID_CELL = 0;
}

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the solid boundary
	PressureProjection(const LevelSet& surface,
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
	
	void project(VectorGrid<Real>& velocity);

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
	const VectorGrid<Real> &myGhostFluidWeights, &myCutCellWeights;
	VectorGrid<MarkedCells> myValidFaces;

	const LevelSet &mySurface;
	
	ScalarGrid<Real> myPressure;

	const ScalarGrid<Real> *myInitialGuessPressure;
	bool myUseInitialGuessPressure;
	
};

#endif