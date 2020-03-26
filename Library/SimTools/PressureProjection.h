#ifndef LIBRARY_PRESSURE_PROJECTION_H
#define LIBRARY_PRESSURE_PROJECTION_H

#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Utilities.h"
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

namespace FluidSim2D::SimTools
{

using namespace Utilities;
using namespace SurfaceTrackers;

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the solid boundary
	PressureProjection(const LevelSet& surface,
						const VectorGrid<float>& cutCellWeights,
						const VectorGrid<float>& ghostFluidWeights,
						const VectorGrid<float>& solidVelocity);

	void project(VectorGrid<float>& velocity);

	void setInitialGuess(const ScalarGrid<float>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myUseInitialGuessPressure = true;
		myInitialGuessPressure = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuessPressure = false;
	}

	ScalarGrid<float> getPressureGrid()
	{
		return myPressure;
	}

	const VectorGrid<VisitedCellLabels>& getValidFaces()
	{
		return myValidFaces;
	}

	void drawPressure(Renderer& renderer) const;

private:

	const VectorGrid<float>& mySolidVelocity;
	const VectorGrid<float>& myGhostFluidWeights;
	const VectorGrid<float>& myCutCellWeights;

	// Store flags for solved faces
	VectorGrid<VisitedCellLabels> myValidFaces;

	const LevelSet& mySurface;

	ScalarGrid<float> myPressure;

	const ScalarGrid<float>* myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}

#endif