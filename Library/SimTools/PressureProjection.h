#ifndef FLUIDSIM2D_PRESSURE_PROJECTION_H
#define FLUIDSIM2D_PRESSURE_PROJECTION_H

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

namespace FluidSim2D
{

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the solid boundary
	PressureProjection(const LevelSet& surface,
						const VectorGrid<double>& cutCellWeights,
						const VectorGrid<double>& ghostFluidWeights,
						const VectorGrid<double>& solidVelocity);

	void project(VectorGrid<double>& velocity);

	void setInitialGuess(const ScalarGrid<double>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myUseInitialGuessPressure = true;
		myInitialGuessPressure = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuessPressure = false;
	}

	ScalarGrid<double> getPressureGrid()
	{
		return myPressure;
	}

	const VectorGrid<VisitedCellLabels>& getValidFaces()
	{
		return myValidFaces;
	}

	void drawPressure(Renderer& renderer) const;

private:

	const VectorGrid<double>& mySolidVelocity;
	const VectorGrid<double>& myGhostFluidWeights;
	const VectorGrid<double>& myCutCellWeights;

	// Store flags for solved faces
	VectorGrid<VisitedCellLabels> myValidFaces;

	const LevelSet& mySurface;

	ScalarGrid<double> myPressure;

	const ScalarGrid<double>* myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}

#endif