#ifndef FLUIDSIM2D_GEOMETRIC_PRESSURE_PROJECTION_H
#define FLUIDSIM2D_GEOMETRIC_PRESSURE_PROJECTION_H

#include "GeometricConjugateGradientSolver.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// GeometricPressureProjection.h/cpp
// Ryan Goldade 2019
//
////////////////////////////////////

namespace FluidSim2D
{

class GeometricPressureProjection
{
	using StoreReal = double;
	using SolveReal = double;

public:
	GeometricPressureProjection(const LevelSet& surface,
								const VectorGrid<double>& cutCellWeights,
								const VectorGrid<double>& ghostFluidWeights,
								const VectorGrid<double>& solidVelocity);

	void project(VectorGrid<double>& velocity, bool useMGPreconditioner);

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

	const ScalarGrid<double> *myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}

#endif