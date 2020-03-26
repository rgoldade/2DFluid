#ifndef LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H
#define LIBRARY_GEOMETRIC_PRESSURE_PROJECTION_H

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

namespace FluidSim2D::SimTools
{

using namespace SurfaceTrackers;
using namespace Utilities;

class GeometricPressureProjection
{
	using StoreReal = double;
	using SolveReal = double;

public:
	GeometricPressureProjection(const LevelSet& surface,
								const VectorGrid<float>& cutCellWeights,
								const VectorGrid<float>& ghostFluidWeights,
								const VectorGrid<float>& solidVelocity);

	void project(VectorGrid<float>& velocity, bool useMGPreconditioner);

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

	const ScalarGrid<float> *myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}

#endif