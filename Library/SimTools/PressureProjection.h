#ifndef LIBRARY_PRESSUREPROJECTION_H
#define LIBRARY_PRESSUREPROJECTION_H

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
						const LevelSet& solidSurface,
						const VectorGrid<Real>& solidVelocity)
		: mySurface(surface)
		, mySolidSurface(solidSurface)
		, mySolidVelocity(solidVelocity)
		, myGhostFluidWeights(computeGhostFluidWeights(surface))
		, myCutCellWeights(computeCutCellWeights(solidSurface, true))
		, myUseInitialGuess(false)
		, myInitialGuess(nullptr)
	{
		assert(surface.isGridMatched(solidSurface));

		assert(solidVelocity.size(0)[0] - 1 == surface.size()[0] &&
			solidVelocity.size(0)[1] == surface.size()[1] &&
			solidVelocity.size(1)[0] == surface.size()[0] &&
			solidVelocity.size(1)[1] - 1 == surface.size()[1]);

		myPressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
		myValidFaces = VectorGrid<MarkedCells>(surface.xform(), surface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
		myFluidCellIndex = UniformGrid<int>(surface.size(), PressureCellLabels::UNSOLVED_CELL);
	}
	
	// The liquid weights refer to the volume of liquid in each cell. This is useful for ghost fluid.
	// Note that the surface should be extrapolated into the solid boundary by 1 voxel before computing the
	// weights. 
	// The fluid weights refer to the cut-cell length of fluid (air and liquid) through a cell face.
	// In both cases, 0 means "empty" and 1 means "full".
	void project(VectorGrid<Real>& velocity);

	void setInitialGuess(const ScalarGrid<Real>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myUseInitialGuess = true;
		myInitialGuess = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuess = false;
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
	UniformGrid<int> myFluidCellIndex;

	const ScalarGrid<Real> *myInitialGuess;
	bool myUseInitialGuess;
	const VectorGrid<Real> myGhostFluidWeights, myCutCellWeights;
};

#endif