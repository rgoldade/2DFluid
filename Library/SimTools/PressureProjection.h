#ifndef LIBRARY_PRESSUREPROJECTION_H
#define LIBRARY_PRESSUREPROJECTION_H

#include "Common.h"
#include "LevelSet2D.h"
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

static constexpr int UNSOLVED = -1;

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the solid boundary
	PressureProjection(const LevelSet2D& surface, const VectorGrid<Real>& liquidVelocity,
			    const LevelSet2D& solidSurface, const VectorGrid<Real>& solidVelocity)
		: myFluidSurface(surface)
		, myFluidVelocity(liquidVelocity)
		, mySolidSurface(solidSurface)
		, mySolidVelocity(solidVelocity)
		, myUseInitialGuess(false)
		, myInitialGuess(nullptr)
	{
		assert(surface.isGridMatched(solidSurface));

		// For efficiency sake, this should only take in velocity on a staggered grid
		// that matches the center sampled liquid and solid surfaces.
		assert(liquidVelocity.size(0)[0] - 1 == surface.size()[0] &&
			liquidVelocity.size(0)[1] == surface.size()[1] &&
			liquidVelocity.size(1)[0] == surface.size()[0] &&
			liquidVelocity.size(1)[1] - 1 == surface.size()[1]);

		assert(solidVelocity.size(0)[0] - 1 == surface.size()[0] &&
			solidVelocity.size(0)[1] == surface.size()[1] &&
			solidVelocity.size(1)[0] == surface.size()[0] &&
			solidVelocity.size(1)[1] - 1 == surface.size()[1]);

		myPressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
		myValid = VectorGrid<MarkedCells>(surface.xform(), surface.size(), MarkedCells::UNVISITED, VectorGridSettings::SampleType::STAGGERED);
		myFluidCellIndex = UniformGrid<int>(surface.size(), UNSOLVED);
	}
	
	// The liquid weights refer to the volume of liquid in each cell. This is useful for ghost fluid.
	// Note that the surface should be extrapolated into the solid boundary by 1 voxel before computing the
	// weights. 
	// The fluid weights refer to the cut-cell length of fluid (air and liquid) through a cell face.
	// In both cases, 0 means "empty" and 1 means "full".
	void project(const VectorGrid<Real>& ghostFluidWeights, const VectorGrid<Real>& cutCellWeights);

	void setInitialGuess(const ScalarGrid<Real>& initialGuessPressure)
	{
		assert(myFluidSurface.isGridMatched(initialGuessPressure));
		myUseInitialGuess = true;
		myInitialGuess = &initialGuessPressure;
	}

	ScalarGrid<Real> getPressureGrid()
	{
		return myPressure;
	}

	// Apply solution to a velocity field at solvable faces
	void applySolution(VectorGrid<Real>& velocity, const VectorGrid<Real>& ghostFluidWeights);
	void applyValid(VectorGrid<MarkedCells> &valid);

	void drawPressure(Renderer& renderer) const;

private:

	const VectorGrid<Real> &myFluidVelocity, &mySolidVelocity;

	VectorGrid<MarkedCells> myValid; // Store solved faces

	const LevelSet2D &myFluidSurface, &mySolidSurface;
	
	ScalarGrid<Real> myPressure;
	UniformGrid<int> myFluidCellIndex;

	const ScalarGrid<Real> *myInitialGuess;
	bool myUseInitialGuess;
};

#endif