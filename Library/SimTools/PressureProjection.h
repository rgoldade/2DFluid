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
static constexpr Real MINTHETA = 0.01;

class PressureProjection
{

public:
	// For variational solve, surface should be extrapolated into the collision volume
	PressureProjection(Real dt, const LevelSet2D& surface, const VectorGrid<Real>& vel,
							const LevelSet2D& collision, const VectorGrid<Real>& collisionVelocity)
		: myDt(dt)
		, mySurface(surface)
		, myVelocity(vel)
		, myCollision(collision)
		, myCollisionVelocity(collisionVelocity)
		{
			assert(surface.isMatched(collision));

			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(vel.size(0)[0] - 1 == surface.size()[0] &&
				vel.size(0)[1] == surface.size()[1] &&
				vel.size(1)[0] == surface.size()[0] &&
				vel.size(1)[1] - 1 == surface.size()[1]);

			assert(collisionVelocity.size(0)[0] - 1 == surface.size()[0] &&
				collisionVelocity.size(0)[1] == surface.size()[1] &&
				collisionVelocity.size(1)[0] == surface.size()[0] &&
				collisionVelocity.size(1)[1] - 1 == surface.size()[1]);

			myPressure = ScalarGrid<Real>(surface.xform(), surface.size(), 0);
			myValid = VectorGrid<Real>(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
			myLiquidCells = UniformGrid<int>(surface.size(), UNSOLVED);
		}
	
	// The liquid weights refer to the volume of liquid in each cell. This is useful for ghost fluid.
	// Note that the surface should be extrapolated into the collision volume by 1 voxel before computing the
	// weights. 
	// The fluid weights refer to the cut-cell length of fluid (air and liquid) through a cell face.
	// In both cases, 0 means "empty" and 1 means "full".
	void project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights);

	// Apply solution to a velocity field at solvable faces
	void applySolution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights);
	void applyValid(VectorGrid<Real> &valid);

	void drawPressure(Renderer& renderer) const;

private:

	const VectorGrid<Real>& myVelocity, &myCollisionVelocity;

	VectorGrid<Real> myValid; // Store solved faces

	const LevelSet2D &mySurface, &myCollision;
	
	ScalarGrid<Real> myPressure;
	UniformGrid<int> myLiquidCells;
	Real myDt;
};

#endif