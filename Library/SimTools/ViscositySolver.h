#ifndef LIBRARY_VISCOSITYSOLVER_H
#define LIBRARY_VISCOSITYSOLVER_H

#include "LevelSet2D.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ViscositySolver.h/cpp
// Ryan Goldade 2017
//
// Variational viscosity solver.
// Uses ghost fluid weights for moving
// collisions. Uses volume control
// weights for the various tensor and
// velocity sample positions.
//
////////////////////////////////////

class ViscositySolver
{
	static constexpr int UNSOLVED = -2;
	static constexpr int SOLIDBOUNDARY = -1;

public:

	ViscositySolver(Real dt, const LevelSet2D& surface, VectorGrid<Real>& velocity,
					const LevelSet2D& solidSurface, const VectorGrid<Real>& solidVelocity)
		: myDt(dt)
		, myVelocity(velocity)
		, mySurface(surface)
		, mySolidVelocity(solidVelocity)
		, mySolidSurface(solidSurface)
		{
			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(mySurface.isMatched(mySolidSurface));

			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(myVelocity.size(0)[0] - 1 == mySurface.size()[0] &&
				myVelocity.size(0)[1] == mySurface.size()[1] &&
				myVelocity.size(1)[0] == mySurface.size()[0] &&
				myVelocity.size(1)[1] - 1 == mySurface.size()[1]);

			assert(mySolidVelocity.size(0)[0] - 1 == mySurface.size()[0] &&
				mySolidVelocity.size(0)[1] == mySurface.size()[1] &&
				mySolidVelocity.size(1)[0] == mySurface.size()[0] &&
				mySolidVelocity.size(1)[1] - 1 == mySurface.size()[1]);
		}

	void setViscosity(Real mu);
	void setViscosity(const ScalarGrid<Real>& mu);

	void solve(const VectorGrid<Real>& faceVolumes,
				ScalarGrid<Real>& centerVolumes,
				ScalarGrid<Real>& nodeVolumes,
				const ScalarGrid<Real>& solidCenterVolumes,
				const ScalarGrid<Real>& solidNodeVolumes);
private:

	VectorGrid<Real>& myVelocity;
	const VectorGrid<Real>& mySolidVelocity;
	
	const LevelSet2D& mySurface;
	const LevelSet2D& mySolidSurface;

	ScalarGrid<Real> myViscosity;

	Real myDt;
};

#endif