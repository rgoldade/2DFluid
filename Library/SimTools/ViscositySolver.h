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
	static constexpr int COLLISION = -1;

public:

	ViscositySolver(Real dt, const LevelSet2D& surface, VectorGrid<Real>& velocity,
					const LevelSet2D& collision, const VectorGrid<Real>& collision_vel)
		: myDt(dt)
		, myVelocity(velocity)
		, mySurface(surface)
		, myCollisionVelocity(collision_vel)
		, myCollisionSurface(collision)
		{
			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(mySurface.isMatched(myCollisionSurface));

			// For efficiency sake, this should only take in velocity on a staggered grid
			// that matches the center sampled surface and collision
			assert(myVelocity.size(0)[0] - 1 == mySurface.size()[0] &&
				myVelocity.size(0)[1] == mySurface.size()[1] &&
				myVelocity.size(1)[0] == mySurface.size()[0] &&
				myVelocity.size(1)[1] - 1 == mySurface.size()[1]);

			assert(myCollisionVelocity.size(0)[0] - 1 == mySurface.size()[0] &&
				myCollisionVelocity.size(0)[1] == mySurface.size()[1] &&
				myCollisionVelocity.size(1)[0] == mySurface.size()[0] &&
				myCollisionVelocity.size(1)[1] - 1 == mySurface.size()[1]);
		}

	void setViscosity(Real mu);
	void setViscosity(const ScalarGrid<Real>& mu);

	void solve(const VectorGrid<Real>& faceVolumes,
				ScalarGrid<Real>& centerVolumes,
				ScalarGrid<Real>& nodeVolumes,
				const ScalarGrid<Real>& collisionCenterVolumes,
				const ScalarGrid<Real>& collisionNodeVolumes);
private:

	VectorGrid<Real>& myVelocity;
	const VectorGrid<Real>& myCollisionVelocity;
	
	const LevelSet2D& mySurface;
	const LevelSet2D& myCollisionSurface;

	ScalarGrid<Real> myViscosity;

	Real myDt;
};

#endif