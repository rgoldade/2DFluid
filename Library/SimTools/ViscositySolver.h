#ifndef LIBRARY_VISCOSITYSOLVER_H
#define LIBRARY_VISCOSITYSOLVER_H

#include "Common.h"
#include "LevelSet.h"
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

namespace ViscosityCellLabels
{
	constexpr int UNSOLVED_CELL = -2;
	constexpr int SOLID_CELL = -1;
}

void ViscositySolver(const Real dt,
						const LevelSet& surface,
						VectorGrid<Real>& velocity,
						const LevelSet& solidSurface,
						const VectorGrid<Real>& solidVelocity,
						const ScalarGrid<Real>& viscosity);

#endif