#ifndef LIBRARY_VISCOSITY_SOLVER_H
#define LIBRARY_VISCOSITY_SOLVER_H

#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Utilities.h"
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

namespace FluidSim2D::SimTools
{
using namespace SurfaceTrackers;
using namespace Utilities;

void ViscositySolver(float dt,
						const LevelSet& surface,
						VectorGrid<float>& velocity,
						const LevelSet& solidSurface,
						const VectorGrid<float>& solidVelocity,
						const ScalarGrid<float>& viscosity);
}

#endif