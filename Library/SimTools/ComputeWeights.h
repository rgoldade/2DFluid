#ifndef FLUIDSIM2D_COMPUTE_WEIGHTS_H
#define FLUIDSIM2D_COMPUTE_WEIGHTS_H

#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ComputeWeights.h/cpp
// Ryan Goldade 2017
//
// Useful collection of tools to compute
// control volume weights for use in both
// pressure projection and viscosity solves.
//
////////////////////////////////////

namespace FluidSim2D
{

VectorGrid<double> computeGhostFluidWeights(const LevelSet& surface);

VectorGrid<double> computeCutCellWeights(const LevelSet& surface, bool invert = false);

ScalarGrid<double> computeSupersampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType, int samples);

VectorGrid<double> computeSupersampledFaceAreas(const LevelSet& surface, int samples);

}

#endif