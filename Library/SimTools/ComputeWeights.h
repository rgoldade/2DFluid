#ifndef LIBRARY_COMPUTE_WEIGHTS_H
#define LIBRARY_COMPUTE_WEIGHTS_H

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

namespace FluidSim2D::SimTools
{

using namespace SurfaceTrackers;
using namespace Utilities;

VectorGrid<float> computeGhostFluidWeights(const LevelSet& surface);

VectorGrid<float> computeCutCellWeights(const LevelSet& surface, bool invert = false);

ScalarGrid<float> computeSupersampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType, int samples);

VectorGrid<float> computeSupersampledFaceAreas(const LevelSet& surface, int samples);

}

#endif