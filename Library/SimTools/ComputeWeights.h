#ifndef LIBRARY_COMPUTEWEIGHTS_H
#define LIBRARY_COMPUTEWEIGHTS_H

#include "Common.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"

class LevelSet;

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

VectorGrid<Real> computeGhostFluidWeights(const LevelSet& surface);
VectorGrid<Real> computeCutCellWeights(const LevelSet& surface, bool invert = false);
ScalarGrid<Real> computeSuperSampledAreas(const LevelSet& surface,
											ScalarGridSettings::SampleType sampleType,
											int samples);
VectorGrid<Real> computeSuperSampledFaceAreas(const LevelSet& surface, int samples);

#endif