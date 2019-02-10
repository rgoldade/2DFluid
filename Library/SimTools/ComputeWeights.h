#ifndef LIBRARY_COMPUTEWEIGHTS_H
#define LIBRARY_COMPUTEWEIGHTS_H

#include "Common.h"
#include "LevelSet2D.h"
#include "ScalarGrid.h"
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

VectorGrid<Real> computeGhostFluidWeights(const LevelSet2D& surface);
VectorGrid<Real> computeCutCellWeights(const LevelSet2D &surface, bool invert = false);
ScalarGrid<Real> computeSupersampledAreas(const LevelSet2D& surface,
	ScalarGridSettings::SampleType sampleType,
	unsigned samples);
VectorGrid<Real> computeSupersampledFaceAreas(const LevelSet2D& surface, unsigned samples);

#endif