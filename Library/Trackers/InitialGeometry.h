#ifndef LIBRARY_INITIAL_GEOMETRY_H
#define LIBRARY_INITIAL_GEOMETRY_H

///////////////////////////////////
//
// InitialConditions.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

#include "Common.h"

class EdgeMesh;

namespace InitialGeometry
{
	EdgeMesh makeCircleMesh(const Vec2R& center = Vec2R(0), Real radius = 1., Real divisions = 10.);
	EdgeMesh makeSquareMesh(const Vec2R& center = Vec2R(0), const Vec2R& scale = Vec2R(1.));
	EdgeMesh makeDiamondMesh(const Vec2R& center = Vec2R(0), const Vec2R& scale = Vec2R(1.));
	EdgeMesh makeNotchedDiskMesh();
	EdgeMesh makeVortexMesh();
}

#endif