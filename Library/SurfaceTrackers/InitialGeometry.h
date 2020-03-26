#ifndef LIBRARY_INITIAL_GEOMETRY_H
#define LIBRARY_INITIAL_GEOMETRY_H

#include "EdgeMesh.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// InitialGeometry.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

namespace FluidSim2D::SurfaceTrackers
{

EdgeMesh makeCircleMesh(const Vec2f& center = Vec2f(0), float radius = 1., float divisions = 10.);
EdgeMesh makeSquareMesh(const Vec2f& center = Vec2f(0), const Vec2f& scale = Vec2f(1.));
EdgeMesh makeDiamondMesh(const Vec2f& center = Vec2f(0), const Vec2f& scale = Vec2f(1.));
EdgeMesh makeNotchedDiskMesh();
EdgeMesh makeVortexMesh();

}

#endif