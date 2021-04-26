#ifndef FLUIDSIM2D_INITIAL_GEOMETRY_H
#define FLUIDSIM2D_INITIAL_GEOMETRY_H

#include "EdgeMesh.h"
#include "Utilities.h"

///////////////////////////////////
//
// InitialGeometry.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

namespace FluidSim2D
{

EdgeMesh makeCircleMesh(const Vec2d& center = Vec2d::Zero(), double radius = 1, double divisions = 10);
EdgeMesh makeSquareMesh(const Vec2d& center = Vec2d::Zero(), const Vec2d& scale = Vec2d::Ones());
EdgeMesh makeDiamondMesh(const Vec2d& center = Vec2d::Zero(), const Vec2d& scale = Vec2d::Ones());
EdgeMesh makeNotchedDiskMesh();
EdgeMesh makeVortexMesh();

}

#endif