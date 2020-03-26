#ifndef LIBRARY_PREDICATES_H
#define LIBRARY_PREDICATES_H

#include "GridUtilities.h"
#include "Vec.h"

namespace FluidSim2D::SurfaceTrackers
{
using namespace Utilities;

using REAL = float;
using Vec2R = Vec<2, REAL>;

REAL exactinit(); // call this before anything else

REAL orient2d(const REAL *pa,
	const REAL *pb,
	const REAL *pc);

REAL orient3d(const REAL *pa,
	const REAL *pb,
	const REAL *pc,
	const REAL *pd);

REAL incircle(const REAL *pa,
	const REAL *pb,
	const REAL *pc,
	const REAL *pd);

REAL insphere(const REAL *pa,
	const REAL *pb,
	const REAL *pc,
	const REAL *pd,
	const REAL *pe);

enum class IntersectionLabels { YES, ON, NO };

// Check that a mesh edge crosses over a grid edge.
// Rotate grid edge to x-axis to perform check.
IntersectionLabels exactEdgeIntersect(Vec2R q, Vec2R r, Vec2R s, Axis axis);

}
#endif