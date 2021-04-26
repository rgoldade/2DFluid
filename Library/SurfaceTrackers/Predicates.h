#ifndef FLUIDSIM2D_PREDICATES_H
#define FLUIDSIM2D_PREDICATES_H

#include "GridUtilities.h"

namespace FluidSim2D
{

using REAL = double;
using Vec2R = Vec2t<REAL>;

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