#ifndef LIBRARY_GEOMETRIC_PREDICATES_H
#define LIBRARY_GEOMETRIC_PREDICATES_H

#include "Common.h"

Real exactinit(); // call this before anything else

Real orient2d(const Real *pa,
			    const Real *pb,
				const Real *pc);

Real orient3d(const Real *pa,
				const Real *pb,
				const Real *pc,
				const Real *pd);

Real incircle(const Real *pa,
				const Real *pb,
				const Real *pc,
				const Real *pd);

Real insphere(const Real *pa,
				const Real *pb,
				const Real *pc,
				const Real *pd,
				const Real *pe);

enum class Intersection { YES, ON, NO };
enum class Axis { XAXIS, YAXIS };
// Check that a mesh edge crosses over a grid edge.
// Rotate grid edge to x-axis to perform check.
Intersection exactEdgeIntersect(const Vec2R q, const Vec2R r, const Vec2R s, Axis axis = Axis::XAXIS);

#endif