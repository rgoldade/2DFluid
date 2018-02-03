#pragma once

#include "Vec.h"
#include "Predicates.h"

///////////////////////////////////
//
// core.h
// Ryan Goldade 2016
//
// Everything should include this since
// we're defining floating point values
// as Real and not double or float.
//
//
////////////////////////////////////

typedef double Real;
typedef Vec<2, Real> Vec2R;
typedef Vec<3, Real> Vec3R;

// Helper arrays to iterate over neighbouring cells or faces
static Vec2i cell_offset[] = { Vec2i(-1,0), Vec2i(1,0), Vec2i(0,-1), Vec2i(0,1) }; // cell to cell offset
static Vec2i cell_to_face[] = { Vec2i(0,0), Vec2i(1,0), Vec2i(0,0), Vec2i(0,1) };
static Vec2i cell_to_node[] = { Vec2i(0,0), Vec2i(1,0), Vec2i(0,1), Vec2i(1,1) };
static Vec2i face_to_cell[2][2] = { { Vec2i(-1,0), Vec2i(0,0) },{ Vec2i(0,-1), Vec2i(0,0) } };
static Vec2i face_to_node[2][2] = { { Vec2i(0,0), Vec2i(0,1) },
									{ Vec2i(0,0), Vec2i(1,0) } };

static Vec4i node_to_face[4] = { Vec4i(-1,0,1,0), Vec4i(0,0,1,0), Vec4i(0,-1,0,1), Vec4i(0,0,0,1) };
static Vec2i node_to_cell[4] = { Vec2i(0,0), Vec2i(-1,0), Vec2i(-1,-1), Vec2i(0,-1) };

static Vec3f colours[] = { Vec3f(1., 0., 0.),
Vec3f(0., 1., 0.),
Vec3f(0., 0., 1.),
Vec3f(1., 1., 0.),
Vec3f(1., 0., 1.),
Vec3f(0., 1., 1.) };

// BFS markers
enum marked { UNVISITED, VISITED, FINISHED };

// Check that a mesh edge crosses over a grid edge.
// Rotate grid edge to x-axis to perform check.
enum AXIS { XAXIS, YAXIS };
enum INTERSECTION { YES, ON, NO };
static INTERSECTION exact_edge_intersect(const Vec2R &a, const Vec2R &b, const Vec2R &c, AXIS axis = XAXIS)
{
	Vec2R q, r, s;
	if (axis == YAXIS)
	{
		q = Vec2R(a[1], a[0]);
		r = Vec2R(b[1], b[0]);
		s = Vec2R(c[1], c[0]);
	}
	else
	{
		q = a;
		r = b;
		s = c;
	}

	// Make sure y-axis bb crosses x-axis grid
	// Degenerate cases should be handled with predicates
	if ((q[1] > s[1] && r[1] > s[1]) ||
		(q[1] <= s[1] && r[1] <= s[1])) return NO;

	//Sort for winding
	if (q[1] > r[1])
	{
		Vec2R temp = q;
		q = r;
		r = temp;
	}

	Real qrs = orient2d(q.v, r.v, s.v);

	// Check that "s" lies to the left of the edge, 
	// signifying that axis aligned ray starting 
	// from "s" point in the positive direction
	// will intersect the edge
	if (qrs > 0) return YES;

	// Check that "s" lies strictly on the edge.
	// This needs to be handled 
	else if (qrs == 0.) return ON;

	// Ray "s" doesn't intersect the edge
	else return NO;
}

static const int mc_template[16][4] =
{ { -1,-1,-1,-1 },
{ 3, 0,-1,-1 },
{ 0, 1,-1,-1 },
{ 3, 1,-1,-1 },

{ 1, 2,-1,-1 },
{ 3, 0, 1, 2 },
{ 0, 2,-1,-1 },
{ 3, 2,-1,-1 },

{ 2, 3,-1,-1 },
{ 2, 0,-1,-1 },
{ 0, 1, 2, 3 },
{ 2, 1,-1,-1 },

{ 1, 3,-1,-1 },
{ 1, 0,-1,-1 },
{ 0, 3,-1,-1 },
{ -1,-1,-1,-1 } };
