#pragma once

#include "Vec.h"

///////////////////////////////////
//
// Common.h
// Ryan Goldade 2016
//
// Everything should include this since
// we're defining floating point values
// as Real and not double or float.
//
////////////////////////////////////

using Real = double;
using Vec2R = Vec<Real, 2>;
using Vec3R = Vec<Real, 3>;

// Helper arrays to iterate over neighbouring cells or faces
static const Vec2i cell_to_cell[] = { Vec2i(-1,0), Vec2i(1,0), Vec2i(0,-1), Vec2i(0,1) };
static const Vec2ui cell_to_face[] = { Vec2ui(0,0), Vec2ui(1,0), Vec2ui(0,0), Vec2ui(0,1) };
static const Vec2ui cell_to_node[] = { Vec2ui(0,0), Vec2ui(1,0), Vec2ui(0,1), Vec2ui(1,1) };
static const Vec2i face_to_cell[2][2] = { { Vec2i(-1,0), Vec2i(0,0) },{ Vec2i(0,-1), Vec2i(0,0) } };
static const Vec2ui face_to_node[2][2] = { { Vec2ui(0,0), Vec2ui(0,1) },
											{ Vec2ui(0,0), Vec2ui(1,0) } };

// First two indices are for offsets in the x,y direction. The third is that axis of the face and the fourth
// is the axis of the gradient when using finite differencing.
static const Vec4i node_to_face[4] = { Vec4i(-1,0,1,0), Vec4i(0,0,1,0), Vec4i(0,-1,0,1), Vec4i(0,0,0,1) };
static const Vec2i node_to_cell[4] = { Vec2i(0,0), Vec2i(-1,0), Vec2i(-1,-1), Vec2i(0,-1) };

static const Vec3f colours[] = { Vec3f(1., 0., 0.),
									Vec3f(0., 1., 0.),
									Vec3f(0., 0., 1.),
									Vec3f(1., 1., 0.),
									Vec3f(1., 0., 1.),
									Vec3f(0., 1., 1.) };

template<typename T, typename Function>
void for_each_voxel_range(const Vec<T, 2>& start, const Vec<T, 2>& end, const Function& f)
{
	for (T i = start[0]; i < end[0]; ++i)
		for (T j = start[1]; j < end[1]; ++j)
			f(Vec<T, 2>(i, j));
}

// BFS markers
enum class MarkedCells { UNVISITED, VISITED, FINISHED };

static const int mc_template[16][4] = { { -1,-1,-1,-1 },
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