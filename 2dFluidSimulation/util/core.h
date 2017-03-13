#pragma once

#include <vec.h>

///////////////////////////////////
//
// grid.h/cpp
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

// Helper arrays to iterate over neighbouring cells or faces
static Vec2i cell_offset[] = { Vec2i(-1,0), Vec2i(1,0), Vec2i(0,-1), Vec2i(0,1) }; // cell to cell offset
static Vec2i cell_to_face[] = { Vec2i(0,0), Vec2i(1,0), Vec2i(0,0), Vec2i(0,1) };
static Vec2i face_to_cell[2][2] = { { Vec2i(-1,0), Vec2i(0,0) },{ Vec2i(0,-1), Vec2i(0,0) } };
// BFS markers
enum marked { UNVISITED, VISITED, FINISHED };