#ifndef LIBRARY_COMMON_H
#define LIBRARY_COMMON_H

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

static constexpr Real MINTHETA = 0.01;

// Helper arrays to iterate over neighbouring cells or faces
static const Vec2i cellToCellArray[] = { Vec2i(-1,0), Vec2i(1,0), Vec2i(0,-1), Vec2i(0,1) };
static const Vec2ui cellToFaceArray[] = { Vec2ui(0,0), Vec2ui(1,0), Vec2ui(0,0), Vec2ui(0,1) };
static const Vec2ui cellToNodeArray[] = { Vec2ui(0,0), Vec2ui(1,0), Vec2ui(0,1), Vec2ui(1,1) };
static const Vec2i faceToCellArray[2][2] = { { Vec2i(-1,0), Vec2i(0,0) },{ Vec2i(0,-1), Vec2i(0,0) } };
static const Vec2ui faceToNodeArray[2][2] = { { Vec2ui(0,0), Vec2ui(0,1) },
											{ Vec2ui(0,0), Vec2ui(1,0) } };

// First two indices are for offsets in the x,y direction. The third is that axis of the face and the fourth
// is the axis of the gradient when using finite differencing.
static const Vec4i nodeToFaceArray[4] = { Vec4i(-1,0,1,0), Vec4i(0,0,1,0), Vec4i(0,-1,0,1), Vec4i(0,0,0,1) };
static const Vec2i nodeToCellArray[4] = { Vec2i(0,0), Vec2i(-1,0), Vec2i(-1,-1), Vec2i(0,-1) };

// Helper fuctions to map between geometry components in the grid. It's preferrable to use these
// compared to the offsets above because only the necessary pieces are modified and the concepts
// are encapsulated better.

inline Vec2i cellToCell(Vec2i cell, const unsigned axis, const unsigned direction)
{
	assert(axis < 2 && direction < 2);

	if (direction == 0)
		--cell[axis];
	else
		++cell[axis];

	return cell;
}

inline Vec2ui cellToFace(Vec2ui cell, const unsigned axis, const unsigned direction)
{
	assert(axis < 2 && direction < 2);

	if (direction == 1)
		++cell[axis];
	
	return cell;
}

inline Vec2i faceToCell(Vec2i face, const unsigned axis, const unsigned direction)
{
	assert(axis < 2 && direction < 2);

	if (direction == 0)
		--face[axis];

	return face;
}

// An x-aligned face really means that the face normal is in the
// x-direction so the face nodes must be y-direction offsets.
inline Vec2ui faceToNode(Vec2ui face, const unsigned faceAxis, const unsigned direction)
{
	assert(faceAxis < 2 && direction < 2);

	if (direction == 1)
		++face[(faceAxis + 1) % 2];

	return face;
}

// Map cell to nodes CCW from bottom-left
inline Vec2ui cellToNode(Vec2ui cell, unsigned index)
{
	assert(index < 4);
	switch (index)
	{
	case 1:
		++cell[0];
		break;
	case 2:
		cell += Vec2ui(1);
		break;
	case 3:
		++cell[1];
		break;
	}

	return cell;
}

// Map cell to face using the same winding order as cellToNode
inline Vec3ui cellToFace(const Vec2ui &cell, unsigned index)
{
	Vec3ui face;
	face[0] = cell[0]; face[1] = cell[1];
	unsigned axis = index % 2 == 0 ? 1 : 0;
	face[2] = axis;
	assert(index < 4);
	
	if (index == 1 || index == 2)
		++face[axis];
	
	return face;
}

// Offset node index in the axis direction.
inline Vec2i nodeToFace(Vec2i node, const unsigned axis, const unsigned direction)
{
	assert(axis < 2 && direction < 2);

	if (direction == 0)
		--node[axis];

	return node;
}

inline Vec2i nodeToCell(Vec2i node, const unsigned cellIndex)
{
	if (cellIndex == 2 || cellIndex == 3)
		--node[0];
	if (cellIndex == 1 || cellIndex == 2)
		--node[1];

	return node;
}

static const Vec3f colours[] = { Vec3f(1., 0., 0.),
									Vec3f(0., 1., 0.),
									Vec3f(0., 0., 1.),
									Vec3f(1., 1., 0.),
									Vec3f(1., 0., 1.),
									Vec3f(0., 1., 1.) };

inline Real lengthFraction(Real phi0, Real phi1)
{
	Real theta = 0.;

	if (phi0 < 0)
	{
		if (phi1 < 0)
			theta = 1.;
		else if (phi1 >= 0)
			theta = phi0 / (phi0 - phi1);
	}
	else if (phi0 >= 0 && phi1 < 0)
		theta = phi1 / (phi1 - phi0);

	return theta;
}

template<typename T, typename Function>
void forEachVoxelRange(const Vec<T, 2>& start, const Vec<T, 2>& end, const Function& f)
{
	for (T i = start[0]; i < end[0]; ++i)
		for (T j = start[1]; j < end[1]; ++j)
			f(Vec<T, 2>(i, j));
}

// BFS markers
enum class MarkedCells { UNVISITED = -1, VISITED = 0, FINISHED = 1 };

// The marching squares template uses a binary encoding of
// inside/outside cell nodes to provide a set of
// grid edges + isosurface intersection to build mesh edges from.
//
// 3 --- 2 --- 2
// |           |
// 3           1
// |           |
// 0 --- 0 --- 1
//

static const int marchingSquaresTemplate[16][4] = { { -1,-1,-1,-1 },
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

#endif