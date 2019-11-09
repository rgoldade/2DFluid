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

using Real = float;
using Vec2R = Vec<Real, 2>;
using Vec3R = Vec<Real, 3>;

constexpr int tbbGrainSize = 500;
constexpr Real MINTHETA = 0.01;

// Helper fuctions to map between geometry components in the grid. It's preferrable to use these
// compared to the offsets above because only the necessary pieces are modified and the concepts
// are encapsulated better.

inline Vec2i cellToCell(const Vec2i& cell, int axis, int direction)
{
	Vec2i adjacentCell(cell);
	if (direction == 0)
		--adjacentCell[axis];
	else
	{
		assert(direction == 1);
		++adjacentCell[axis];
	}

	return adjacentCell;
}

inline Vec2i cellToFace(const Vec2i& cell, int axis, int direction)
{
	Vec2i face(cell);
	if (direction == 1)
		++face[axis];
	else assert(direction == 0);

	return face;
}

inline Vec2i faceToCell(const Vec2i& face, int axis, int direction)
{
	Vec2i cell(face);
	if (direction == 0)
		--cell[axis];
	else assert(direction == 1);

	return cell;
}

// An x-aligned face really means that the face normal is in the
// x-direction so the face nodes must be y-direction offsets.
inline Vec2i faceToNode(const Vec2i& face, int faceAxis, int direction)
{
	assert(faceAxis >= 0 && faceAxis < 2);

	Vec2i node(face);
	if (direction == 1)
		++node[(faceAxis + 1) % 2];
	else assert(direction == 0);

	return node;
}

// Map cell to nodes CCW from bottom-left
inline Vec2i cellToNode(const Vec2i& cell, int nodeIndex)
{
	assert(nodeIndex >= 0 && nodeIndex < 4);

	Vec2i node(cell);
	for (int axis : {0, 1})
	{
		if (nodeIndex & (1 << axis))
			++node[axis];
	}

	return node;
}

// Map cell to nodes CCW from bottom-left
inline Vec2i cellToNodeCCW(const Vec2i& cell, int nodeIndex)
{
	Vec2i node(cell);
	assert(nodeIndex >= 0 && nodeIndex < 4);

	switch (nodeIndex)
	{
	case 1:
		++node[0];
		break;
	case 2:
		node += Vec2i(1);
		break;
	case 3:
		++node[1];
		break;
	}

	return node;
}

// Map cell to face using the same winding order as cellToNodeCCW
inline Vec3i cellToFaceCCW(const Vec2i &cell, int index)
{
	int axis = index % 2 == 0 ? 1 : 0;
	Vec3i face(cell[0], cell[1], axis);

	assert(index < 4);
	
	if (index == 1 || index == 2)
		++face[axis];
	
	return face;
}

// Offset node index in the axis direction.
inline Vec2i nodeToFace(const Vec2i& node, int faceAxis, int direction)
{
	Vec2i face(node);

	if (direction == 0)
		--face[faceAxis];
	else assert(direction == 1);

	return face;
}

inline Vec2i nodeToCellCCW(const Vec2i& node, int cellIndex)
{
	Vec2i cell(node);

	assert(cellIndex >= 0 && cellIndex < 4);
	if (cellIndex == 2 || cellIndex == 3)
		--cell[0];
	if (cellIndex == 1 || cellIndex == 2)
		--cell[1];

	return cell;
}

inline Vec2i nodeToCell(const Vec2i& node, int cellIndex)
{
	assert(cellIndex >= 0 && cellIndex < 4);

	Vec2i cell(node);
	for (int axis : {0, 1})
	{
		if (!(cellIndex & (1 << axis)))
			--cell[axis];
	}

	return cell;
}

const Vec3f colours[] = { Vec3f(1., 0., 0.),
							Vec3f(0., 1., 0.),
							Vec3f(0., 0., 1.),
							Vec3f(1., 1., 0.),
							Vec3f(1., 0., 1.),
							Vec3f(0., 1., 1.) };

template<typename RealType>
RealType lengthFraction(RealType phi0, RealType phi1)
{
	RealType theta = 0.;

	if (phi0 <= 0)
	{
		if (phi1 <= 0)
			theta = 1.;
		else // if (phi1 > 0)
			theta = phi0 / (phi0 - phi1);
	}
	else if (phi0 > 0 && phi1 <= 0)
		theta = phi1 / (phi1 - phi0);

	assert(theta >= 0 && theta <= 1);

	return theta;
}

template<typename T, typename Function>
void forEachVoxelRange(const Vec<T, 2>& start, const Vec<T, 2>& end, const Function& f)
{
	Vec<T, 2> cell;
	for (cell[0] = start[0]; cell[0] < end[0]; ++cell[0])
		for (cell[1] = start[1]; cell[1] < end[1]; ++cell[1])
			f(cell);
}

template<typename T, typename Function>
void forEachVoxelRangeReverse(const Vec<T, 2>& start, const Vec<T, 2>& end, const Function& f)
{
	Vec<T, 2> cell;
	for (cell[0] = end[0] - 1; cell[0] >= start[0]; --cell[0])
		for (cell[1] = end[1] - 1; cell[1] >= start[1]; --cell[1])
			f(cell);
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