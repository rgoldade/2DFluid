#ifndef FLUIDSIM2D_GRID_UTILITIES_H
#define FLUIDSIM2D_GRID_UTILITIES_H

#include <assert.h>

#include "Utilities.h"

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

namespace FluidSim2D
{

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

enum class Axis { XAXIS, YAXIS };

// Helper fuctions to map between geometry components in the grid.

FORCE_INLINE Vec2i cellToCell(const Vec2i& cell, int axis, int direction)
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

FORCE_INLINE Vec2i cellToFace(const Vec2i& cell, int axis, int direction)
{
	Vec2i face(cell);

	if (direction == 1)
		++face[axis];
	else assert(direction == 0);

	return face;
}

// Map cell to face using the same winding order as cellToNodeCCW
FORCE_INLINE Vec3i cellToFaceCCW(const Vec2i &cell, int index)
{
	int axis = index % 2 == 0 ? 1 : 0;
	Vec3i face(cell[0], cell[1], axis);

	assert(index < 4);

	if (index == 1 || index == 2)
		++face[axis];

	return face;
}

FORCE_INLINE Vec2i cellToNode(const Vec2i& cell, int nodeIndex)
{
	assert(nodeIndex >= 0 && nodeIndex < 4);

	Vec2i node(cell);

	for (int axis : {0, 1})
	{
		if (nodeIndex & (1 << axis))
		{
			++node[axis];
		}
	}

	return node;
}

// Map cell to nodes CCW from bottom-left
FORCE_INLINE Vec2i cellToNodeCCW(const Vec2i& cell, int nodeIndex)
{
	const Vec2i cellToNodeOffsets[4] = { Vec2i::Zero(), Vec2i(1, 0), Vec2i::Ones(), Vec2i(0, 1) };

	assert(nodeIndex >= 0 && nodeIndex < 4);

	Vec2i node(cell);
	node += cellToNodeOffsets[nodeIndex];

	return node;
}

FORCE_INLINE Vec2i faceToCell(const Vec2i& face, int axis, int direction)
{
	Vec2i cell(face);

	if (direction == 0)
		--cell[axis];
	else assert(direction == 1);

	return cell;
}

// An x-aligned face really means that the face normal is in the
// x-direction so the face nodes must be y-direction offsets.
FORCE_INLINE Vec2i faceToNode(const Vec2i& face, int faceAxis, int direction)
{
	assert(faceAxis >= 0 && faceAxis < 2);

	Vec2i node(face);

	if (direction == 1)
		++node[(faceAxis + 1) % 2];
	else assert(direction == 0);

	return node;
}

// Offset node index in the axis direction.
FORCE_INLINE Vec2i nodeToFace(const Vec2i& node, int offsetAxis, int direction)
{
	Vec2i face(node);

	if (direction == 0)
		--face[offsetAxis];
	else assert(direction == 1);

	return face;
}

FORCE_INLINE Vec2i nodeToCellCCW(const Vec2i& node, int cellIndex)
{
	const Vec2i nodeToCellOffsets[4] = { Vec2i::Zero(), Vec2i(-1,0), Vec2i(-1,-1), Vec2i(0,-1) };
	
	assert(cellIndex >= 0 && cellIndex < 4);

	Vec2i cell(node);
	cell += nodeToCellOffsets[cellIndex];

	return cell;
}

FORCE_INLINE Vec2i nodeToCell(const Vec2i& node, int cellIndex)
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

const std::vector<Vec3d> colour_swatch = { Vec3d(1, 0, 0),
											Vec3d(0, 1, 0),
											Vec3d(0, 0, 1),
											Vec3d(1, 1, 0),
											Vec3d(1, 0, 1),
											Vec3d(0, 1, 1) };

template<typename Function>
void forEachVoxelRange(const Vec2i& start, const Vec2i& end, const Function& f)
{
	Vec2i cell;
	for (cell[0] = start[0]; cell[0] < end[0]; ++cell[0])
		for (cell[1] = start[1]; cell[1] < end[1]; ++cell[1])
			f(cell);
}

template<typename Function>
void forEachVoxelRangeReverse(const Vec2i& start, const Vec2i& end, const Function& f)
{
	Vec2i cell;
	for (cell[0] = end[0] - 1; cell[0] >= start[0]; --cell[0])
		for (cell[1] = end[1] - 1; cell[1] >= start[1]; --cell[1])
			f(cell);
}

}

#endif