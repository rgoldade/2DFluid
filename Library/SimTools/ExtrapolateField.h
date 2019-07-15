#ifndef LIBRARY_EXTRAPOLATEFIELD_H
#define LIBRARY_EXTRAPOLATEFIELD_H

#include <queue>

#include "Common.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ExtrapolateField.h/cpp
// Ryan Goldade 2016
//
// Extrapolates field from the boundary
// of a mask outward based on a simple
// BFS flood fill approach. Values
// are averaged from FINISHED neighbours.
// Note that because this process happens
// "in order" there could be some bias
// based on which boundary locations
// are inserted into the queue first.
//
//
////////////////////////////////////

template<typename Field>
class ExtrapolateField
{
public:
	ExtrapolateField(Field& field)
		: myField(field)
		{}

	void extrapolate(const ScalarGrid<MarkedCells>& mask, int bandwidth = std::numeric_limits<int>::max());

private:
	Field& myField;
};

template<typename Field>
void ExtrapolateField<Field>::extrapolate(const ScalarGrid<MarkedCells>& mask, int bandwidth)
{
	assert(bandwidth >= 0);

	// Run a BFS flood outwards from masked cells and average the values of the neighbouring "finished" cells
	// It could be made more accurate if we used the value of the "closer" cell (smaller SDF value)
	// It could be made more efficient if we truncated the BFS after a large enough distance (max SDF value)
	assert(myField.isGridMatched(mask));

	// Make local copy of mask
	UniformGrid<MarkedCells> markedCells(myField.size(), MarkedCells::UNVISITED);

	forEachVoxelRange(Vec2i(0), myField.size(), [&](const Vec2i& cell)
	{
		if (mask(cell) == MarkedCells::FINISHED) markedCells(cell) = MarkedCells::FINISHED;
	});

	using Voxel = std::pair<Vec2i, int>;

	// Initialize flood fill queue
	std::queue<Voxel> markerQ;

	// Load up neighbouring faces and push into queue
	forEachVoxelRange(Vec2i(0), myField.size(), [&](const Vec2i& cell)
	{
		if (markedCells(cell) == MarkedCells::FINISHED)
		{
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					// Boundary check
					if (direction == 0 && adjacentCell[axis] < 0)
						continue;
					else if (direction == 1 && adjacentCell[axis] >= markedCells.size()[axis])
						continue;

					if (markedCells(adjacentCell) == MarkedCells::UNVISITED)
					{
						markerQ.push(Voxel(adjacentCell, 1));
						markedCells(adjacentCell) = MarkedCells::VISITED;
					}
				}
		}
	});

	while (!markerQ.empty())
	{
		// Store reference to updated faces to set as finished after
		// this layer is completed
		std::queue<Voxel> tempQ = markerQ;
		// Store references to next layer of faces
		std::queue<Voxel> newLayer;

		while (!markerQ.empty())
		{
			Voxel voxel = markerQ.front();
			Vec2i cell = voxel.first;
			markerQ.pop();

			assert(markedCells(cell) == MarkedCells::VISITED);

			Real value = 0.;
			Real count = 0.;

			if (voxel.second < bandwidth)
			{
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						// Boundary check
						if (direction == 0 && adjacentCell[axis] < 0)
							continue;
						else if (direction == 1 && adjacentCell[axis] >= markedCells.size()[axis])
							continue;

						if (markedCells(adjacentCell) == MarkedCells::FINISHED)
						{
							value += myField(adjacentCell);
							++count;
						}
						else if (markedCells(adjacentCell) == MarkedCells::UNVISITED)
						{
							newLayer.push(Voxel(adjacentCell, voxel.second + 1));
							markedCells(adjacentCell) = MarkedCells::VISITED;
						}
					}

				assert(count > 0);
				myField(cell) = value / count;
			}
		}

		//Set update cells to finished
		while (!tempQ.empty())
		{
			Voxel voxel = tempQ.front();
			Vec2i localCell = voxel.first;
			tempQ.pop();
			assert(markedCells(localCell) == MarkedCells::VISITED);
			markedCells(localCell) = MarkedCells::FINISHED;
		}

		//Copy new queue
		std::swap(markerQ, newLayer);
	}
}

#endif