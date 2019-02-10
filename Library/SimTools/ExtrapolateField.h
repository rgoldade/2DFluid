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

	void extrapolate(const Field& mask, unsigned bandwidth = std::numeric_limits<unsigned>::max());

private:
	Field& myField;
};

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, unsigned bandwidth);

template<typename Field>
void ExtrapolateField<Field>::extrapolate(const Field& mask, unsigned bandwidth)
{
	// Run a BFS flood outwards from masked cells and average the values of the neighbouring "finished" cells
	// It could be made more accurate if we used the value of the "closer" cell (smaller SDF value)
	// It could be made more efficient if we truncated the BFS after a large enough distance (max SDF value)
	assert(myField.isMatched(mask));

	UniformGrid<MarkedCells> markedCells(myField.size(), MarkedCells::UNVISITED);

	using Voxel = std::pair<Vec2ui, unsigned>;

	// Initialize flood fill queue
	std::queue<Voxel> markerQ;

	// If a face has fluid through it, the weight should be greater than zero
	// and so it should be marked
	forEachVoxelRange(Vec2ui(0), myField.size(), [&](const Vec2ui& cell)
	{
		if (mask(cell) > 0.) markedCells(cell) = MarkedCells::FINISHED;
	});

	// Load up neighbouring faces and push into queue
	forEachVoxelRange(Vec2ui(0), myField.size(), [&](const Vec2ui& cell)
	{
		if (markedCells(cell) == MarkedCells::FINISHED)
		{
			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

					// Boundary check
					if (direction == 0 && adjacentCell[axis] < 0)
						continue;
					else if (direction == 1 && adjacentCell[axis] >= markedCells.size()[axis])
						continue;

					if (markedCells(Vec2ui(adjacentCell)) == MarkedCells::UNVISITED)
					{
						markerQ.push(Voxel(Vec2ui(adjacentCell), 1));
						markedCells(Vec2ui(adjacentCell)) = MarkedCells::VISITED;
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
			Vec2ui cell = voxel.first;
			markerQ.pop();

			assert(markedCells(cell) == MarkedCells::VISITED);

			Real value = 0.;
			Real count = 0.;

			if (voxel.second < bandwidth)
			{
				for (unsigned axis : {0, 1})
					for (unsigned direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

						// Boundary check
						if (direction == 0 && adjacentCell[axis] < 0)
							continue;
						else if (direction == 1 && adjacentCell[axis] >= markedCells.size()[axis])
							continue;

						if (markedCells(Vec2ui(adjacentCell)) == MarkedCells::FINISHED)
						{
							value += myField(Vec2ui(adjacentCell));
							++count;
						}
						else if (markedCells(Vec2ui(adjacentCell)) == MarkedCells::UNVISITED)
						{
							newLayer.push(Voxel(Vec2ui(adjacentCell), voxel.second + 1));
							markedCells(Vec2ui(adjacentCell)) = MarkedCells::VISITED;
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
			Vec2ui idx = voxel.first;
			tempQ.pop();
			assert(markedCells(idx) == MarkedCells::VISITED);
			markedCells(idx) = MarkedCells::FINISHED;
		}

		//Copy new queue
		std::swap(markerQ, newLayer);
	}
}

#endif