#ifndef LIBRARY_EXTRAPOLATE_FIELD_H
#define LIBRARY_EXTRAPOLATE_FIELD_H

#include "UniformGrid.h"
#include "Utilities.h"
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

namespace FluidSim2D::SimTools
{

using namespace Utilities;

template<typename Field>
void extrapolateField(Field& field, UniformGrid<VisitedCellLabels> finishedCellMask, int bandwidth)
{
	assert(bandwidth > 0);
	assert(field.size() == finishedCellMask.size());

	// Build an initial list of cells adjacent to finished cells in the provided mask grid

	std::vector<Vec2i> toVisitCells;

	tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelToVisitCells;
	tbb::parallel_for(tbb::blocked_range<int>(0, finishedCellMask.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localToVisitCells = parallelToVisitCells.local();
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = finishedCellMask.unflatten(cellIndex);

			// Load up adjacent unfinished cells
			if (finishedCellMask(cell) == VisitedCellLabels::FINISHED_CELL)
			{
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis])
							continue;

						if (finishedCellMask(adjacentCell) != VisitedCellLabels::FINISHED_CELL)
							localToVisitCells.push_back(adjacentCell);
					}
			}
		}
	});

	mergeLocalThreadVectors(toVisitCells, parallelToVisitCells);

	auto vecCompare = [](const Vec2i& vec0, const Vec2i& vec1)
	{
          if (vec0[0] < vec1[0])
            return true;
          else if (vec0[0] == vec1[0] && vec0[1] < vec1[1])
            return true;

		return false;
	};

	// Now flood outwards layer-by-layer
	for (int layer = 0; layer < bandwidth; ++layer)
	{
		// First sort the list because there could be duplicates
		tbb::parallel_sort(toVisitCells.begin(), toVisitCells.end(), vecCompare);

		// Compute values from adjacent finished cells
		tbb::parallel_for(tbb::blocked_range<int>(0, toVisitCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			// Because the list could contain duplicates, we need to advance forward through possible duplicates
			int cellIndex = range.begin();

			if (cellIndex > 0)
			{
				while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
					++cellIndex;
			}

			Vec2i oldCell(-1);

			for (; cellIndex < range.end(); ++cellIndex)
			{
				Vec2i cell = toVisitCells[cellIndex];

				if (cell == oldCell)
					continue;

				oldCell = cell;

				assert(finishedCellMask(cell) != VisitedCellLabels::FINISHED_CELL);

				// TODO: get template type from field instead of assuming float is valid
				float accumulatedValue = 0;
				float accumulatedCount = 0;

				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis])
							continue;

						if (finishedCellMask(adjacentCell) == VisitedCellLabels::FINISHED_CELL)
						{
							accumulatedValue += field(adjacentCell);
							++accumulatedCount;
						}
					}

				assert(accumulatedCount > 0);

				field(cell) = accumulatedValue / accumulatedCount;
			}
		});

		// Set visited cells to finished
		tbb::parallel_for(tbb::blocked_range<int>(0, toVisitCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			// Because the list could contain duplicates, we need to advance forward through possible duplicates
			int cellIndex = range.begin();

			if (cellIndex > 0)
			{
				while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
					++cellIndex;
			}

			Vec2i oldCell(-1);

			for (; cellIndex < range.end(); ++cellIndex)
			{
				Vec2i cell = toVisitCells[cellIndex];

				if (cell == oldCell)
					continue;

				oldCell = cell;

				assert(finishedCellMask(cell) != VisitedCellLabels::FINISHED_CELL);
				finishedCellMask(cell) = VisitedCellLabels::FINISHED_CELL;
			}
		});

		// Build new layer of cells
		if (layer < bandwidth - 1)
		{
			parallelToVisitCells.clear();

			tbb::parallel_for(tbb::blocked_range<int>(0, toVisitCells.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto& localToVisitCells = parallelToVisitCells.local();

				// Because the list could contain duplicates, we need to advance forward through possible duplicates
				int cellIndex = range.begin();

				if (cellIndex > 0)
				{
					while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
						++cellIndex;
				}

				Vec2i oldCell(-1);

				for (; cellIndex < range.end(); ++cellIndex)
				{
					Vec2i cell = toVisitCells[cellIndex];

					if (cell == oldCell)
						continue;

					oldCell = cell;

					assert(finishedCellMask(cell) == VisitedCellLabels::FINISHED_CELL);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis])
								continue;

							if (finishedCellMask(adjacentCell) != VisitedCellLabels::FINISHED_CELL)
								localToVisitCells.push_back(adjacentCell);
						}
				}
			});

			toVisitCells.clear();
			mergeLocalThreadVectors(toVisitCells, parallelToVisitCells);
		}
	}
}

}
#endif