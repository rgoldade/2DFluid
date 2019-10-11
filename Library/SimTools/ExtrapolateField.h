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

template <typename Field> class ExtrapolateField {
public:
  ExtrapolateField(Field &field) : myField(field) {}

  void extrapolate(const ScalarGrid<MarkedCells> &mask);
  void extrapolate(const ScalarGrid<MarkedCells> &mask, int bandwidth);

private:
  Field &myField;
};

template <typename Field>
void ExtrapolateField<Field>::extrapolate(const ScalarGrid<MarkedCells> &mask) {
  int bandwidth{mask.size()[0] * mask.size()[1]};
  extrapolate(mask, bandwidth);
}

template <typename Field>
void ExtrapolateField<Field>::extrapolate(const ScalarGrid<MarkedCells> &mask,
                                          int bandwidth) {
  assert(bandwidth >= 0);

  // Run a BFS flood outwards from masked cells and average the values of the
  // neighbouring "finished" cells It could be made more accurate if we used the
  // value of the "closer" cell (smaller SDF value) It could be made more
  // efficient if we truncated the BFS after a large enough distance (max SDF
  // value)
  assert(myField.isGridMatched(mask));

  // Make local copy of mask
  UniformGrid<MarkedCells> markedCells(myField.size(), MarkedCells::UNVISITED);

  forEachVoxelRange(Vec2i(0), myField.size(), [&](const Vec2i &cell) {
    if (mask(cell) == MarkedCells::FINISHED)
      markedCells(cell) = MarkedCells::FINISHED;
  });

  // TODO: make all loops parallel

  // Build initial list of cells to process
  std::vector<Vec2i> toVisitCellList;

  // Load up neighbouring faces and push into queue
  forEachVoxelRange(Vec2i(0), markedCells.size(), [&](const Vec2i &cell) {
    if (markedCells(cell) == MarkedCells::FINISHED) {
      for (int axis : {0, 1})
        for (int direction : {0, 1}) {
          Vec2i adjacentCell = cellToCell(cell, axis, direction);

          // Boundary check
          if (adjacentCell[axis] < 0 ||
              adjacentCell[axis] >= markedCells.size()[axis])
            continue;

          if (markedCells(adjacentCell) == MarkedCells::UNVISITED) {
            toVisitCellList.push_back(adjacentCell);
            markedCells(adjacentCell) = MarkedCells::VISITED;
          }
        }
    }
  });

  std::vector<Vec2i> newCellList;
  for (int layer = 0; layer < bandwidth; ++layer) {
    if (!toVisitCellList.empty()) {
      newCellList.clear();

      for (const auto &cell : toVisitCellList) {
        assert(markedCells(cell) == MarkedCells::VISITED);

        Real accumulatedValue = 0.;
        Real accumulatedCells = 0.;

        for (int axis : {0, 1})
          for (int direction : {0, 1}) {
            Vec2i adjacentCell = cellToCell(cell, axis, direction);

            // Boundary check
            if (adjacentCell[axis] < 0 ||
                adjacentCell[axis] >= markedCells.size()[axis])
              continue;

            if (markedCells(adjacentCell) == MarkedCells::FINISHED) {
              accumulatedValue += myField(adjacentCell);
              ++accumulatedCells;
            } else if (markedCells(adjacentCell) == MarkedCells::UNVISITED) {
              newCellList.push_back(adjacentCell);
              markedCells(adjacentCell) = MarkedCells::VISITED;
            }
          }

        assert(accumulatedCells > 0);
        myField(cell) = accumulatedValue / accumulatedCells;
      }

      // Set update cells to finished
      for (const auto &cell : toVisitCellList) {
        assert(markedCells(cell) == MarkedCells::VISITED);
        markedCells(cell) = MarkedCells::FINISHED;
      }

      std::swap(toVisitCellList, newCellList);
    }
  }
}

#endif
