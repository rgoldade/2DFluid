#include <iostream>

#include "QuadtreeGrid.h"

namespace FluidSim2D
{

QuadtreeGrid::QuadtreeGrid(const Transform& xform, const Vec2i& size, int desiredLevels)
    : myBaseSize(size)
{
    // We will stretch the quadtree grid to be the smallest power of 2
    // that will contain the entire supplied grid size.
    // Because of this, we need to keep the original grid size so
    // we don't set ACTIVE cells outside of the supplied grid region.
    Vec2i expandedSize;
    for (int axis : {0, 1})
    {
        double logSize = std::log2(double(size[axis]));
        logSize = std::ceil(logSize);
        expandedSize[axis] = int(std::exp2(logSize));
    }

    // Cap levels to be the floor of the log-base-2 of the smallest
    // axis-aligned dimension.
    // We do this because we don't want empty levels. For example, an
    // input grid with that is 64^2 can only represent 6 levels so even
    // if the input is 7 levels, we should only generate 6.
    myQuadtreeLevels = desiredLevels;
    if (expandedSize[0] > 0 && expandedSize[1] > 0)
    {
        for (int axis : {0, 1})
        {
            if (std::log2(expandedSize[axis]) < myQuadtreeLevels)
            {
                myQuadtreeLevels = int(std::log2(expandedSize[axis]));
            }
        }
    }

    myQuadtreeLabels.resize(myQuadtreeLevels);
    myQuadtreeLabels[0] = ScalarGrid<QuadtreeCellLabel>(xform, expandedSize, QuadtreeCellLabel::INACTIVE);

    double dx = xform.dx();

    for (int level = 1; level < myQuadtreeLevels; ++level)
    {
        dx *= 2.;
        expandedSize = expandedSize / 2;

        Transform tempXform(dx, xform.offset());

        myQuadtreeLabels[level] = ScalarGrid<QuadtreeCellLabel>(tempXform, expandedSize, QuadtreeCellLabel::INACTIVE);
    }
}

int QuadtreeGrid::levels() const { return int(myQuadtreeLabels.size()); }

Vec2i QuadtreeGrid::size(int level) const { return myQuadtreeLabels[level].size(); }

Transform QuadtreeGrid::xform(int level) const { return myQuadtreeLabels[level].xform(); }

const ScalarGrid<QuadtreeGrid::QuadtreeCellLabel>& QuadtreeGrid::grid(int level) const { return myQuadtreeLabels[level]; }

std::vector<std::pair<Vec2i, int>> QuadtreeGrid::getFaceAdjacentCells(const Vec2i& cell, int axis, int direction, int level) const
{
    assert(axis < 2 && direction < 2);

    std::vector<std::pair<Vec2i, int>> adjacentCellList;

    Vec2i adjacentCell = cellToCell(cell, axis, direction);

    if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size(level)[axis])
    {
        return std::vector<std::pair<Vec2i, int>>();
    }

    QuadtreeCellLabel adjacentLabel = myQuadtreeLabels[level](adjacentCell);

    if (adjacentLabel == QuadtreeCellLabel::ACTIVE)
    {
        adjacentCellList.push_back(std::make_pair(adjacentCell, level));
    }
    else if (adjacentLabel == QuadtreeCellLabel::UP)
    {
        Vec2i parentCell = getParentCell(adjacentCell);
        assert(isCellActive(parentCell, level + 1));
        adjacentCellList.push_back(std::make_pair(parentCell, level + 1));
    }
    else if (adjacentLabel == QuadtreeCellLabel::DOWN)
    {
        for (int secondDirection : {0, 1})
        {
            int childIndex = 0;
            if (direction == 0)
            {
                childIndex += 1 << axis;
            }
            if (secondDirection == 1)
            {
                childIndex += 1 << (axis + 1) % 2;
            }

            Vec2i childCell = getChildCell(adjacentCell, childIndex);

            if (isCellActive(childCell, level - 1))
            {
                adjacentCellList.push_back(std::make_pair(childCell, level - 1));
            }
            else
            {
                assert(myQuadtreeLabels[level - 1](childCell) == QuadtreeCellLabel::INACTIVE);
            }
        }
    }

    return adjacentCellList;
}

void QuadtreeGrid::refineGrid()
{
    Transform tempXform(xform(0).dx() / 2., xform(0).offset());
    Vec2i tempSize = size(0) * 2;
    int levels = int(myQuadtreeLabels.size());

    // Set up new marker grids
    std::vector<ScalarGrid<QuadtreeCellLabel>> localQuadtreeLabels;
    localQuadtreeLabels.resize(levels);
    localQuadtreeLabels[0] = ScalarGrid<QuadtreeCellLabel>(tempXform, tempSize, QuadtreeCellLabel::INACTIVE);

    double dx = tempXform.dx();
    for (int level = 1; level < levels; ++level)
    {
        dx *= 2.;
        tempSize /= 2;

        Transform localXform(dx, tempXform.offset());

        localQuadtreeLabels[level] = ScalarGrid<QuadtreeCellLabel>(localXform, tempSize, QuadtreeCellLabel::INACTIVE);
    }

    // Fill new marker grids based on existing but one octave lower
    for (int level = 0; level < levels; ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), localQuadtreeLabels[level].size(), [&](const Vec2i& cell)
        {
            Vec2i parentCell = getParentCell(cell);
            localQuadtreeLabels[level](cell) = myQuadtreeLabels[level](parentCell);
        });
    }

    std::swap(myQuadtreeLabels, localQuadtreeLabels);
}

void QuadtreeGrid::drawGrid(Renderer& renderer)
{
    for (int level = 0; level < myQuadtreeLabels.size(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), myQuadtreeLabels[level].size(), [&](const Vec2i& cell)
        {
            if (myQuadtreeLabels[level](cell) == QuadtreeCellLabel::ACTIVE)
            {
                myQuadtreeLabels[level].drawGridCell(renderer, cell, Vec3d::Zero());
            }
        });
    }
}

void QuadtreeGrid::drawCellConnections(Renderer& renderer)
{
    VecVec2d startPoints;
    VecVec2d endPoints;

    for (int level = 0; level < myQuadtreeLabels.size(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), myQuadtreeLabels[level].size(), [&](const Vec2i& cell)
            {
                if (myQuadtreeLabels[level](cell) == QuadtreeCellLabel::ACTIVE)
                {
                    Vec2d cellPos = myQuadtreeLabels[level].indexToWorld(cell.cast<double>());

                    for (int axis : {0, 1})
                    {
                        for (int direction : {0, 1})
                        {
                            auto faceAdjacentCells = getFaceAdjacentCells(cell, axis, direction, level);

                            for (const auto& adjacentCellPair : faceAdjacentCells)
                            {
                                const Vec2i& adjacentCell = adjacentCellPair.first;
                                int adjacentLevel = adjacentCellPair.second;

                                Vec2d adjacentCellPos = myQuadtreeLabels[adjacentLevel].indexToWorld(adjacentCell.cast<double>());
                                startPoints.push_back(cellPos);
                                endPoints.push_back(adjacentCellPos);
                            }
                        }
                    }
                }
            });
    }

    renderer.addLines(startPoints, endPoints, Vec3d(0, 0, 1));
}


void QuadtreeGrid::setActiveCellsAndParentDown(std::vector<Vec2i>& parentsDown, int level)
{
    tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelParentsDown;

    tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[level].voxelCount()), [&](const tbb::blocked_range<int>& range)
    {
        auto& localParentsDown = parallelParentsDown.local();
        for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
            Vec2i cell = myQuadtreeLabels[level].unflatten(flatIndex);

            QuadtreeCellLabel activity = myQuadtreeLabels[level](cell);
            // If the current cell is labelled UP but a sibling cell is labelled ACTIVE,
            // then this cell must be re-labelled as ACTIVE.
            if (activity == QuadtreeCellLabel::UP)
            {
                Vec2i parentCell = getParentCell(cell);

                for (int childIndex = 0; childIndex != 4; ++childIndex)
                {
                    Vec2i siblingCell = getChildCell(parentCell, childIndex);

                    if (myQuadtreeLabels[level](siblingCell) == QuadtreeCellLabel::ACTIVE)
                    {
                        myQuadtreeLabels[level](cell) = QuadtreeCellLabel::ACTIVE;
                        break;
                    }
                }
#if !defined(NDEBUG)
                // There should never be an instance where a group of silbing cells
                // contain both an UP and INACTIVE label.
                for (int childIndex = 0; childIndex != 4; ++childIndex)
                {
                    Vec2i siblingCell = getChildCell(parentCell, childIndex);
                    assert(myQuadtreeLabels[level](siblingCell) != QuadtreeCellLabel::INACTIVE);
                }
#endif
            }
            // If the cell is labelled ACTIVE, the parent must be labelled DOWN.
            else if (activity == QuadtreeCellLabel::ACTIVE)
            {
                // Label all sibling cells as active
                Vec2i parentCell = getParentCell(cell);
                localParentsDown.push_back(parentCell);
            }
        }
    });

    parentsDown.clear();
    mergeLocalThreadVectors(parentsDown, parallelParentsDown);
}

void QuadtreeGrid::setCellLabels(const std::vector<Vec2i>& cells, int level, QuadtreeCellLabel label)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, int(cells.size())), [&](const tbb::blocked_range<int>& range)
    {
        for (int index = range.begin(); index != range.end(); ++index)
        {
            myQuadtreeLabels[level](cells[index]) = label;
        }
    });
}

void QuadtreeGrid::setParentsUp(std::vector<Vec2i>& parentsUp, int level)
{
    tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelParentsUp;

    tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[level].voxelCount()), [&](const tbb::blocked_range<int>& range)
    {
        auto& localParentsUp = parallelParentsUp.local();
        for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
            Vec2i cell = myQuadtreeLabels[level].unflatten(flatIndex);
            if (myQuadtreeLabels[level](cell) == QuadtreeCellLabel::UP)
            {
                Vec2i parentCell = getParentCell(cell);

                // If the parent is uninitialized (i.e., INACTIVE), then
                // pass the "UP" label from the child.
                if (myQuadtreeLabels[level + 1](parentCell) == QuadtreeCellLabel::INACTIVE)
                {
                    localParentsUp.push_back(parentCell);
                }
                // Since the child is UP, the parent cannot be DOWN. This
                // would imply that a sibling is ACTIVE and that would mean
                // this child should be ACTIVE from the first pass. Also,
                // the parent cannot be UP since we are only just setting
                // parents to UP now. The only other option is ACTIVE since
                // we've capture the INACTIVE event above.
                else
                {
                    assert(myQuadtreeLabels[level + 1](parentCell) == QuadtreeCellLabel::ACTIVE);
                }

#if !defined(NDEBUG)
                // From the first pass, if the sibling is ACTIVE then this
                // child should already be set to ACTIVE. Because INACTIVE
                // cells are "OUTSIDE" of the simulation domain, UP cells
                // are contained inside a boundary of "ACTIVE" cells.
                // Therefore a UP child should never have an INACTIVE
                // sibling. Finally, an UP child cannot have a down sibling
                // because that would violate the face grading rule. If the
                // child is currently UP then all of its siblings must be UP
                // as well.

                for (int childIndex = 0; childIndex != 4; ++childIndex)
                {
                    Vec2i siblingCell = getChildCell(parentCell, childIndex);
                    assert(myQuadtreeLabels[level](siblingCell) == QuadtreeCellLabel::UP);
                }
#endif
            }
        }
    });

    parentsUp.clear();
    mergeLocalThreadVectors(parentsUp, parallelParentsUp);
}

void QuadtreeGrid::setTopLevel(int level)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[level].voxelCount()), [&](const tbb::blocked_range<int>& range)
    {
        for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
            Vec2i cell = myQuadtreeLabels[level].unflatten(flatIndex);
            if (myQuadtreeLabels[level](cell) == QuadtreeCellLabel::UP)
            {
                myQuadtreeLabels[level](cell) = QuadtreeCellLabel::ACTIVE;
            }
        }
    });
}

bool QuadtreeGrid::checkActiveCell(int level)
{
    bool hasActiveCell = false;
    tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[level].voxelCount()), [&](const tbb::blocked_range<int>& range)
    {
        if (hasActiveCell)
        {
            return;
        }

        for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
            Vec2i cell = myQuadtreeLabels[level].unflatten(flatIndex);
            if (myQuadtreeLabels[level](cell) == QuadtreeCellLabel::ACTIVE)
            {
                hasActiveCell = true;
                break;
            }
        }
    });

    return hasActiveCell;
}

void QuadtreeGrid::setFaceGrading(std::vector<Vec2i>& parentsDown, std::vector<Vec2i>& parentsActive, int level)
{
    tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelParentsDown;
    tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelParentsActive;
    tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[level].voxelCount()), [&](const tbb::blocked_range<int>& range)
    {
        auto& localParentsDown = parallelParentsDown.local();
        auto& localParentsActive = parallelParentsActive.local();

        for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
            Vec2i cell = myQuadtreeLabels[level].unflatten(flatIndex);

            QuadtreeCellLabel activity = myQuadtreeLabels[level](cell);

            if (activity == QuadtreeCellLabel::ACTIVE)
            {
                for (int axis : {0, 1})
                {
                    for (int direction : {0, 1})
                    {
                        Vec2i adjacentCell = cellToCell(cell, axis, direction);
                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= myQuadtreeLabels[level].size()[axis])
                        {
                            continue;
                        }

                        if (myQuadtreeLabels[level](adjacentCell) == QuadtreeCellLabel::UP)
                        {
                            Vec2i parentCell = getParentCell(adjacentCell);

                            // This adjacent cell can't be a sibling because if it was UP,
                            // then it should have be set to ACTIVE already.
                            assert(parentCell != getParentCell(cell));

                            // The parent should be INACTIVE because it should not have been
                            // written to yet (meaning the parent has no ACTIVE children).
                            assert(myQuadtreeLabels[level + 1](parentCell) == QuadtreeCellLabel::INACTIVE);

                            localParentsActive.push_back(parentCell);
                        }
                    }
                }
            }
            else if (activity == QuadtreeCellLabel::DOWN)
            {
                Vec2i parentCell = getParentCell(cell);

#if !defined(NDEBUG)
                for (int childIndex = 0; childIndex != 4; ++childIndex)
                {
                    Vec2i siblingCell = getChildCell(parentCell, childIndex);
                    assert(myQuadtreeLabels[level](siblingCell) != QuadtreeCellLabel::UP);
                }
#endif
                localParentsDown.push_back(parentCell);
            }
        }
    });
    
    parentsActive.clear();
    mergeLocalThreadVectors(parentsActive, parallelParentsActive);

    parentsDown.clear();
    mergeLocalThreadVectors(parentsDown, parallelParentsDown);
}

}