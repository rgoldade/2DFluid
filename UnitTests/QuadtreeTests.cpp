#include "gtest/gtest.h"

#include "QuadtreeGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

static double testIso(const Vec2d& pos)
{
    return std::min(pos.norm() - .5 * std::sqrt(2.) * PI, (pos - Vec2d::Constant(PI)).norm() - .5 * std::sqrt(2.) * PI);
}

static QuadtreeGrid buildTestGrid()
{
    Vec2i size(128, 128);
	double dx = PI / double(size[0]);

	Transform xform(dx, Vec2d::Zero());

    auto refiner = [dx](const Vec2d& pos)
	{
        if (std::fabs(testIso(pos)) < dx)
        {
            return 0;
        }
		
		return -1; // There is no outside of the domain in this example
	};

    int quadtreeLevels = 4;
    QuadtreeGrid quadtree(xform, size, quadtreeLevels);
    quadtree.buildTree(refiner);

    return quadtree;
}

TEST(QUADTREE_GRID_TESTS, FULL_LEVEL_TREE)
{
    QuadtreeGrid quadtree = buildTestGrid();
    ASSERT_TRUE(quadtree.levels() == 4);
}

TEST(QUADTREE_GRID_TESTS, COPY_TREE)
{
    QuadtreeGrid quadtree = buildTestGrid();
    QuadtreeGrid copytree = quadtree;

    ASSERT_TRUE(quadtree.levels() == copytree.levels());
    for (int level = 0; level < quadtree.levels(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), quadtree.size(level), [&](const Vec2i& cell)
        {
            EXPECT_EQ(quadtree.getCellLabel(cell, level), copytree.getCellLabel(cell, level));
        });
    }
}

TEST(QUADTREE_GRID_TESTS, PARENT_CHILD_CELLS)
{
    QuadtreeGrid quadtree = buildTestGrid();

    for (int level = 0; level < quadtree.levels(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), quadtree.size(level), [&](const Vec2i& cell)
        {
            std::array<Vec2i, 4> childCells = { Vec2i(2 * cell[0], 2 * cell[1]), Vec2i(2 * cell[0] + 1, 2 * cell[1]), Vec2i(2 * cell[0], 2 * cell[1] + 1), Vec2i(2 * cell[0] + 1, 2 * cell[1] + 1) };
                
            for (int childIndex = 0; childIndex < 4; ++childIndex)
            {
                Vec2i childCell = quadtree.getChildCell(cell, childIndex);
                EXPECT_EQ(childCell, childCells[childIndex]);
            }
           
            EXPECT_EQ(quadtree.getParentCell(cell), cell / 2);
        });
    }
}

TEST(QUADTREE_GRID_TESTS, PARENT_CHILD_FACES)
{
    QuadtreeGrid quadtree = buildTestGrid();

    double dx = 1.;
    for (int level = 0; level < quadtree.levels(); ++level)
    {
        Transform xform(dx * (1 << level), Vec2d::Zero());
        VectorGrid<int> dummyFaces(xform, quadtree.size(level));

        for (int axis = 0; axis < 2; ++axis)
        {
            Vec2i size = dummyFaces.size(axis);

            forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
            {
                std::array<Vec2i, 2> childFaces = { 2 * face, 2 * face };
                childFaces[1][(axis + 1) % 2] += 1;
                for (int childIndex = 0; childIndex < 2; ++childIndex)
                {
                    Vec2i childFace = quadtree.getChildFace(face, axis, childIndex);

                    EXPECT_EQ(childFace, childFaces[childIndex]);
                }

                EXPECT_EQ(quadtree.getParentFace(face), face / 2);

                Vec2i insetNode = 2 * face;
                insetNode[(axis + 1) % 2] += 1;
                EXPECT_EQ(quadtree.getInsetNode(face, axis), insetNode);
            });
        }
    }
}

TEST(QUADTREE_GRID_TESTS, PARENT_CHILD_NODES)
{
    QuadtreeGrid quadtree = buildTestGrid();

    double dx = 1.;
    for (int level = 0; level < quadtree.levels(); ++level)
    {
        Transform xform(dx * (1 << level), Vec2d::Zero());
        ScalarGrid<int> dummyNodes(xform, quadtree.size(level), ScalarGridSettings::SampleType::NODE);

        Vec2i size = dummyNodes.size();

        forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& node)
        {
            EXPECT_EQ(quadtree.getChildNode(node), 2 * node);
            EXPECT_EQ(quadtree.getParentNode(node), node / 2);
        });
    }
}

TEST(QUADTREE_GRID_TESTS, FACE_ADJACENT_CELLS)
{
    QuadtreeGrid quadtree = buildTestGrid();

    for (int level = 0; level < quadtree.levels(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), quadtree.size(level), [&](const Vec2i& cell)
        {
            if (quadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
            {
                for (int axis : {0, 1})
                {
                    for (int direction : {0, 1})
                    {
                        auto& faceAdjacentCells = quadtree.getFaceAdjacentCells(cell, axis, direction, level);

                        // Manually check adjacent cells
                        Vec2i adjacentCell = cellToCell(cell, axis, direction);

                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= quadtree.size(level)[axis])
                        {
                            EXPECT_EQ(faceAdjacentCells.size(), 0);
                        }
                        else
                        {
                            if (quadtree.getCellLabel(adjacentCell, level) == QuadtreeGrid::QuadtreeCellLabel::DOWN)
                            {
                                EXPECT_GT(level, 0);
                                EXPECT_EQ(faceAdjacentCells.size(), 2);

                                std::array<Vec2i, 2> testAdjacentCells = { 2 * adjacentCell, 2 * adjacentCell };

                                if (direction == 0)
                                {
                                    testAdjacentCells[0][axis] += 1;
                                    testAdjacentCells[1][axis] += 1;
                                }

                                testAdjacentCells[1][(axis + 1) % 2] += 1;

                                for (const auto& cells : faceAdjacentCells)
                                {
                                    EXPECT_EQ(cells.second, level - 1);
                                    EXPECT_TRUE(std::find(testAdjacentCells.begin(), testAdjacentCells.end(), cells.first) != testAdjacentCells.end());
                                    EXPECT_EQ(quadtree.getCellLabel(cells.first, cells.second), QuadtreeGrid::QuadtreeCellLabel::ACTIVE);
                                }

                            }
                            else if (quadtree.getCellLabel(adjacentCell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
                            {
                                EXPECT_EQ(faceAdjacentCells.size(), 1);
                                EXPECT_EQ(faceAdjacentCells[0].first, adjacentCell);
                                EXPECT_EQ(faceAdjacentCells[0].second, level);
                            }
                            else if (quadtree.getCellLabel(adjacentCell, level) == QuadtreeGrid::QuadtreeCellLabel::UP)
                            {
                                EXPECT_LT(level, quadtree.levels() - 1);
                                EXPECT_EQ(faceAdjacentCells.size(), 1);
                                EXPECT_EQ(faceAdjacentCells[0].first, quadtree.getParentCell(adjacentCell));
                                EXPECT_EQ(faceAdjacentCells[0].second, level + 1);
                                EXPECT_EQ(quadtree.getCellLabel(faceAdjacentCells[0].first, faceAdjacentCells[0].second), QuadtreeGrid::QuadtreeCellLabel::ACTIVE);
                            }
                        }
                    }
                }
            }
        });
    }
}

TEST(QUADTREE_GRID_TESTS, REFINE_TREE)
{
    QuadtreeGrid quadtree = buildTestGrid();
    QuadtreeGrid copytree = quadtree;

    quadtree.refineGrid();

    ASSERT_TRUE(quadtree.levels() == copytree.levels());

    for (int level = 0; level < quadtree.levels(); ++level)
    {
        EXPECT_EQ(quadtree.size(level), 2 * copytree.size(level));
        EXPECT_EQ(quadtree.xform(level).offset(), copytree.xform(level).offset());
        EXPECT_EQ(quadtree.xform(level).dx(), copytree.xform(level).dx() / 2.);

        forEachVoxelRange(Vec2i::Zero(), quadtree.size(level), [&](const Vec2i& cell)
        {
            Vec2i copyCell = quadtree.getParentCell(cell);
            EXPECT_EQ(quadtree.getCellLabel(cell, level), copytree.getCellLabel(copyCell, level));
        });

        forEachVoxelRange(Vec2i::Zero(), copytree.size(level), [&](const Vec2i& cell)
        {
            for (int childIndex = 0; childIndex < 4; ++childIndex)
            {
                Vec2i refineCell = copytree.getChildCell(cell, childIndex);
                EXPECT_EQ(quadtree.getCellLabel(refineCell, level), copytree.getCellLabel(cell, level));
            }
        });
    }
}

// An INACTIVE fine cell can only have INACTIVE or DOWN ancestors.
// An ACTIVE fine cell can only have DOWN ancestors.
// An UP fine cell has one and only one ACTIVE ancestor.
// A DOWN fine cell should never happen.

TEST(QUADTREE_GRID_TESTS, ANCESTOR_VALIDATION)
{
    QuadtreeGrid quadtree = buildTestGrid();

    forEachVoxelRange(Vec2i::Zero(), quadtree.grid(0).size(), [&](const Vec2i& cell)
    {
        if (quadtree.getCellLabel(cell, 0) == QuadtreeGrid::QuadtreeCellLabel::INACTIVE)
        {
            bool foundDownCell = false;

            Vec2i parentCell(cell);
            for (int level = 1; level < quadtree.levels(); ++level)
            {
                parentCell = quadtree.getParentCell(parentCell);
                QuadtreeGrid::QuadtreeCellLabel cellLabel = quadtree.getCellLabel(parentCell, level);

                if (cellLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)
                {
                    foundDownCell = true;
                }
                else if (cellLabel == QuadtreeGrid::QuadtreeCellLabel::INACTIVE)
                {
                    ASSERT_FALSE(foundDownCell);
                }
                else
                {
                    ASSERT_TRUE(false);
                }
            }
        }
        else if (quadtree.getCellLabel(cell, 0) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
        {
            Vec2i parentCell(cell);
            for (int level = 1; level < quadtree.levels(); ++level)
            {
                parentCell = quadtree.getParentCell(parentCell);
                QuadtreeGrid::QuadtreeCellLabel cellLabel = quadtree.getCellLabel(parentCell, level);

                if (cellLabel != QuadtreeGrid::QuadtreeCellLabel::DOWN)
                {
                    ASSERT_EQ(cellLabel, QuadtreeGrid::QuadtreeCellLabel::DOWN) << int(cellLabel) << int(QuadtreeGrid::QuadtreeCellLabel::DOWN);
                }                
            }
        }
        else if (quadtree.getCellLabel(cell, 0) == QuadtreeGrid::QuadtreeCellLabel::UP)
        {
            bool foundActiveCell = false;

            Vec2i parentCell(cell);

            for (int level = 1; level < quadtree.levels(); ++level)
            {
                parentCell = quadtree.getParentCell(parentCell);
                QuadtreeGrid::QuadtreeCellLabel cellLabel = quadtree.getCellLabel(parentCell, level);

                if (cellLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
                {
                    ASSERT_FALSE(foundActiveCell);
                    foundActiveCell = true;
                }
                else if (cellLabel == QuadtreeGrid::QuadtreeCellLabel::UP)
                {
                    ASSERT_FALSE(foundActiveCell);
                }
                else if (cellLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)
                {
                    ASSERT_TRUE(foundActiveCell);
                }
                else
                {
                    ASSERT_TRUE(false);
                }
            }
        }
        else // DOWN should not be set at the finest level
        {
            ASSERT_TRUE(false);
        }
    });
}

// An UP cell can only have an UP sibling and an ACTIVE or UP
// adjacent cell.
TEST(QUADTREE_GRID_TESTS, UP_ADJACENT)
{
    QuadtreeGrid quadtree = buildTestGrid();

    for (int level = 0; level != quadtree.levels(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), quadtree.grid(level).size(), [&](const Vec2i& cell)
        {
            if (quadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::UP)
            {

                if (level == quadtree.levels() - 1)
                {
                    std::cout << quadtree.levels() << std::endl;
                    ASSERT_TRUE(level < quadtree.levels() - 1);
                }

                Vec2i parentCell = quadtree.getParentCell(cell);

                // Check siblings
                for (int childIndex = 0; childIndex != 4; ++childIndex)
                {
                    Vec2i siblingCell = quadtree.getChildCell(parentCell, childIndex);

			        ASSERT_EQ(quadtree.getCellLabel(siblingCell, level), QuadtreeGrid::QuadtreeCellLabel::UP);
                }

                for (int axis : {0, 1})
                {
                    for (int direction : {0, 1})
                    {
                        Vec2i adjacentCell = cellToCell(cell, axis, direction);

                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= quadtree.size(level)[axis])
                        {
                            continue;
                        }

                        QuadtreeGrid::QuadtreeCellLabel adjacentLabel = quadtree.getCellLabel(adjacentCell, level);

                        ASSERT_TRUE(adjacentLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE || adjacentLabel == QuadtreeGrid::QuadtreeCellLabel::UP);
                    }
                }
            }
        });
    }
}

// If the adjacent cell is ACTIVE, things are good.
// If the adjacent cell is DOWN, then the adjacent child cells must be ACTIVE.
// It the adjacent cell is UP, then the adjacent parent must be ACTIVE.
// The adjacent cells must also reciprocate.
TEST(QUADTREE_GRID_TESTS, ACTIVE_ADJACENT_CELLS)
{
    QuadtreeGrid quadtree = buildTestGrid();

    for (int level = 0; level != quadtree.levels(); ++level)
    {
        forEachVoxelRange(Vec2i::Zero(), quadtree.grid(level).size(), [&](const Vec2i& cell)
        {
            if (quadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
            {
                for (int axis : {0, 1})
                {
                    for (int direction : {0, 1})
                    {
                        Vec2i adjacentCell = cellToCell(cell, axis, direction);

                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= quadtree.grid(level).size()[axis])
                        {
				            continue;
                        }

                        QuadtreeGrid::QuadtreeCellLabel adjacentLabel = quadtree.getCellLabel(adjacentCell, level);

                        std::vector<std::pair<Vec2i, int>> adjacentFaceCellList = quadtree.getFaceAdjacentCells(cell, axis, direction, level);

                        if (adjacentLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)
                        {
                            ASSERT_EQ(adjacentFaceCellList.size(), 2);

                            // Check that adjacent cells are ACTIVE. This
                            // enforces face grading.
                            for (const std::pair<Vec2i, int>& cellItem : adjacentFaceCellList)
                            {
                                Vec2i localCell(cellItem.first);
                                ASSERT_EQ(cellItem.second, level - 1);

                                QuadtreeGrid::QuadtreeCellLabel nephewLabel = quadtree.getCellLabel(localCell, level - 1);
                                ASSERT_EQ(nephewLabel, QuadtreeGrid::QuadtreeCellLabel::ACTIVE);
                            }  
                        }
                        // The parent activity check enforces face grading
                        else if (adjacentLabel == QuadtreeGrid::QuadtreeCellLabel::UP)
                        {
                            ASSERT_TRUE(level < quadtree.levels() - 1);

                            Vec2i parentCell = quadtree.getParentCell(adjacentCell);

                            QuadtreeGrid::QuadtreeCellLabel parentLabel = quadtree.getCellLabel(parentCell, level + 1);

                            ASSERT_EQ(parentLabel, QuadtreeGrid::QuadtreeCellLabel::ACTIVE);
                        }

                        // Check that adjacent active cells also reciprocate
                        for (const std::pair<Vec2i, int>& cellItem : adjacentFaceCellList)
                        {
                            Vec2i localCell(cellItem.first);

                            int otherDirection = (direction + 1) % 2;
                            int localLevel = cellItem.second;

                            std::vector<std::pair<Vec2i, int>> reciprocatingCellList = quadtree.getFaceAdjacentCells(localCell, axis, otherDirection, localLevel);

                            auto result = std::find(reciprocatingCellList.begin(), reciprocatingCellList.end(), std::make_pair(cell, level));

                            ASSERT_TRUE(result != reciprocatingCellList.end());
                        }                    
                    }
                }
            }
        });
    }
}