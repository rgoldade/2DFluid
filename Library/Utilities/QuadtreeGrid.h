#ifndef FLUIDSIM2D_QUADTREE_GRID_H
#define FLUIDSIM2D_QUADTREE_GRID_H

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"

///////////////////////////////////
//
// QuadtreeGrid.h
// Ryan Goldade 2017
//
////////////////////////////////////

namespace FluidSim2D
{

class QuadtreeGrid
{
public:

	enum class QuadtreeCellLabel {INACTIVE, ACTIVE, UP, DOWN};

	QuadtreeGrid(const Transform& xform, const Vec2i& size, int desiredLevels);

	// Use a refinement functor to build the grading of the tree. The functor
	// should take a Vec2d in world space and return a signed value where 0 indicates an ACTIVE cell
	// at the finest resolution, -'ve indicates an UP cell and +'ve indicates an INACTIVE cell
	template<typename Refinement>
	void buildTree(Refinement &refiner);

	int levels() const;

	Vec2i size(int level) const;

	const ScalarGrid<QuadtreeCellLabel>& grid(int level) const;

	Transform xform(int level) const;

	FORCE_INLINE QuadtreeCellLabel getCellLabel(const Vec2i& cell, int level) const
	{
		return myQuadtreeLabels[level](cell);
	}

	std::vector<std::pair<Vec2i, int>> getFaceAdjacentCells(const Vec2i& cell, int axis, int direction, int level) const;

	FORCE_INLINE Vec2i getParentCell(const Vec2i& cell) const 
	{ 
		return cell / 2;
	}

	FORCE_INLINE Vec2i getParentFace(const Vec2i& face) const
	{
		return getParentCell(face);
	}

	FORCE_INLINE Vec2i getParentNode(const Vec2i& node) const
	{
		return getParentCell(node);
	}

	FORCE_INLINE Vec2i getChildCell(const Vec2i& cell, int childIndex) const
    {
        assert(childIndex < 4);

        // Child index is a binary flag indicating a forward offset of 1
        // in each axis direction.
        Vec2i childCell = 2 * cell;
        for (int axis : {0, 1})
        {
            if (childIndex & (1 << axis))
            {
                ++childCell[axis];
            }
        }

        return childCell;
    }

	FORCE_INLINE Vec2i getChildFace(const Vec2i& face, int faceAxis, int childIndex) const
	{
		assert(faceAxis < 2 && childIndex < 2);
		Vec2i childFace = 2 * face;

		if (childIndex == 1)
		{
			childFace[(faceAxis + 1) % 2] += 1;
		}

		return childFace;
	}

	FORCE_INLINE Vec2i getChildNode(const Vec2i& node) const
	{
		return node * 2;
	}

	FORCE_INLINE Vec2i getInsetNode(const Vec2i& face, int axis) const
	{
		Vec2i node = 2 * face;
		node[(axis + 1) % 2] += 1;
		return node;
	}

	FORCE_INLINE bool isCellActive(const Vec2i& cell, int level) const 
	{
		return myQuadtreeLabels[level](cell) == QuadtreeCellLabel::ACTIVE;
	}

	void refineGrid();

	FORCE_INLINE Vec2d indexToWorld(const Vec2d& pos, int level) const
	{
		return myQuadtreeLabels[level].indexToWorld(pos);
	}

	FORCE_INLINE Vec2d worldToIndex(const Vec2d& pos, int level) const
	{
		return myQuadtreeLabels[level].worldToIndex(pos);
	}

	void drawGrid(Renderer& renderer);

	void drawCellConnections(Renderer& renderer);

private:

	template<typename Refinement>
	void setBaseGridLabels(Refinement& refiner);

	void setActiveCellsAndParentDown(std::vector<Vec2i>& parentsDown, int level);

	void setCellLabels(const std::vector<Vec2i>& cells, int level, QuadtreeCellLabel label);

	void setFaceGrading(std::vector<Vec2i>& parentsDown, std::vector<Vec2i>& parentsActivej, int level);

	void setParentsUp(std::vector<Vec2i>& parentsUp, int level);

	void setTopLevel(int level);

	bool checkActiveCell(int level);

	// Markers to label active cells, if the cell is "above"
	// or "below" the leaf (aka active cell).
	std::vector<ScalarGrid<QuadtreeCellLabel>> myQuadtreeLabels;

	Vec2i myBaseSize;

	int myQuadtreeLevels;
};

template<typename Refinement>
void QuadtreeGrid::setBaseGridLabels(Refinement& refiner)
{
	tbb::parallel_for(tbb::blocked_range<int>(0, myQuadtreeLabels[0].voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec2i cell = myQuadtreeLabels[0].unflatten(flatIndex);

			if (cell[0] >= myBaseSize[0] || cell[1] >= myBaseSize[1])
			{
				assert(myQuadtreeLabels[0](cell) == QuadtreeCellLabel::INACTIVE);
				continue;
			}

			double type = refiner(myQuadtreeLabels[0].indexToWorld(cell.cast<double>()));
			if (type == 0)
			{
				myQuadtreeLabels[0](cell) = QuadtreeCellLabel::ACTIVE;
			}
			else if (type < 0)
			{
				myQuadtreeLabels[0](cell) = QuadtreeCellLabel::UP;
			}
			else if (type > 0)
			{
				assert(myQuadtreeLabels[0](cell) == QuadtreeCellLabel::INACTIVE);
			}
		}
	});
}

template<typename Refinement>
void QuadtreeGrid::buildTree(Refinement& refiner)
{
	// Pass over the finest resolution and activate cells where the mask grid is 0. Then activate its siblings.
	setBaseGridLabels(refiner);

	std::vector<Vec2i> parentsDown;
	std::vector<Vec2i> parentsActive;
	std::vector<Vec2i> parentsUp;

	for (int level = 0; level < myQuadtreeLabels.size() - 1; ++level)
	{
		// First build pass:
		// If cell is labeled UP and its sibling is labeled ACTIVE, set it to ACTIVE.
		// If a cell is labelled ACTIVE, set parent of cell to DOWN.

		parentsDown.clear();
		setActiveCellsAndParentDown(parentsDown, level);

		setCellLabels(parentsDown, level + 1, QuadtreeCellLabel::DOWN);

		// Second pass:
		// If cell is labeled ACTIVE, any face-adjacent cells that are labeled UP
		// will have their parents listed as ACTIVE. If cell is DOWN, set parent to DOWN.

		parentsDown.clear();
		parentsActive.clear();
		setFaceGrading(parentsDown, parentsActive, level);

		setCellLabels(parentsDown, level + 1, QuadtreeCellLabel::DOWN);
		setCellLabels(parentsActive, level + 1, QuadtreeCellLabel::ACTIVE);

		// Third pass:
		// If cell is UP and the parent is INACTIVE, set parent to UP.
		parentsUp.clear();
		setParentsUp(parentsUp, level);

		setCellLabels(parentsUp, level + 1, QuadtreeCellLabel::UP);
	}

    // Clean up pass on the top level
	setTopLevel(int(myQuadtreeLabels.size() - 1));

    // Traverse the grid levels and find the lowest level without any ACTIVE
    // voxels. Since there are no active grid cells above this level, we
    // might as well set the internal maximum level to this capped level.
	int cappedLevel = 0;
    for (; cappedLevel < myQuadtreeLevels; ++cappedLevel)
    {

		bool hasActiveCell = checkActiveCell(cappedLevel);

		if (!hasActiveCell)
		{
			assert(cappedLevel != 0);
			break;
		}
    }

	myQuadtreeLabels.resize(cappedLevel);
}

//template<typename Refinement>
//bool QuadtreeGrid::unit_test_graded(Refinement &refinement) const
//{
//	unsigned maxlevels = m_markers.size();
//
//	// Make sure that there's only one active cell in each path from the finest to coarsest
//	// stack of cells.
//	{
//		Vec2i size = m_markers[0].size();
//
//		bool passed = true;
//		forEachVoxelRange(Vec2i(0), size, [&](Vec2i cell)
//		{
//			if (!passed) return;
//
//			unsigned count = 0;
//			for (unsigned level = 0; level < maxlevels; ++level)
//			{
//				if (m_markers[level](cell) == QuadtreeCellLabel::ACTIVE) ++count;
//				cell = get_parent(cell);
//			}
//
//			if (count == 0)
//			{
//				Vec2R pos = m_markers[0].indexToWorld(Vec2R(cell));
//				if (refinement(pos) <= 0)
//				{
//					passed = false;
//					return;
//				}
//
//			}
//			if (count > 1)
//			{
//				passed = false;
//				return;
//			}
//		});
//
//		if (!passed) return false;
//	}
//
//	// Make sure that an active cell's adjacent active cells reciprocate
//	for (unsigned level = 0; level < maxlevels; ++level)
//	{
//		Vec2i size = m_markers[level].size();
//
//		bool passed = true;
//		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
//		{
//			if (!passed) return;
//
//			if (m_markers[level](cell) == QuadtreeCellLabel::ACTIVE || m_markers[level](cell) == QuadtreeCellLabel::INACTIVE)
//			{
//				for (unsigned dir = 0; dir < 4; ++dir)
//				{
//					Vec2i adjcell = Vec2i(cell) + cell_to_cell[dir];
//
//					unsigned axis = dir / 2;
//					if (adjcell[axis] < 0 || adjcell[axis] >= size[axis])
//						continue;
//
//					std::vector<Vec3ui> adj_cells = get_face_adjacent_cells(cell, level, dir);
//
//					for (unsigned adj = 0; adj < adj_cells.size(); ++adj)
//					{
//						Vec2i reciprocate_cell(adj_cells[adj][0], adj_cells[adj][1]);
//
//						int offset = (dir % 2 == 0) ? 1 : -1;
//
//						std::vector<Vec3ui> return_adj_cells = get_face_adjacent_cells(reciprocate_cell, adj_cells[adj][2], dir + offset);
//
//						auto result = std::find(return_adj_cells.begin(), return_adj_cells.end(), Vec3ui(cell[0], cell[1], level));
//						if (result == return_adj_cells.end())
//						{
//							passed = false;
//							return;
//						}
//					}
//				}
//			}
//		});
//
//		if (!passed) return false;
//	}
//
//	return true;
//}

}

#endif